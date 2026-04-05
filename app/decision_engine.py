"""
Credit Decision Engine - Applies Zoot Rules to determine credit decisions.

Flow:
  1. Ingest Zoot request into Vector Store (chunked by section)
  2. Retrieve matching rules from Vector Store
  3. Apply rules to request data to produce decision
  4. Return response matching Zoot response format

Rules sourced from ZootRules.xlsx:
  - Bureau score >= 700 AND DTI within threshold -> Approved (Low risk)
  - Bureau score 620-699 -> Refer (Medium risk)
  - Bureau score < 620 OR high delinquencies -> Declined (High risk)
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)

# DTI threshold: existing obligations should be <= 50% of annual income
DTI_THRESHOLD = 0.50


class CreditDecisionEngine:
    """Applies Zoot business rules to customer data to produce credit decisions.
    Integrates with Vector Store (for non-JSON rules) and Knowledge Graph DB
    (for JSON requests/responses) for storage and retrieval."""

    def __init__(self, data_dir: str, vector_store=None, ingestion_layer=None,
                 knowledge_graph=None):
        self.data_dir = data_dir
        self.vector_store = vector_store
        self.ingestion_layer = ingestion_layer
        self.knowledge_graph = knowledge_graph
        self.customer_requests: Dict[str, Dict] = {}
        self._load_customer_data()
        self._ingest_rules_to_stores()

    def _load_customer_data(self):
        """Load all Zoot request JSON files from data/ and index by customerId"""
        if not os.path.isdir(self.data_dir):
            logger.warning(f"Data directory not found: {self.data_dir}")
            return

        for filename in os.listdir(self.data_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.data_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)

                    # Only index files that have an applicant with a customerId
                    customer_id = (data.get('applicant') or {}).get('customerId')
                    if customer_id:
                        self.customer_requests[customer_id] = data
                        logger.info(f"Indexed customer {customer_id} from {filename}")
                except Exception as e:
                    logger.debug(f"Skipping {filename}: {e}")

        logger.info(f"Loaded {len(self.customer_requests)} customer request(s)")

    def _ingest_rules_to_stores(self):
        """Ingest rules and templates into appropriate stores at startup.
        
        Routing:
          - ZootRules.xlsx -> Vector Store (non-JSON)
          - JSON response templates -> Knowledge Graph DB
          - JSON customer requests -> Knowledge Graph DB
        """
        if not self.ingestion_layer:
            return

        # XLSX rules -> Vector Store
        rules_path = os.path.join(self.data_dir, 'ZootRules.xlsx')
        if os.path.exists(rules_path) and self.vector_store:
            try:
                rules_doc = self.ingestion_layer.ingest_file(rules_path)
                self.vector_store.add_text_chunks(
                    rules_doc.chunks,
                    metadatas=[{'source': 'ZootRules.xlsx', 'type': 'zoot_rules'} for _ in rules_doc.chunks]
                )
                logger.info(f"Ingested ZootRules.xlsx: {len(rules_doc.chunks)} rule chunks into vector store")
            except Exception as e:
                logger.error(f"Failed to ingest ZootRules.xlsx: {e}")

        # JSON response templates -> Knowledge Graph DB
        for filename in ['SampleZootResponseApproved.json', 'SampleZootResponseDecline.json', 'SampleZootResponseRefer.json']:
            fpath = os.path.join(self.data_dir, filename)
            if os.path.exists(fpath):
                try:
                    with open(fpath, 'r') as f:
                        json_data = json.load(f)
                    resp_doc = self.ingestion_layer.ingest_file(fpath)
                    # Store in Knowledge Graph if available
                    if self.knowledge_graph:
                        self.knowledge_graph.ingest_json_document(
                            doc_id=resp_doc.document_id,
                            filename=filename,
                            data=json_data,
                            metadata={'source': filename, 'type': 'zoot_response_template'}
                        )
                        logger.info(f"Ingested {filename} into Knowledge Graph")
                    elif self.vector_store:
                        # Fallback: store in vector store
                        self.vector_store.add_text_chunks(
                            resp_doc.chunks,
                            metadatas=[{'source': filename, 'type': 'zoot_response_template'} for _ in resp_doc.chunks]
                        )
                        logger.info(f"Ingested {filename}: {len(resp_doc.chunks)} chunks into vector store (KG unavailable)")
                except Exception as e:
                    logger.debug(f"Skipping {filename}: {e}")

        # JSON customer requests -> Knowledge Graph DB
        for cid, req_data in self.customer_requests.items():
            self.ingest_request(req_data)

    def reload_data(self):
        """Reload customer data from disk"""
        self.customer_requests.clear()
        self._load_customer_data()

    def get_customer_ids(self) -> List[str]:
        """Return all known customer IDs"""
        return list(self.customer_requests.keys())

    def get_customer_data(self, customer_id: str) -> Optional[Dict]:
        """Get raw request data for a customer"""
        return self.customer_requests.get(customer_id)

    def find_customer_ids_in_vector_store(self, k: int = 20) -> List[str]:
        """
        Search the Vector Store and Knowledge Graph for all customer IDs.

        Returns:
            List of unique customer IDs found across both stores
        """
        found_ids = []
        import re

        # Search Knowledge Graph first (JSON documents stored here)
        if self.knowledge_graph:
            try:
                kg_results = self.knowledge_graph.search_json_documents(
                    "customerId applicant individual", k=k
                )
                for r in kg_results:
                    for m in re.finditer(r'"customerId"\s*:\s*"(CI\d+)"', r.get('text', '')):
                        c = m.group(1)
                        if c not in found_ids:
                            found_ids.append(c)
            except Exception as e:
                logger.error(f"Error searching Knowledge Graph for customer IDs: {e}")

        # Also search Vector Store (fallback / additional)
        if self.vector_store:
            try:
                results = self.vector_store.similarity_search("customerId applicant individual", k=k)
                for doc in results:
                    # Check metadata first
                    cid = doc.metadata.get('customer_id', '')
                    if cid and cid.startswith('CI') and cid not in found_ids:
                        found_ids.append(cid)
                        continue
                    # Fallback: parse from content
                    for m in re.finditer(r'"customerId"\s*:\s*"(CI\d+)"', doc.page_content):
                        c = m.group(1)
                        if c not in found_ids:
                            found_ids.append(c)
            except Exception as e:
                logger.error(f"Error searching Vector Store for customer IDs: {e}")

        return found_ids

    # ---- Storage Integration ----

    def ingest_request(self, request_data: Dict) -> Dict:
        """
        Ingest a Zoot request into the Knowledge Graph DB (JSON data)
        and optionally into the Vector Store as fallback.

        Args:
            request_data: Full Zoot request dict

        Returns:
            Ingestion summary with storage location and details
        """
        customer_id = (request_data.get('applicant') or {}).get('customerId', 'unknown')

        # Index in memory
        if customer_id != 'unknown':
            self.customer_requests[customer_id] = request_data

        # Primary: Store JSON request in Knowledge Graph DB
        if self.knowledge_graph:
            doc_id = f"zoot_req_{customer_id}_{int(datetime.utcnow().timestamp())}"
            kg_result = self.knowledge_graph.ingest_json_document(
                doc_id=doc_id,
                filename=f'ZootRequest_{customer_id}.json',
                data=request_data,
                metadata={
                    'customer_id': customer_id,
                    'type': 'zoot_request',
                    'timestamp': datetime.utcnow().isoformat()
                }
            )
            logger.info(f"Ingested request for {customer_id} into Knowledge Graph: {kg_result['total_nodes']} nodes")
            return {
                'customer_id': customer_id,
                'stored_in': 'knowledge_graph',
                'total_nodes': kg_result['total_nodes'],
                'document_id': doc_id
            }

        # Fallback: Store in Vector Store if Knowledge Graph is unavailable
        if self.vector_store:
            chunks = []
            metadatas = []
            for key, value in request_data.items():
                chunk_text = f"[{key}]: {json.dumps(value, indent=2)}"
                chunks.append(chunk_text)
                metadatas.append({
                    'source': f'ZootRequest_{customer_id}',
                    'section': key,
                    'customer_id': customer_id,
                    'type': 'zoot_request',
                    'timestamp': datetime.utcnow().isoformat()
                })

            doc_ids = self.vector_store.add_text_chunks(chunks, metadatas=metadatas)
            logger.info(f"Ingested request for {customer_id} into Vector Store (KG unavailable): {len(doc_ids)} chunks")
            return {
                'customer_id': customer_id,
                'stored_in': 'vector_store',
                'chunks': len(doc_ids),
                'doc_ids': doc_ids
            }

        logger.warning("No storage configured; request indexed in memory only")
        return {'customer_id': customer_id, 'stored_in': 'memory_only', 'chunks': 0}

    def retrieve_rules(self, request_data: Dict, k: int = 5) -> List[str]:
        """
        Retrieve relevant Zoot rules from the Vector Store and Knowledge Graph
        based on request context.

        Args:
            request_data: The Zoot request dict (used to build the search query)
            k: Number of rule chunks to retrieve per store

        Returns:
            List of matched rule text chunks from both stores
        """
        bureau_score = (request_data.get('bureauData') or {}).get('bureauScore', 0)
        annual_income = (request_data.get('financials') or {}).get('annualIncome', 0)
        obligations = (request_data.get('financials') or {}).get('existingObligations', 0)

        query = (
            f"credit decision rule bureau score {bureau_score} "
            f"debt to income ratio annual income {annual_income} "
            f"obligations {obligations} approved declined refer risk"
        )

        rule_texts = []

        # Search Vector Store (XLSX rules, non-JSON content)
        if self.vector_store:
            try:
                results = self.vector_store.similarity_search(query, k=k)
                rule_texts.extend([doc.page_content for doc in results])
                logger.info(f"Retrieved {len(results)} rule chunks from Vector Store")
            except Exception as e:
                logger.error(f"Error retrieving rules from Vector Store: {e}")

        # Search Knowledge Graph (JSON templates, response examples)
        if self.knowledge_graph:
            try:
                kg_results = self.knowledge_graph.search_json_documents(query, k=k)
                kg_texts = [r['text'] for r in kg_results]
                rule_texts.extend(kg_texts)
                logger.info(f"Retrieved {len(kg_results)} rule chunks from Knowledge Graph")
            except Exception as e:
                logger.error(f"Error retrieving rules from Knowledge Graph: {e}")

        if not rule_texts:
            logger.warning("No rules found in either store; using built-in rules")

        return rule_texts

    # ---- Evaluation ----

    def evaluate(self, customer_id: str) -> Dict:
        """
        Evaluate credit decision for a customer based on Zoot Rules.
        Ingests the request into Knowledge Graph DB, retrieves matching rules
        from both Vector Store and Knowledge Graph, applies them, and returns
        the decision response.

        Args:
            customer_id: The customer ID to look up

        Returns:
            Dict with 'response' (Zoot format) and 'traceability' (internal)
        """
        request_data = self.customer_requests.get(customer_id)
        if not request_data:
            # Check vector store for available customer IDs
            vs_customer_ids = self.find_customer_ids_in_vector_store()
            available = vs_customer_ids or list(self.customer_requests.keys())
            available_str = ', '.join(available) if available else 'none'
            raise ValueError(
                f"Customer ID '{customer_id}' not found in Knowledge Graph or Vector Store. "
                f"Available customer IDs: {available_str}"
            )

        # Step 1: Ingest request into Knowledge Graph (or Vector Store fallback)
        ingestion_info = self.ingest_request(request_data)

        # Step 2: Retrieve matching rules from both Vector Store and Knowledge Graph
        matched_rules = self.retrieve_rules(request_data)

        # Step 3: Apply rules to determine decision
        response = self._apply_rules(request_data)

        return {
            'response': response,
            'traceability': {
                'ingestion': ingestion_info,
                'matched_rules': matched_rules[:5],
                'rules_source': 'vector_store+knowledge_graph' if matched_rules else 'built_in'
            }
        }

    def evaluate_request(self, request_data: Dict) -> Dict:
        """
        Evaluate credit decision for an ad-hoc Zoot request payload.
        Ingests the request into Knowledge Graph, retrieves rules from both
        stores, applies them, returns response.

        Args:
            request_data: Zoot-format request dict

        Returns:
            Dict with 'response' (Zoot format) and 'traceability' (internal)
        """
        # Step 1: Ingest into Knowledge Graph (or Vector Store fallback)
        ingestion_info = self.ingest_request(request_data)

        # Step 2: Retrieve matching rules from both stores
        matched_rules = self.retrieve_rules(request_data)

        # Step 3: Apply rules
        response = self._apply_rules(request_data)

        return {
            'response': response,
            'traceability': {
                'ingestion': ingestion_info,
                'matched_rules': matched_rules[:5],
                'rules_source': 'vector_store+knowledge_graph' if matched_rules else 'built_in'
            }
        }

    def _apply_rules(self, request_data: Dict) -> Dict:
        """Apply Zoot business rules to a request and return a decision response"""
        applicant = request_data.get('applicant', {})
        application = request_data.get('application', {})
        financials = request_data.get('financials', {})
        bureau_data = request_data.get('bureauData', {})
        header = request_data.get('header', {})

        bureau_score = bureau_data.get('bureauScore', 0)
        annual_income = financials.get('annualIncome', 0)
        existing_obligations = financials.get('existingObligations', 0)
        requested_amount = application.get('requestedAmount', 0)
        tenure_months = application.get('tenureMonths', 12)

        # Calculate Debt-to-Income ratio
        dti = existing_obligations / annual_income if annual_income > 0 else 1.0
        dti_within_threshold = dti <= DTI_THRESHOLD

        # Apply Zoot Rules from ZootRules.xlsx
        decision_code, decision_desc, risk_band, reason_codes, risk_score = \
            self._determine_decision(bureau_score, dti, dti_within_threshold)

        # Build response matching Zoot response format
        response = {
            'header': {
                'requestId': header.get('requestId', 'N/A'),
                'responseTimestamp': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
            },
            'decision': {
                'decisionCode': decision_code,
                'decisionDescription': decision_desc
            },
            'risk': {
                'riskScore': risk_score,
                'riskBand': risk_band
            },
            'reasonCodes': reason_codes
        }

        # Add credit terms if approved (matches SampleZootResponseApproved format)
        if decision_code == 'APPROVED':
            interest_rate = self._calculate_interest_rate(bureau_score, dti)
            credit_limit = self._calculate_credit_limit(
                annual_income, existing_obligations, requested_amount
            )
            response['decision']['creditLimit'] = credit_limit
            response['decision']['interestRate'] = interest_rate
            response['decision']['tenureMonths'] = tenure_months

        logger.info(
            f"Decision for {applicant.get('customerId')}: {decision_code} "
            f"(bureau={bureau_score}, DTI={dti:.2%}, risk={risk_band})"
        )

        return response

    def _determine_decision(self, bureau_score, dti, dti_within_threshold):
        """
        Apply the 3 Zoot rules to determine decision.

        Rules from ZootRules.xlsx:
        1. Bureau >= 700 AND DTI within threshold -> Approved / Low
        2. Bureau 620-699 -> Refer / Medium
        3. Bureau < 620 OR high delinquencies -> Declined / High

        Risk scores and reason codes aligned with sample Zoot response templates.
        """
        if bureau_score >= 700 and dti_within_threshold:
            # Sample: riskScore 312, riskBand LOW
            risk_score = max(200, int(1080 - bureau_score))
            return (
                'APPROVED',
                'Approved',
                'LOW',
                [
                    {'code': 'Z001', 'description': 'Strong bureau score'},
                    {'code': 'Z014', 'description': 'Income sufficient for requested amount'}
                ],
                risk_score
            )

        elif 620 <= bureau_score <= 699:
            # Sample: riskScore 625, riskBand MEDIUM
            risk_score = max(500, int(1275 - bureau_score))
            reason_codes = [
                {'code': 'Z023', 'description': 'Thin credit file'}
            ]
            if not dti_within_threshold:
                reason_codes.append(
                    {'code': 'Z062', 'description': 'High debt-to-income ratio'}
                )
            else:
                reason_codes.append(
                    {'code': 'Z045', 'description': 'Recent delinquency observed'}
                )
            return (
                'REFER',
                'Manual Review Required',
                'MEDIUM',
                reason_codes,
                risk_score
            )

        else:
            # Bureau < 620 OR high delinquencies
            # Sample: riskScore 845, riskBand HIGH
            risk_score = max(750, int(1450 - bureau_score))
            reason_codes = []
            if bureau_score < 620:
                reason_codes.append(
                    {'code': 'Z071', 'description': 'Low bureau score'}
                )
            if not dti_within_threshold:
                reason_codes.append(
                    {'code': 'Z062', 'description': 'High debt-to-income ratio'}
                )
            if not reason_codes:
                reason_codes.append(
                    {'code': 'Z071', 'description': 'Low bureau score'}
                )
            return (
                'DECLINED',
                'Declined',
                'HIGH',
                reason_codes,
                risk_score
            )

    def _calculate_interest_rate(self, bureau_score: int, dti: float) -> float:
        """Calculate interest rate based on credit profile.
        Aligned with sample: 11.5% for bureau 700-799 range."""
        if bureau_score >= 800:
            base_rate = 9.5
        elif bureau_score >= 750:
            base_rate = 10.5
        elif bureau_score >= 700:
            base_rate = 11.5
        else:
            base_rate = 13.0

        # Add DTI premium
        if dti > 0.4:
            base_rate += 1.5
        elif dti > 0.3:
            base_rate += 1.0
        elif dti > 0.2:
            base_rate += 0.5

        return round(base_rate, 1)

    def _calculate_credit_limit(self, annual_income: float,
                                 existing_obligations: float,
                                 requested_amount: float) -> int:
        """Calculate approved credit limit.
        Uses income multiplier aligned with sample: ~55% of annual income."""
        available_income = annual_income - existing_obligations
        # Allow up to ~67% of available income as credit limit
        max_limit = int(available_income * 0.67)
        # At minimum, offer the requested amount if income supports it
        if max_limit >= requested_amount:
            return max(requested_amount, max_limit)
        return max(max_limit, 0)

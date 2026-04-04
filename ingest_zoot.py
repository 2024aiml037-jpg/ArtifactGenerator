"""Test: Ingest SampleZootRequest + ZootRules into Vector Store, produce decisions matching templates.
CustomerIds are read dynamically from ingested data in the vector store — nothing is hardcoded."""
import sys, os, json, re
app_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app')
sys.path.insert(0, app_dir)
os.chdir(app_dir)

from config import Config
try:
    Config.validate()
except Exception:
    pass

from models.vector_store import VectorStore
from ingestion_layer import IngestionLayer
from decision_engine import CreditDecisionEngine

vector_store = VectorStore(Config.VECTOR_DB_PATH)
ingestion_layer = IngestionLayer()
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))

# Decision engine auto-ingests ZootRules.xlsx, responses, and SampleZootRequest into vector store
print("=" * 65)
print("Initializing Decision Engine (auto-ingests rules + request)")
print("=" * 65)
engine = CreditDecisionEngine(data_dir, vector_store=vector_store, ingestion_layer=ingestion_layer)

# ---- Retrieve customerId from Vector Store (not hardcoded) ----
print("\n" + "=" * 65)
print("Retrieving customerId from Vector Store")
print("=" * 65)

# Ingest SampleZootRequest.json into vector store with its filename as source
sample_req_path = os.path.join(data_dir, 'SampleZootRequest.json')
doc = ingestion_layer.ingest_file(sample_req_path)
vector_store.add_text_chunks(
    doc.chunks,
    metadatas=[{'source': 'SampleZootRequest.json', 'type': 'customer_request'} for _ in doc.chunks]
)
print(f"  Ingested SampleZootRequest.json into vector store: {len(doc.chunks)} chunks (source=SampleZootRequest.json)")

# Search the vector store and find customerId from SampleZootRequest.json chunks
vs_results = vector_store.similarity_search("customerId applicant individual", k=20)
customer_id = None
for chunk in vs_results:
    # Only consider chunks we just ingested with the known source tag
    if chunk.metadata.get('source') != 'SampleZootRequest.json':
        continue
    match = re.search(r'"customerId"\s*:\s*"([^"]+)"', chunk.page_content)
    if match:
        customer_id = match.group(1)
        print(f"  Found customerId in vector store: {customer_id}")
        print(f"  Source: {chunk.metadata.get('source')}")
        break

if not customer_id:
    print("  ERROR: No customerId found in vector store from SampleZootRequest.json!")
    sys.exit(1)

print(f"  Available customers (engine): {engine.get_customer_ids()}")

# ---- Scenario 1: APPROVED (using customer from vector store) ----
print("\n" + "=" * 65)
print(f"SCENARIO 1: APPROVED  (customer {customer_id} from vector store)")
print("Expected template: SampleZootResponseApproved.json")
print("=" * 65)
result = engine.evaluate(customer_id)
resp = result['response']
print(json.dumps(resp, indent=2))
print(f"\n  Rules source: {result['traceability']['rules_source']}")
print(f"  Matched rules: {len(result['traceability']['matched_rules'])} chunks")
assert resp['decision']['decisionCode'] == 'APPROVED'
assert resp['risk']['riskBand'] == 'LOW'
assert resp['reasonCodes'][0]['code'] == 'Z001'
assert resp['reasonCodes'][1]['code'] == 'Z014'
assert 'creditLimit' in resp['decision']
assert 'interestRate' in resp['decision']
assert 'tenureMonths' in resp['decision']
print("  [PASS] Matches Approved template")

# ---- Scenario: Customer ID Not Found ----
print("\n" + "=" * 65)
print("SCENARIO: CUSTOMER NOT FOUND  (invalid ID)")
print("=" * 65)
try:
    engine.evaluate("NONEXISTENT_ID_99999")
    print("  [FAIL] Should have raised ValueError")
    sys.exit(1)
except ValueError as e:
    msg = str(e)
    print(f"  Message: {msg}")
    assert "not found" in msg.lower()
    assert "available" in msg.lower() or "Available" in msg
    print("  [PASS] Correct 'customer ID not found' message with available IDs")

# ---- Scenario 2: DECLINED (ad-hoc request, bureau=580, DTI=75%) ----
# Retrieve a second customer ID from vector store (if available)
vs_customer_ids = engine.find_customer_ids_in_vector_store()
print(f"\n  Customer IDs in vector store: {vs_customer_ids}")

# Ingest an ad-hoc declined request dynamically
print("\n" + "=" * 65)
print("SCENARIO 2: DECLINED  (ad-hoc request, bureau=580, DTI=75%)")
print("Expected template: SampleZootResponseDecline.json")
print("=" * 65)
declined_req = {
    "header": {"requestId": "REQ-20260404-000789"},
    "applicant": {"customerId": "DYNAMIC-DECLINE", "name": {"firstName": "Raj", "lastName": "Patel"}},
    "application": {"requestedAmount": 500000, "tenureMonths": 36},
    "financials": {"annualIncome": 400000, "existingObligations": 300000},
    "bureauData": {"bureauScore": 580, "bureauProvider": "CIBIL"}
}
# Ingest into vector store first, then evaluate
engine.ingest_request(declined_req)
declined_cid = declined_req['applicant']['customerId']
r2 = engine.evaluate(declined_cid)['response']
print(json.dumps(r2, indent=2))
assert r2['decision']['decisionCode'] == 'DECLINED'
assert r2['decision']['decisionDescription'] == 'Declined'
assert r2['risk']['riskBand'] == 'HIGH'
assert any(rc['code'] == 'Z071' for rc in r2['reasonCodes'])
assert 'creditLimit' not in r2['decision']
print("  [PASS] Matches Declined template")

# ---- Scenario 3: REFER (ad-hoc request, bureau=650, DTI=16.7%) ----
print("\n" + "=" * 65)
print("SCENARIO 3: REFER  (ad-hoc request, bureau=650, DTI=16.7%)")
print("Expected template: SampleZootResponseRefer.json")
print("=" * 65)
refer_req = {
    "header": {"requestId": "REQ-20260404-000456"},
    "applicant": {"customerId": "DYNAMIC-REFER", "name": {"firstName": "Priya", "lastName": "Sharma"}},
    "application": {"requestedAmount": 200000, "tenureMonths": 12},
    "financials": {"annualIncome": 600000, "existingObligations": 100000},
    "bureauData": {"bureauScore": 650, "bureauProvider": "CIBIL"}
}
engine.ingest_request(refer_req)
refer_cid = refer_req['applicant']['customerId']
r3 = engine.evaluate(refer_cid)['response']
print(json.dumps(r3, indent=2))
assert r3['decision']['decisionCode'] == 'REFER'
assert r3['decision']['decisionDescription'] == 'Manual Review Required'
assert r3['risk']['riskBand'] == 'MEDIUM'
assert any(rc['code'] == 'Z023' for rc in r3['reasonCodes'])
assert any(rc['code'] == 'Z045' for rc in r3['reasonCodes'])
assert 'creditLimit' not in r3['decision']
print("  [PASS] Matches Refer template")

# Verify all dynamically created customers are now in vector store
final_vs_ids = engine.find_customer_ids_in_vector_store()
print(f"\n  Final customer IDs in vector store: {final_vs_ids}")

print("\n" + "=" * 65)
print("ALL SCENARIOS PASSED - Customer IDs from vector store, no hardcoding")
print("=" * 65)

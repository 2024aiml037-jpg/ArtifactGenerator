"""
AI Extraction Layer - Extracts requirements, rules, and entities using LLMs
"""

import logging
from typing import List, Dict, Any
from datetime import datetime
import json

from models.schemas import (
    ExtractedEntity, ExtractionResult, EntityType, 
    Traceability, Metadata, ConfidenceLevel, IngestedDocument
)

logger = logging.getLogger(__name__)


class ExtractionLayer:
    """Handles AI-powered extraction of entities from documents"""

    def __init__(self, llm_service):
        """
        Initialize extraction layer with an LLM service
        
        Args:
            llm_service: LLMService instance for making LLM calls
        """
        self.llm_service = llm_service
        self.entity_counter = 0

    def extract_requirements(self, content: str, doc_metadata: Metadata) -> List[ExtractedEntity]:
        """Extract requirement entities from content"""
        prompt = f"""Analyze the following document and extract all functional and non-functional requirements.
For each requirement, provide:
1. Requirement ID (REQ-XXX format)
2. Requirement text
3. Type (Functional/Non-Functional)
4. Priority level (High/Medium/Low)

Document:
{content}

Return as JSON array with objects containing: id, text, type, priority"""

        try:
            response = self.llm_service.invoke_extraction(prompt)
            requirements = self._parse_entities_response(response, doc_metadata, EntityType.REQUIREMENT)
            logger.info(f"Extracted {len(requirements)} requirements from document")
            return requirements
        except Exception as e:
            logger.error(f"Error extracting requirements: {str(e)}")
            raise

    def extract_rules(self, content: str, doc_metadata: Metadata) -> List[ExtractedEntity]:
        """Extract business rules and constraints from content"""
        prompt = f"""Analyze the following document and extract all business rules, constraints, and policies.
For each rule, provide:
1. Rule ID (RULE-XXX format)
2. Rule text
3. Rule type (Business Rule/Constraint/Policy)
4. Applicable domain/scope

Document:
{content}

Return as JSON array with objects containing: id, text, type, domain"""

        try:
            response = self.llm_service.invoke_extraction(prompt)
            rules = self._parse_entities_response(response, doc_metadata, EntityType.RULE)
            logger.info(f"Extracted {len(rules)} rules from document")
            return rules
        except Exception as e:
            logger.error(f"Error extracting rules: {str(e)}")
            raise

    def extract_entities(self, content: str, doc_metadata: Metadata) -> List[ExtractedEntity]:
        """Extract domain entities and data models from content"""
        prompt = f"""Analyze the following document and extract all domain entities, data models, and important concepts.
For each entity, provide:
1. Entity ID (ENT-XXX format)
2. Entity name
3. Description
4. Attributes/properties (if mentioned)
5. Relationships to other entities (if mentioned)

Document:
{content}

Return as JSON array with objects containing: id, name, description, attributes, relationships"""

        try:
            response = self.llm_service.invoke_extraction(prompt)
            entities = self._parse_entities_response(response, doc_metadata, EntityType.ENTITY)
            logger.info(f"Extracted {len(entities)} entities from document")
            return entities
        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}")
            raise

    def extract_apis(self, content: str, doc_metadata: Metadata) -> List[ExtractedEntity]:
        """Extract API specifications and integrations"""
        prompt = f"""Analyze the following document and extract API specifications, endpoints, and integrations.
For each API, provide:
1. API ID (API-XXX format)
2. API name/endpoint
3. HTTP method and path
4. Input parameters
5. Output/response format
6. Authentication requirements

Document:
{content}

Return as JSON array with objects containing: id, name, endpoint, method, parameters, response, auth"""

        try:
            response = self.llm_service.invoke_extraction(prompt)
            apis = self._parse_entities_response(response, doc_metadata, EntityType.API)
            logger.info(f"Extracted {len(apis)} APIs from document")
            return apis
        except Exception as e:
            logger.error(f"Error extracting APIs: {str(e)}")
            raise

    def extract_all_entities(self, ingested_doc: IngestedDocument) -> ExtractionResult:
        """
        Extract all types of entities from an ingested document
        
        Args:
            ingested_doc: IngestedDocument to extract from
            
        Returns:
            ExtractionResult with all extracted entities
        """
        extracted_entities = []
        errors = []
        warnings = []

        try:
            # Extract different entity types
            try:
                extracted_entities.extend(self.extract_requirements(ingested_doc.content, ingested_doc.metadata))
            except Exception as e:
                errors.append(f"Requirement extraction failed: {str(e)}")

            try:
                extracted_entities.extend(self.extract_rules(ingested_doc.content, ingested_doc.metadata))
            except Exception as e:
                errors.append(f"Rule extraction failed: {str(e)}")

            try:
                extracted_entities.extend(self.extract_entities(ingested_doc.content, ingested_doc.metadata))
            except Exception as e:
                errors.append(f"Entity extraction failed: {str(e)}")

            try:
                extracted_entities.extend(self.extract_apis(ingested_doc.content, ingested_doc.metadata))
            except Exception as e:
                warnings.append(f"API extraction failed: {str(e)}")

            # Add design patterns if detected
            try:
                design_entities = self._extract_design_patterns(ingested_doc.content, ingested_doc.metadata)
                extracted_entities.extend(design_entities)
            except Exception as e:
                warnings.append(f"Design pattern extraction failed: {str(e)}")

            logger.info(f"Extraction completed. Total entities: {len(extracted_entities)}")

            return ExtractionResult(
                document_id=ingested_doc.document_id,
                extracted_entities=extracted_entities,
                errors=errors,
                warnings=warnings
            )

        except Exception as e:
            logger.error(f"Fatal error during extraction: {str(e)}")
            raise

    def _parse_entities_response(self, response: str, doc_metadata: Metadata, entity_type: EntityType) -> List[ExtractedEntity]:
        """
        Parse LLM response into ExtractedEntity objects
        
        Args:
            response: LLM response text
            doc_metadata: Metadata of source document
            entity_type: Type of entity being extracted
            
        Returns:
            List of ExtractedEntity objects
        """
        entities = []
        
        try:
            # Try to parse as JSON
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]
            
            parsed = json.loads(json_str) if isinstance(json_str, str) else json_str
            
            if not isinstance(parsed, list):
                parsed = [parsed]
            
            for item in parsed:
                entity_id = f"{entity_type.value}_{self.entity_counter}"
                self.entity_counter += 1
                
                # Determine confidence based on response structure
                confidence = 0.9 if all(k in item for k in ['id', 'text']) else 0.7
                
                traceability = Traceability(
                    source_id=doc_metadata.filename or "unknown",
                    source_metadata=doc_metadata,
                    confidence_score=confidence,
                    confidence_level=self._get_confidence_level(confidence),
                    extraction_method="llm"
                )
                
                entity = ExtractedEntity(
                    id=entity_id,
                    type=entity_type,
                    text=item.get('text', item.get('name', str(item))),
                    description=item.get('description'),
                    traceability=traceability,
                    attributes=item
                )
                
                entities.append(entity)
                
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response, creating single entity: {str(e)}")
            entity_id = f"{entity_type.value}_{self.entity_counter}"
            self.entity_counter += 1
            
            traceability = Traceability(
                source_id=doc_metadata.filename or "unknown",
                source_metadata=doc_metadata,
                confidence_score=0.5,
                confidence_level=ConfidenceLevel.LOW,
                extraction_method="llm"
            )
            
            entity = ExtractedEntity(
                id=entity_id,
                type=entity_type,
                text=response[:500],  # Use first 500 chars
                traceability=traceability
            )
            entities.append(entity)
        
        return entities

    def _extract_design_patterns(self, content: str, doc_metadata: Metadata) -> List[ExtractedEntity]:
        """Extract design patterns mentioned in document"""
        prompt = f"""Identify design patterns, architectural patterns, or technology choices mentioned.
Document:
{content}

Return as JSON array."""

        try:
            response = self.llm_service.invoke_extraction(prompt)
            return self._parse_entities_response(response, doc_metadata, EntityType.DESIGN)
        except Exception as e:
            logger.warning(f"Design pattern extraction failed: {str(e)}")
            return []

    def _get_confidence_level(self, score: float) -> ConfidenceLevel:
        """Map confidence score to confidence level"""
        if score >= 0.8:
            return ConfidenceLevel.HIGH
        elif score >= 0.6:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW

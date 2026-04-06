"""
Validation Engine - Detects conflicts, gaps, and inconsistencies in knowledge
"""

import logging
from typing import List, Dict
from datetime import datetime

from models.schemas import (
    NormalizedEntity, Conflict, ValidationResult, EntityType
)

logger = logging.getLogger(__name__)


class ValidationEngine:
    """Handles validation and conflict detection"""

    def __init__(self, llm_service=None):
        """
        Initialize validation engine
        
        Args:
            llm_service: Optional LLM service for semantic conflict detection
        """
        self.llm_service = llm_service
        self.conflict_counter = 0

    def validate(self, normalized_entities: List[NormalizedEntity], 
                 knowledge_graph_edges: List[Dict] = None) -> ValidationResult:
        """
        Validate normalized entities and detect conflicts
        
        Args:
            normalized_entities: List of normalized entities
            knowledge_graph_edges: Optional graph edges for relationship validation
            
        Returns:
            ValidationResult with detected issues
        """
        conflicts = []
        gaps = []
        inconsistencies = []

        # Basic validation checks
        conflicts.extend(self._detect_contradictions(normalized_entities))
        gaps.extend(self._detect_gaps(normalized_entities))
        inconsistencies.extend(self._detect_inconsistencies(normalized_entities))

        # Graph-based validation if edges provided
        if knowledge_graph_edges:
            conflicts.extend(self._validate_relationships(normalized_entities, knowledge_graph_edges))

        # Calculate validation score with weighted penalties
        # Conflicts are most severe, gaps are minor, inconsistencies are moderate
        high_severity_conflicts = len([c for c in conflicts if c.severity == "high"])
        other_conflicts = len(conflicts) - high_severity_conflicts
        penalty = (high_severity_conflicts * 0.15) + (other_conflicts * 0.08) + (len(gaps) * 0.03) + (len(inconsistencies) * 0.05)
        validation_score = max(0.0, min(1.0, 1.0 - penalty))

        is_valid = len([c for c in conflicts if c.severity == "high"]) == 0

        logger.info(f"Validation completed: {len(conflicts)} conflicts, {len(gaps)} gaps, "
                   f"{len(inconsistencies)} inconsistencies. Score: {validation_score:.2f}")

        return ValidationResult(
            document_id="batch",  # TODO: track document_id through pipeline
            is_valid=is_valid,
            conflicts=conflicts,
            gaps=gaps,
            inconsistencies=inconsistencies,
            validation_score=validation_score
        )

    def _detect_contradictions(self, entities: List[NormalizedEntity]) -> List[Conflict]:
        """Detect contradictory statements"""
        contradictions = []

        # Look for conflicting requirements
        requirements = [e for e in entities if e.entity_type == EntityType.REQUIREMENT]
        
        for i, req1 in enumerate(requirements):
            for req2 in requirements[i + 1:]:
                if self._are_contradictory(req1.canonical_text, req2.canonical_text):
                    conflict = Conflict(
                        conflict_id=f"conflict_{self.conflict_counter}",
                        type="contradiction",
                        involved_entities=[req1.canonical_id, req2.canonical_id],
                        description=f"Contradictory requirements: '{req1.canonical_text}' vs '{req2.canonical_text}'",
                        severity="high"
                    )
                    contradictions.append(conflict)
                    self.conflict_counter += 1

        return contradictions

    def _detect_gaps(self, entities: List[NormalizedEntity]) -> List[str]:
        """Detect missing or incomplete information"""
        gaps = []

        # Check for entities without descriptions
        for entity in entities:
            # Only flag truly empty or trivially short descriptions
            if len(entity.canonical_text.strip()) < 5:
                gaps.append(f"Entity '{entity.canonical_id}' has very short description (possible incomplete extraction)")

        # Check for isolated entities — only flag if majority are isolated
        entities_with_relations = set()
        for entity in entities:
            if entity.unified_attributes.get('related_entities'):
                entities_with_relations.add(entity.canonical_id)

        isolated = [e.canonical_id for e in entities if e.canonical_id not in entities_with_relations 
                   and e.entity_type not in [EntityType.DESIGN, EntityType.API]]
        
        if isolated and len(isolated) > len(entities) * 0.5:
            gaps.append(f"Found {len(isolated)} isolated entities without relationships")

        return gaps

    def _detect_inconsistencies(self, entities: List[NormalizedEntity]) -> List[str]:
        """Detect logical inconsistencies"""
        inconsistencies = []

        # Check for version mismatches
        versions_set = set()
        for entity in entities:
            version = entity.traceability[0].source_metadata.version if entity.traceability else "unknown"
            versions_set.add(version)

        if len(versions_set) > 1:
            inconsistencies.append(f"Multiple versions detected: {versions_set}")

        # Check for entities with conflicting confidence levels
        low_confidence_entities = [
            e for e in entities 
            if any(t.confidence_score < 0.6 for t in e.traceability)
        ]
        
        if len(low_confidence_entities) > len(entities) * 0.3:  # More than 30% low confidence
            inconsistencies.append(f"High proportion of low-confidence extractions: {len(low_confidence_entities)}/{len(entities)}")

        return inconsistencies

    def _validate_relationships(self, entities: List[NormalizedEntity], 
                               edges: List[Dict]) -> List[Conflict]:
        """Validate knowledge graph relationships"""
        conflicts = []

        entity_ids = {e.canonical_id for e in entities}

        for edge in edges:
            source = edge.get('source')
            target = edge.get('target')
            
            # Check for dangling references
            if source and source not in entity_ids:
                conflicts.append(Conflict(
                    conflict_id=f"conflict_{self.conflict_counter}",
                    type="gap",
                    involved_entities=[target] if target else [],
                    description=f"Dangling reference to non-existent entity: {source}",
                    severity="medium"
                ))
                self.conflict_counter += 1
            
            if target and target not in entity_ids:
                conflicts.append(Conflict(
                    conflict_id=f"conflict_{self.conflict_counter}",
                    type="gap",
                    involved_entities=[source] if source else [],
                    description=f"Dangling reference to non-existent entity: {target}",
                    severity="medium"
                ))
                self.conflict_counter += 1

        return conflicts

    def _are_contradictory(self, text1: str, text2: str) -> bool:
        """
        Detect if two statements are contradictory
        Uses simple heuristic rules
        """
        text1_lower = text1.lower()
        text2_lower = text2.lower()

        # Simple contradiction patterns
        contradictory_pairs = [
            ("must", "must not"),
            ("required", "not required"),
            ("mandatory", "optional"),
            ("always", "never"),
            ("enabled", "disabled"),
            ("on", "off"),
        ]

        for positive, negative in contradictory_pairs:
            if positive in text1_lower and negative in text2_lower:
                # Check if they refer to the same thing — require strong overlap
                words1 = set(text1_lower.split()) - {"the", "a", "an", "is", "are", "to", "and", "or", "of", "in", "for", "with", "be", "shall", "should", "will"}
                words2 = set(text2_lower.split()) - {"the", "a", "an", "is", "are", "to", "and", "or", "of", "in", "for", "with", "be", "shall", "should", "will"}
                shared_words = words1 & words2
                if len(shared_words) > 3:  # Require 4+ meaningful shared words
                    return True

        # Use LLM for semantic contradiction if available
        if self.llm_service:
            try:
                prompt = f"Are these two statements contradictory? 1) {text1} 2) {text2} Answer: YES or NO"
                response = self.llm_service.invoke_for_validation(prompt)
                return "YES" in response.upper()
            except Exception as e:
                logger.warning(f"LLM contradiction check failed: {e}")
                return False

        return False

    def resolve_conflict(self, conflict_id: str, resolution: str, feedback: str = ""):
        """
        Mark a conflict as resolved
        
        Args:
            conflict_id: ID of the conflict
            resolution: How the conflict was resolved
            feedback: Optional feedback for future improvements
        """
        logger.info(f"Conflict {conflict_id} resolved: {resolution}")
        # TODO: Store resolution in database for feedback loop

    def get_validation_report(self, result: ValidationResult) -> Dict:
        """Generate a validation report"""
        return {
            "is_valid": result.is_valid,
            "validation_score": result.validation_score,
            "total_issues": len(result.conflicts) + len(result.gaps) + len(result.inconsistencies),
            "conflicts": len(result.conflicts),
            "gaps": len(result.gaps),
            "inconsistencies": len(result.inconsistencies),
            "high_severity_issues": sum(1 for c in result.conflicts if c.severity == "high"),
            "medium_severity_issues": sum(1 for c in result.conflicts if c.severity == "medium"),
            "low_severity_issues": sum(1 for c in result.conflicts if c.severity == "low"),
            "conflict_details": [
                {
                    "id": c.conflict_id,
                    "type": c.type,
                    "severity": c.severity,
                    "description": c.description
                }
                for c in result.conflicts
            ]
        }

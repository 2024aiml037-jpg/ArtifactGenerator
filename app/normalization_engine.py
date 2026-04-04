"""
Normalization Engine - Deduplicates extracted entities using embeddings and semantic similarity
"""

import logging
from typing import List, Dict
from datetime import datetime

import numpy as np
from sentence_transformers import SentenceTransformer

from models.schemas import (
    ExtractedEntity, NormalizedEntity, NormalizationResult
)

logger = logging.getLogger(__name__)


class NormalizationEngine:
    """Handles entity deduplication and normalization"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize normalization engine with embedding model
        
        Args:
            model_name: HuggingFace model for embeddings (default: lightweight MiniLM)
        """
        try:
            self.embedder = SentenceTransformer(model_name)
            logger.info(f"Loaded embedding model: {model_name}")
        except Exception as e:
            logger.warning(f"Failed to load embedding model {model_name}: {e}. Using basic deduplication.")
            self.embedder = None
        
        self.similarity_threshold = 0.85  # Threshold for considering entities as duplicates
        self.entity_counter = 0

    def normalize(self, extracted_entities: List[ExtractedEntity]) -> NormalizationResult:
        """
        Normalize and deduplicate entities
        
        Args:
            extracted_entities: List of extracted entities
            
        Returns:
            NormalizationResult with deduplicated entities
        """
        if not extracted_entities:
            return NormalizationResult(
                normalized_entities=[],
                duplicates_removed=0,
                merges_performed=0
            )

        # Group entities by type
        entities_by_type = {}
        for entity in extracted_entities:
            if entity.type not in entities_by_type:
                entities_by_type[entity.type] = []
            entities_by_type[entity.type].append(entity)

        # Normalize each group
        normalized_all = []
        total_duplicates = 0
        total_merges = 0

        for entity_type, entities in entities_by_type.items():
            normalized, duplicates, merges = self._normalize_entity_group(entities)
            normalized_all.extend(normalized)
            total_duplicates += duplicates
            total_merges += merges

        logger.info(f"Normalization: {total_duplicates} duplicates removed, {total_merges} merges performed")

        return NormalizationResult(
            normalized_entities=normalized_all,
            duplicates_removed=total_duplicates,
            merges_performed=total_merges
        )

    def _normalize_entity_group(self, entities: List[ExtractedEntity]) -> tuple:
        """
        Normalize a group of same-type entities
        
        Returns:
            Tuple of (normalized_entities, duplicates_removed, merges_performed)
        """
        if not entities:
            return [], 0, 0

        # Build similarity matrix
        similarity_matrix = self._build_similarity_matrix(entities)

        # Identify clusters of similar entities
        clusters = self._identify_clusters(similarity_matrix)

        # Create canonical entities from clusters
        normalized_entities = []
        merge_count = 0
        duplicate_count = 0

        for cluster in clusters:
            if len(cluster) == 1:
                # No duplicates, just normalize
                entity = entities[cluster[0]]
                normalized = self._create_normalized_entity([entity])
                normalized_entities.append(normalized)
            else:
                # Merge duplicates
                cluster_entities = [entities[i] for i in cluster]
                normalized = self._create_normalized_entity(cluster_entities)
                normalized_entities.append(normalized)
                
                merge_count += 1
                duplicate_count += len(cluster) - 1

        return normalized_entities, duplicate_count, merge_count

    def _build_similarity_matrix(self, entities: List[ExtractedEntity]) -> np.ndarray:
        """Build similarity matrix between entities"""
        n = len(entities)
        similarity_matrix = np.eye(n)

        if self.embedder is None:
            # Use simple string similarity fallback
            for i in range(n):
                for j in range(i + 1, n):
                    sim = self._string_similarity(entities[i].text, entities[j].text)
                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim
        else:
            # Use embeddings
            texts = [e.text for e in entities]
            embeddings = self.embedder.encode(texts)
            
            for i in range(n):
                for j in range(i + 1, n):
                    sim = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                    )
                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim

        return similarity_matrix

    def _string_similarity(self, text1: str, text2: str) -> float:
        """Simple string similarity using character overlap"""
        if not text1 or not text2:
            return 0.0
        
        set1 = set(text1.lower().split())
        set2 = set(text2.lower().split())
        
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0

    def _identify_clusters(self, similarity_matrix: np.ndarray) -> List[List[int]]:
        """
        Identify clusters of similar entities using greedy clustering
        
        Returns:
            List of clusters (each cluster is a list of entity indices)
        """
        n = len(similarity_matrix)
        visited = [False] * n
        clusters = []

        for i in range(n):
            if visited[i]:
                continue
            
            cluster = [i]
            visited[i] = True
            
            for j in range(i + 1, n):
                if not visited[j] and similarity_matrix[i, j] >= self.similarity_threshold:
                    cluster.append(j)
                    visited[j] = True
            
            clusters.append(cluster)

        return clusters

    def _create_normalized_entity(self, entities: List[ExtractedEntity]) -> NormalizedEntity:
        """
        Create a canonical normalized entity from a group of similar entities
        
        Args:
            entities: List of entities to merge
            
        Returns:
            NormalizedEntity with merged information
        """
        # Use longest text as canonical
        canonical_entity = max(entities, key=lambda e: len(e.text))
        
        canonical_id = f"norm_{self.entity_counter}"
        self.entity_counter += 1
        
        # Collect all traceability info
        traceability_list = [e.traceability for e in entities]
        
        # Merge attributes
        unified_attributes = {}
        for entity in entities:
            unified_attributes.update(entity.attributes)
        
        # Collect synonyms from non-canonical entities
        synonyms = [e.text for e in entities if e.text != canonical_entity.text]
        
        # Collect all related entities
        all_related = set()
        for entity in entities:
            all_related.update(entity.related_entities)
        
        return NormalizedEntity(
            canonical_id=canonical_id,
            canonical_text=canonical_entity.text,
            entity_type=canonical_entity.type,
            merged_from=[e.id for e in entities if e.id != canonical_entity.id],
            traceability=traceability_list,
            unified_attributes=unified_attributes,
            synonyms=synonyms
        )

    def update_similarity_threshold(self, threshold: float):
        """Update similarity threshold for deduplication"""
        if 0.0 <= threshold <= 1.0:
            self.similarity_threshold = threshold
            logger.info(f"Updated similarity threshold to {threshold}")
        else:
            raise ValueError("Threshold must be between 0.0 and 1.0")

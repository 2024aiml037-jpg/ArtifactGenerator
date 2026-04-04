"""
Enhanced Vector Store with Traceability and Metadata Support
"""

import chromadb
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from models.schemas import NormalizedEntity, Traceability

logger = logging.getLogger(__name__)


class VectorStore:
    """Enhanced vector store with traceability and metadata"""
    
    def __init__(self, path: str):
        """
        Initialize vector store
        
        Args:
            path: Path for persisting vector database
        """
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = Chroma(
            persist_directory=path,
            embedding_function=self.embeddings
        )
        self.entity_metadata = {}  # Store traceability metadata
        self.document_index = {}  # Map documents to source
        self.embedding_cache = {}  # Cache embeddings for deduplication

    def add_documents(self, documents):
        """Add documents to vector store (backward compatible)"""
        self.vector_store.add_documents(documents)
        logger.debug(f"Added {len(documents)} documents to vector store")

    def add_text_chunks(self, chunks: List[str], metadatas: Optional[List[Dict]] = None) -> List[str]:
        """
        Add raw text chunks to vector store
        
        Args:
            chunks: List of text strings to add
            metadatas: Optional list of metadata dictionaries
            
        Returns:
            List of document IDs added
        """
        if not chunks:
            return []
        
        try:
            doc_ids = self.vector_store.add_texts(
                texts=chunks,
                metadatas=metadatas or [{"timestamp": datetime.utcnow().isoformat()} for _ in chunks]
            )
            logger.debug(f"Added {len(chunks)} text chunks to vector store")
            return doc_ids
        except Exception as e:
            logger.error(f"Error adding text chunks: {str(e)}")
            raise

    def add_normalized_entities(self, entities: List[NormalizedEntity]) -> List[str]:
        """
        Add normalized entities with full traceability
        
        Args:
            entities: List of normalized entities
            
        Returns:
            List of document IDs added
        """
        doc_ids = []
        
        for entity in entities:
            # Create document with metadata
            doc_id = f"entity_{entity.canonical_id}"
            
            # Include traceability in document content for semantic search
            traceability_summary = self._create_traceability_summary(entity.traceability)
            
            content = f"""
            Entity: {entity.canonical_text}
            Type: {entity.entity_type.value}
            Synonyms: {', '.join(entity.synonyms) if entity.synonyms else 'None'}
            Attributes: {self._serialize_attributes(entity.unified_attributes)}
            Sources: {traceability_summary}
            Confidence: {self._get_avg_confidence(entity.traceability)}
            """
            
            # Create metadata dictionary
            metadata = {
                "entity_id": entity.canonical_id,
                "entity_type": entity.entity_type.value,
                "confidence": self._get_avg_confidence(entity.traceability),
                "source_count": len(entity.traceability),
                "synonyms": ",".join(entity.synonyms) if entity.synonyms else "",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Add to vector store
            try:
                self.vector_store.add_texts(
                    texts=[content],
                    metadatas=[metadata],
                    ids=[doc_id]
                )
                
                # Store traceability information
                self.entity_metadata[doc_id] = {
                    "entity": entity,
                    "traceability": entity.traceability,
                    "metadata": metadata,
                    "added_at": datetime.utcnow().isoformat()
                }
                
                doc_ids.append(doc_id)
                logger.debug(f"Added entity {entity.canonical_id} to vector store with traceability")
                
            except Exception as e:
                logger.error(f"Error adding entity {entity.canonical_id}: {str(e)}")
        
        return doc_ids

    def similarity_search(self, query: str, k: int = 4) -> List[Tuple[str, float]]:
        """
        Perform similarity search (backward compatible)
        
        Args:
            query: Query text
            k: Number of results
            
        Returns:
            List of search results
        """
        return self.vector_store.similarity_search(query, k=k)

    def similarity_search_with_score(self, query: str, k: int = 4) -> List[Tuple[str, float]]:
        """
        Similarity search with relevance scores
        
        Args:
            query: Query text
            k: Number of results
            
        Returns:
            List of (document, score) tuples
        """
        return self.vector_store.similarity_search_with_score(query, k=k)

    def search_by_entity_type(self, entity_type: str, k: int = 10) -> List[Dict]:
        """
        Search for entities of specific type
        
        Args:
            entity_type: Type of entity to search for
            k: Number of results
            
        Returns:
            List of matching entities with metadata
        """
        results = []
        for doc_id, metadata in self.entity_metadata.items():
            if metadata['metadata'].get('entity_type') == entity_type:
                results.append({
                    "entity_id": metadata['entity'].canonical_id,
                    "text": metadata['entity'].canonical_text,
                    "type": entity_type,
                    "confidence": metadata['metadata']['confidence'],
                    "sources": metadata['traceability']
                })
        
        return results[:k]

    def get_entity_traceability(self, entity_id: str) -> Optional[Dict]:
        """
        Get traceability information for an entity
        
        Args:
            entity_id: Entity ID
            
        Returns:
            Traceability information or None
        """
        for doc_id, metadata in self.entity_metadata.items():
            if metadata['entity'].canonical_id == entity_id:
                return {
                    "entity_id": entity_id,
                    "entity_text": metadata['entity'].canonical_text,
                    "confidence_scores": [t.confidence_score for t in metadata['traceability']],
                    "average_confidence": metadata['metadata']['confidence'],
                    "sources": [
                        {
                            "source": t.source_metadata.filename,
                            "source_type": t.source_metadata.source.value,
                            "confidence": t.confidence_score,
                            "extraction_method": t.extraction_method,
                            "verified": t.verified_by is not None
                        }
                        for t in metadata['traceability']
                    ],
                    "merged_from": metadata['entity'].merged_from,
                    "added_at": metadata['added_at']
                }
        return None

    def get_statistics(self) -> Dict:
        """Get vector store statistics"""
        return {
            "total_entities": len(self.entity_metadata),
            "embedding_cache_size": len(self.embedding_cache),
            "entities_by_type": self._count_by_type(),
            "average_confidence": self._calculate_avg_confidence()
        }

    def _create_traceability_summary(self, traceability: List[Traceability]) -> str:
        """Create a text summary of traceability"""
        if not traceability:
            return "No source information"
        
        sources = [f"{t.source_metadata.filename}" for t in traceability]
        return "; ".join(set(sources))

    def _serialize_attributes(self, attributes: Dict) -> str:
        """Serialize attributes for storage"""
        items = [f"{k}={v}" for k, v in attributes.items() if not isinstance(v, (dict, list))]
        return "; ".join(items[:5]) if items else "None"

    def _get_avg_confidence(self, traceability: List[Traceability]) -> float:
        """Calculate average confidence score"""
        if not traceability:
            return 0.5
        return sum(t.confidence_score for t in traceability) / len(traceability)

    def _count_by_type(self) -> Dict[str, int]:
        """Count entities by type"""
        counts = {}
        for metadata in self.entity_metadata.values():
            entity_type = metadata['metadata'].get('entity_type', 'unknown')
            counts[entity_type] = counts.get(entity_type, 0) + 1
        return counts

    def _calculate_avg_confidence(self) -> float:
        """Calculate average confidence across all entities"""
        if not self.entity_metadata:
            return 0.0
        
        total = sum(m['metadata']['confidence'] for m in self.entity_metadata.values())
        return total / len(self.entity_metadata)

    def delete_entity(self, entity_id: str):
        """Delete an entity from vector store"""
        doc_id = f"entity_{entity_id}"
        if doc_id in self.entity_metadata:
            del self.entity_metadata[doc_id]
            logger.info(f"Deleted entity {entity_id} from vector store")

    def update_entity_verification(self, entity_id: str, verified_by: str):
        """Mark an entity as verified by a user"""
        for doc_id, metadata in self.entity_metadata.items():
            if metadata['entity'].canonical_id == entity_id:
                if metadata['traceability']:
                    metadata['traceability'][0].verified_by = verified_by
                    metadata['traceability'][0].verification_timestamp = datetime.utcnow()
                logger.info(f"Entity {entity_id} verified by {verified_by}")
                break

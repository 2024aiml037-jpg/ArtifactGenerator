"""
Knowledge Graph - Builds and manages relationships between entities
Also serves as the primary store for JSON documents (structured data),
while non-JSON files are stored in the Vector Store.
"""

import logging
from typing import List, Dict, Set, Optional, Tuple
import json
import re
from datetime import datetime

import networkx as nx

from models.schemas import (
    NormalizedEntity, KnowledgeGraphNode, KnowledgeGraphEdge, EntityType
)

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """Manages the knowledge graph with entities and relationships.
    
    Also stores ingested JSON documents as graph nodes with section-based
    relationships, enabling structured search alongside entity relationships.
    """

    def __init__(self):
        """Initialize a directed graph for knowledge representation"""
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, KnowledgeGraphNode] = {}
        self.edges: Dict[str, KnowledgeGraphEdge] = {}
        self.node_counter = 0
        self.edge_counter = 0
        # JSON document storage: tracks ingested JSON docs and their node IDs
        self.json_documents: Dict[str, Dict] = {}  # doc_id -> {filename, root_node_id, node_ids}

    # ==================== JSON Document Storage ====================

    def ingest_json_document(self, doc_id: str, filename: str, data: dict,
                             metadata: Optional[Dict] = None) -> Dict:
        """
        Ingest a JSON document into the knowledge graph as structured nodes.
        Each top-level key becomes a node; nested objects become child nodes
        linked by 'part_of' relationships.

        Args:
            doc_id: Unique document ID
            filename: Source filename
            data: Parsed JSON data (dict)
            metadata: Optional extra metadata

        Returns:
            Summary dict with root_node_id and total nodes created
        """
        node_ids = []
        meta = metadata or {}
        meta.update({'source_file': filename, 'document_id': doc_id,
                     'ingested_at': datetime.utcnow().isoformat()})

        # Create a root document node
        root_node_id = f"json_root_{self.node_counter}"
        self.node_counter += 1
        root_node = KnowledgeGraphNode(
            node_id=root_node_id,
            entity_id=doc_id,
            node_type=EntityType.ENTITY,
            properties={
                'text': f"JSON Document: {filename}",
                'filename': filename,
                'node_kind': 'json_root',
                **meta
            }
        )
        self.nodes[root_node_id] = root_node
        self.graph.add_node(root_node_id, **root_node.dict())
        node_ids.append(root_node_id)

        # Recursively add sections
        self._add_json_sections(data, root_node_id, filename, doc_id, node_ids, depth=0)

        self.json_documents[doc_id] = {
            'filename': filename,
            'root_node_id': root_node_id,
            'node_ids': node_ids,
            'metadata': meta
        }

        logger.info(f"Ingested JSON '{filename}' into knowledge graph: "
                     f"{len(node_ids)} nodes created (doc_id={doc_id})")
        return {
            'document_id': doc_id,
            'root_node_id': root_node_id,
            'total_nodes': len(node_ids),
            'stored_in': 'knowledge_graph'
        }

    def _add_json_sections(self, data, parent_node_id: str, filename: str,
                           doc_id: str, node_ids: list, depth: int):
        """Recursively add JSON sections as child nodes linked to parent"""
        if depth > 5:
            return  # safety limit

        items = []
        if isinstance(data, dict):
            items = list(data.items())
        elif isinstance(data, list):
            items = [(f"item_{i}", v) for i, v in enumerate(data)]
        else:
            return

        for key, value in items:
            section_node_id = f"json_sec_{self.node_counter}"
            self.node_counter += 1

            # Build a human-readable text for the section
            if isinstance(value, (dict, list)):
                text = f"[{key}]: {json.dumps(value, indent=2)}"
            else:
                text = f"[{key}]: {value}"

            section_node = KnowledgeGraphNode(
                node_id=section_node_id,
                entity_id=f"{doc_id}_{key}",
                node_type=EntityType.ENTITY,
                properties={
                    'text': text,
                    'section_key': key,
                    'filename': filename,
                    'document_id': doc_id,
                    'node_kind': 'json_section',
                    'depth': depth
                }
            )
            self.nodes[section_node_id] = section_node
            self.graph.add_node(section_node_id, **section_node.dict())
            node_ids.append(section_node_id)

            # Link to parent with 'part_of' relationship
            edge_id = f"edge_{self.edge_counter}"
            self.edge_counter += 1
            edge = KnowledgeGraphEdge(
                edge_id=edge_id,
                source_node_id=section_node_id,
                target_node_id=parent_node_id,
                relationship_type='part_of',
                strength=1.0,
                metadata={'document_id': doc_id}
            )
            self.edges[edge_id] = edge
            self.graph.add_edge(section_node_id, parent_node_id,
                                relation='part_of', strength=1.0, edge_id=edge_id)

            # Recurse into nested dicts/lists
            if isinstance(value, (dict, list)):
                self._add_json_sections(value, section_node_id, filename,
                                        doc_id, node_ids, depth + 1)

    def search_json_documents(self, query: str, k: int = 5) -> List[Dict]:
        """
        Search JSON documents stored in the knowledge graph using keyword matching.
        Returns the most relevant node texts and metadata.

        Args:
            query: Search query string
            k: Maximum number of results

        Returns:
            List of dicts with 'text', 'node_id', 'filename', 'score'
        """
        query_lower = query.lower()
        query_terms = set(re.split(r'\W+', query_lower)) - {'', 'the', 'a', 'an', 'is', 'of', 'for', 'in', 'to'}

        scored_results = []
        for node_id, node in self.nodes.items():
            node_kind = node.properties.get('node_kind', '')
            if node_kind not in ('json_section', 'json_root'):
                continue

            text = node.properties.get('text', '')
            text_lower = text.lower()

            # Score by term overlap
            score = sum(1 for term in query_terms if term in text_lower)
            if score > 0:
                scored_results.append({
                    'text': text,
                    'node_id': node_id,
                    'entity_id': node.entity_id,
                    'filename': node.properties.get('filename', 'unknown'),
                    'section_key': node.properties.get('section_key', ''),
                    'document_id': node.properties.get('document_id', ''),
                    'score': score
                })

        # Sort by score descending, return top k
        scored_results.sort(key=lambda x: x['score'], reverse=True)
        return scored_results[:k]

    def get_json_document_nodes(self, doc_id: str) -> List[Dict]:
        """Get all nodes for a specific JSON document"""
        doc_info = self.json_documents.get(doc_id)
        if not doc_info:
            return []

        results = []
        for nid in doc_info['node_ids']:
            node = self.nodes.get(nid)
            if node:
                results.append({
                    'node_id': nid,
                    'entity_id': node.entity_id,
                    'text': node.properties.get('text', ''),
                    'section_key': node.properties.get('section_key', ''),
                })
        return results

    def get_json_documents_list(self) -> List[Dict]:
        """List all ingested JSON documents"""
        return [
            {
                'document_id': doc_id,
                'filename': info['filename'],
                'total_nodes': len(info['node_ids']),
                'root_node_id': info['root_node_id']
            }
            for doc_id, info in self.json_documents.items()
        ]

    # ==================== Entity Management ====================

    def add_entities(self, entities: List[NormalizedEntity]):
        """Add normalized entities as nodes to the knowledge graph"""
        for entity in entities:
            self.add_node(entity)

    def add_node(self, entity: NormalizedEntity):
        """Add a single entity as a node"""
        node_id = f"node_{self.node_counter}"
        self.node_counter += 1

        node = KnowledgeGraphNode(
            node_id=node_id,
            entity_id=entity.canonical_id,
            node_type=entity.entity_type,
            properties={
                "text": entity.canonical_text,
                "confidence": sum(t.confidence_score for t in entity.traceability) / len(entity.traceability) if entity.traceability else 0.5,
                "synonyms": entity.synonyms,
                "attributes": entity.unified_attributes
            }
        )

        self.nodes[node_id] = node
        self.graph.add_node(node_id, **node.dict())
        logger.debug(f"Added node {node_id} for entity {entity.canonical_id}")

    def add_relationship(self, source_entity_id: str, target_entity_id: str, 
                        relationship_type: str, strength: float = 0.8):
        """
        Add a relationship between two entities

        Args:
            source_entity_id: ID of source entity
            target_entity_id: ID of target entity
            relationship_type: Type of relationship (e.g., "requires", "implements", "depends_on")
            strength: Strength of relationship (0-1)
        """
        source_node = self._find_node_by_entity_id(source_entity_id)
        target_node = self._find_node_by_entity_id(target_entity_id)

        if not source_node or not target_node:
            logger.warning(f"Cannot add relationship: missing node for {source_entity_id} or {target_entity_id}")
            return

        edge_id = f"edge_{self.edge_counter}"
        self.edge_counter += 1

        edge = KnowledgeGraphEdge(
            edge_id=edge_id,
            source_node_id=source_node.node_id,
            target_node_id=target_node.node_id,
            relationship_type=relationship_type,
            strength=strength
        )

        self.edges[edge_id] = edge
        self.graph.add_edge(source_node.node_id, target_node.node_id, 
                           relation=relationship_type, strength=strength, edge_id=edge_id)
        
        logger.debug(f"Added relationship {relationship_type} from {source_entity_id} to {target_entity_id}")

    def auto_discover_relationships(self, entities: List[NormalizedEntity], llm_service=None):
        """
        Auto-discover relationships between entities using LLM or text analysis
        
        Args:
            entities: List of normalized entities
            llm_service: Optional LLM service for semantic relationship discovery
        """
        logger.info("Starting auto-discovery of relationships...")

        for i, source_entity in enumerate(entities):
            for target_entity in entities[i + 1:]:
                if source_entity.canonical_id == target_entity.canonical_id:
                    continue

                # Check for explicit mentions of relationships
                relationships = self._find_relationships_in_text(
                    source_entity.canonical_text,
                    target_entity.canonical_text,
                    source_entity.unified_attributes,
                    target_entity.unified_attributes
                )

                for rel_type, strength in relationships:
                    self.add_relationship(
                        source_entity.canonical_id,
                        target_entity.canonical_id,
                        rel_type,
                        strength
                    )

                # Use LLM for semantic relationship discovery if available
                if llm_service:
                    try:
                        relationships = self._discover_semantic_relationships(
                            source_entity, target_entity, llm_service
                        )
                        for rel_type, strength in relationships:
                            self.add_relationship(
                                source_entity.canonical_id,
                                target_entity.canonical_id,
                                rel_type,
                                strength
                            )
                    except Exception as e:
                        logger.warning(f"Semantic relationship discovery failed: {e}")

        logger.info(f"Relationship discovery completed. Found {len(self.edges)} relationships")

    def _find_relationships_in_text(self, source_text: str, target_text: str,
                                   source_attrs: Dict, target_attrs: Dict) -> List[Tuple[str, float]]:
        """Find relationships mentioned in entity texts"""
        relationships = []

        # Simple keyword matching for common relationships
        relationship_keywords = {
            "requires": ["requires", "needs", "depends on", "prerequisite"],
            "implements": ["implements", "fulfills", "satisfies"],
            "conflicts_with": ["conflicts", "contradicts"],
            "documents": ["documents", "describes", "specifies"],
            "version_of": ["version", "release", "variant"],
            "part_of": ["part of", "component of", "subcomponent"],
            "aggregates": ["contains", "includes", "comprises"],
        }

        combined_text = f"{source_text} {target_text}".lower()

        for rel_type, keywords in relationship_keywords.items():
            for keyword in keywords:
                if keyword in combined_text:
                    relationships.append((rel_type, 0.7))

        # Check for attribute-based relationships
        if source_attrs.get('type') and target_attrs.get('type'):
            if source_attrs['type'] == 'Functional' and target_attrs['type'] == 'Non-Functional':
                relationships.append(("constraints", 0.8))

        return relationships

    def _discover_semantic_relationships(self, source_entity: NormalizedEntity,
                                        target_entity: NormalizedEntity,
                                        llm_service) -> List[Tuple[str, float]]:
        """Use LLM to discover semantic relationships"""
        prompt = f"""Based on these two entities, what is the relationship between them?
        
Entity 1: {source_entity.canonical_text}
Entity 2: {target_entity.canonical_text}

Identify if there are relationships like: requires, implements, depends_on, conflicts_with, 
part_of, documents, related_to.

Return JSON: {{"relationships": [{{"type": "...", "strength": 0.0-1.0}}]}}"""

        try:
            response = llm_service.invoke_extraction(prompt)
            # Parse response and extract relationships
            import json
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            else:
                json_str = response
            
            parsed = json.loads(json_str)
            return [(r['type'], r['strength']) for r in parsed.get('relationships', [])]
        except Exception as e:
            logger.warning(f"Semantic relationship discovery error: {e}")
            return []

    def _find_node_by_entity_id(self, entity_id: str) -> Optional[KnowledgeGraphNode]:
        """Find a graph node by entity ID"""
        for node in self.nodes.values():
            if node.entity_id == entity_id:
                return node
        return None

    def get_related_entities(self, entity_id: str, depth: int = 1) -> List[str]:
        """Get entities related to a given entity"""
        node = self._find_node_by_entity_id(entity_id)
        if not node:
            return []

        related = set()

        # BFS to find related entities
        current_level = {node.node_id}
        for _ in range(depth):
            next_level = set()
            for node_id in current_level:
                successors = self.graph.successors(node_id)
                predecessors = self.graph.predecessors(node_id)
                neighbors = list(successors) + list(predecessors)
                next_level.update(neighbors)
            current_level = next_level - {node.node_id}
            related.update(current_level)

        # Convert node IDs back to entity IDs
        result = []
        for node_id in related:
            if node_id in self.nodes:
                result.append(self.nodes[node_id].entity_id)

        return result

    def detect_cycles(self) -> List[List[str]]:
        """Detect cycles in the knowledge graph"""
        try:
            cycles = list(nx.simple_cycles(self.graph))
            logger.info(f"Detected {len(cycles)} cycles in knowledge graph")
            return cycles
        except:
            logger.warning("Cycle detection failed or no cycles found")
            return []

    def get_connected_components(self) -> List[Set[str]]:
        """Get connected components in the graph"""
        # Convert to undirected for component analysis
        undirected = self.graph.to_undirected()
        components = list(nx.connected_components(undirected))
        logger.info(f"Found {len(components)} connected components")
        return components

    def get_orphaned_entities(self) -> List[str]:
        """Get entities with no relationships"""
        orphaned = []
        for node_id, node in self.nodes.items():
            in_degree = self.graph.in_degree(node_id)
            out_degree = self.graph.out_degree(node_id)
            
            if in_degree == 0 and out_degree == 0:
                orphaned.append(node.entity_id)

        if orphaned:
            logger.warning(f"Found {len(orphaned)} orphaned entities")

        return orphaned

    def get_statistics(self) -> Dict:
        """Get graph statistics"""
        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "density": nx.density(self.graph),
            "connected_components": len(self.get_connected_components()),
            "cycles": len(self.detect_cycles()),
            "orphaned_entities": len(self.get_orphaned_entities()),
            "avg_in_degree": sum(d for n, d in self.graph.in_degree()) / max(1, len(self.nodes)),
            "avg_out_degree": sum(d for n, d in self.graph.out_degree()) / max(1, len(self.nodes)),
        }

    def export_to_json(self) -> str:
        """Export graph to JSON format"""
        graph_data = {
            "nodes": [
                {
                    "id": node.node_id,
                    "entity_id": node.entity_id,
                    "type": node.node_type.value,
                    "properties": node.properties
                }
                for node in self.nodes.values()
            ],
            "edges": [
                {
                    "id": edge.edge_id,
                    "source": edge.source_node_id,
                    "target": edge.target_node_id,
                    "relationship": edge.relationship_type,
                    "strength": edge.strength
                }
                for edge in self.edges.values()
            ]
        }
        return json.dumps(graph_data, indent=2)

    def visualize_subgraph(self, entity_id: str, depth: int = 1) -> Dict:
        """Get visualization data for a subgraph around an entity"""
        node = self._find_node_by_entity_id(entity_id)
        if not node:
            return {}

        # Get related entities
        related_node_ids = self.get_related_entities(entity_id, depth)
        related_nodes = [self._find_node_by_entity_id(eid) for eid in related_node_ids]

        center_node = self.nodes.get(node.node_id)

        subgraph_nodes = [center_node] + related_nodes
        subgraph_node_ids = {n.node_id for n in subgraph_nodes if n}

        subgraph_edges = [
            e for e in self.edges.values()
            if e.source_node_id in subgraph_node_ids and e.target_node_id in subgraph_node_ids
        ]

        return {
            "center": center_node.dict() if center_node else None,
            "nodes": [n.dict() for n in subgraph_nodes if n],
            "edges": [e.dict() for e in subgraph_edges]
        }

"""
Knowledge Graph - Builds and manages relationships between entities
"""

import logging
from typing import List, Dict, Set, Optional, Tuple
import json

import networkx as nx

from models.schemas import (
    NormalizedEntity, KnowledgeGraphNode, KnowledgeGraphEdge, EntityType
)

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """Manages the knowledge graph with entities and relationships"""

    def __init__(self):
        """Initialize a directed graph for knowledge representation"""
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, KnowledgeGraphNode] = {}
        self.edges: Dict[str, KnowledgeGraphEdge] = {}
        self.node_counter = 0
        self.edge_counter = 0

    def add_entities(self, entities: List[NormalizedEntity]):
        """Add normalized entities as nodes to the graph"""
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
        self.graph.add_node(node_id, **node.model_dump())
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
            "center": center_node.model_dump() if center_node else None,
            "nodes": [n.model_dump() for n in subgraph_nodes if n],
            "edges": [e.model_dump() for e in subgraph_edges]
        }

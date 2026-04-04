"""
Data models and schemas for the Knowledge Platform.
Defines structured representations of documents, entities, and knowledge.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum


class EntityType(str, Enum):
    """Types of extracted entities"""
    REQUIREMENT = "requirement"
    RULE = "rule"
    ENTITY = "entity"
    API = "api"
    DATABASE = "database"
    CODE = "code"
    DESIGN = "design"


class SourceType(str, Enum):
    """Types of data sources"""
    PDF = "pdf"
    WORD = "word"
    CONFLUENCE = "confluence"
    DATABASE = "database"
    CODE = "code"
    API = "api"
    TEXT = "txt"


class ConfidenceLevel(str, Enum):
    """Confidence levels for extracted data"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Metadata(BaseModel):
    """Metadata for any document or entity"""
    source: SourceType
    source_url: Optional[str] = None
    filename: Optional[str] = None
    version: str = "1.0"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    author: Optional[str] = None
    tags: List[str] = []
    custom_fields: Dict[str, Any] = {}


class Traceability(BaseModel):
    """Traceability information for audit and lineage"""
    source_id: str
    source_metadata: Metadata
    confidence_score: float = Field(ge=0.0, le=1.0)
    confidence_level: ConfidenceLevel
    extraction_method: str  # e.g., "llm", "regex", "manual"
    chunk_reference: Optional[str] = None
    verified_by: Optional[str] = None
    verification_timestamp: Optional[datetime] = None
    version_history: List[Dict[str, Any]] = []


class ExtractedEntity(BaseModel):
    """An extracted entity from documents"""
    id: str
    type: EntityType
    text: str
    description: Optional[str] = None
    traceability: Traceability
    related_entities: List[str] = []  # IDs of related entities
    attributes: Dict[str, Any] = {}


class NormalizedEntity(BaseModel):
    """Deduplicated and normalized entity"""
    canonical_id: str
    canonical_text: str
    entity_type: EntityType
    merged_from: List[str] = []  # IDs of merged entities
    traceability: List[Traceability] = []  # Multiple sources
    unified_attributes: Dict[str, Any] = {}
    synonyms: List[str] = []


class Conflict(BaseModel):
    """Detected conflict in knowledge"""
    conflict_id: str
    type: str  # "contradiction", "gap", "inconsistency", "ambiguity"
    involved_entities: List[str]
    description: str
    severity: str = Field(default="medium")  # "low", "medium", "high"
    resolution_status: str = Field(default="unresolved")  # "unresolved", "reviewing", "resolved"
    suggested_resolution: Optional[str] = None
    reviewer_feedback: Optional[str] = None


class KnowledgeGraphNode(BaseModel):
    """Node in the knowledge graph"""
    node_id: str
    entity_id: str
    node_type: EntityType
    properties: Dict[str, Any] = {}


class KnowledgeGraphEdge(BaseModel):
    """Edge in the knowledge graph showing relationships"""
    edge_id: str
    source_node_id: str
    target_node_id: str
    relationship_type: str  # "requires", "implements", "depends_on", "documents", etc.
    strength: float = Field(ge=0.0, le=1.0)  # Strength of relationship
    metadata: Dict[str, Any] = {}


class IngestedDocument(BaseModel):
    """Structured representation of an ingested document"""
    document_id: str
    content: str
    metadata: Metadata
    chunks: List[str] = []
    extraction_status: str = Field(default="pending")  # "pending", "extracted", "normalized", "validated"


class ExtractionResult(BaseModel):
    """Results from AI extraction"""
    document_id: str
    extracted_entities: List[ExtractedEntity]
    extraction_timestamp: datetime = Field(default_factory=datetime.utcnow)
    errors: List[str] = []
    warnings: List[str] = []


class NormalizationResult(BaseModel):
    """Results from normalization/deduplication"""
    normalized_entities: List[NormalizedEntity]
    duplicates_removed: int
    merges_performed: int
    normalization_timestamp: datetime = Field(default_factory=datetime.utcnow)


class ValidationResult(BaseModel):
    """Results from validation"""
    document_id: str
    is_valid: bool
    conflicts: List[Conflict] = []
    gaps: List[str] = []
    inconsistencies: List[str] = []
    validation_timestamp: datetime = Field(default_factory=datetime.utcnow)
    validation_score: float = Field(ge=0.0, le=1.0)


class GeneratedDocument(BaseModel):
    """Auto-generated document from knowledge"""
    document_id: str
    document_type: str  # "requirements", "design", "rules", "test_cases"
    title: str
    content: str
    source_entities: List[str]  # IDs of entities used
    generation_timestamp: datetime = Field(default_factory=datetime.utcnow)
    requires_review: bool = False


class KnowledgeSnapshot(BaseModel):
    """Point-in-time snapshot of knowledge state"""
    snapshot_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    total_entities: int
    total_relationships: int
    conflicts_count: int
    validation_score: float
    changes_from_previous: Dict[str, Any] = {}


class UserFeedback(BaseModel):
    """User feedback for continuous improvement"""
    feedback_id: str
    entity_id: str
    feedback_type: str  # "correction", "suggestion", "validation", "edit"
    original_text: str
    corrected_text: Optional[str] = None
    feedback_timestamp: datetime = Field(default_factory=datetime.utcnow)
    user_id: Optional[str] = None
    notes: Optional[str] = None

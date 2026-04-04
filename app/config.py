"""
Application Configuration
Loads settings from environment variables with defaults
"""

import os 
from dotenv import load_dotenv 

load_dotenv()

class Config:
    """Base configuration"""
    
    # LLM Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    
    # AWS Configuration
    AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
    AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
    AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
    AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
    
    # Vector Database Configuration
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "vector_db")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    
    # Extraction Configuration
    EXTRACTION_CONFIDENCE_THRESHOLD = float(os.getenv("EXTRACTION_CONFIDENCE_THRESHOLD", "0.6"))
    DEDUPLICATION_SIMILARITY_THRESHOLD = float(os.getenv("DEDUPLICATION_SIMILARITY_THRESHOLD", "0.85"))
    
    # Validation Configuration
    ENABLE_CONFLICT_DETECTION = os.getenv("ENABLE_CONFLICT_DETECTION", "true").lower() == "true"
    ENABLE_GAP_DETECTION = os.getenv("ENABLE_GAP_DETECTION", "true").lower() == "true"
    VALIDATION_SCORE_THRESHOLD = float(os.getenv("VALIDATION_SCORE_THRESHOLD", "0.7"))
    
    # Knowledge Graph Configuration
    AUTO_DISCOVER_RELATIONSHIPS = os.getenv("AUTO_DISCOVER_RELATIONSHIPS", "true").lower() == "true"
    RELATIONSHIP_STRENGTH_THRESHOLD = float(os.getenv("RELATIONSHIP_STRENGTH_THRESHOLD", "0.5"))
    
    # Observability Configuration
    ENABLE_METRICS = os.getenv("ENABLE_METRICS", "true").lower() == "true"
    METRICS_PORT = int(os.getenv("METRICS_PORT", "8081"))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # Document Processing Configuration
    MAX_DOCUMENT_SIZE_MB = int(os.getenv("MAX_DOCUMENT_SIZE_MB", "50"))
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # API Configuration
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8080"))
    DEBUG_MODE = os.getenv("DEBUG_MODE", "true").lower() == "true"
    
    # Confluence Configuration (optional)
    CONFLUENCE_URL = os.getenv("CONFLUENCE_URL")
    CONFLUENCE_USERNAME = os.getenv("CONFLUENCE_USERNAME")
    CONFLUENCE_API_TOKEN = os.getenv("CONFLUENCE_API_TOKEN")
    
    # Database Configuration (optional)
    DATABASE_URL = os.getenv("DATABASE_URL")
    
    # Feature Flags
    ENABLE_INGESTION_LAYER = os.getenv("ENABLE_INGESTION_LAYER", "true").lower() == "true"
    ENABLE_EXTRACTION_LAYER = os.getenv("ENABLE_EXTRACTION_LAYER", "true").lower() == "true"
    ENABLE_NORMALIZATION = os.getenv("ENABLE_NORMALIZATION", "true").lower() == "true"
    ENABLE_VALIDATION = os.getenv("ENABLE_VALIDATION", "true").lower() == "true"
    ENABLE_KNOWLEDGE_GRAPH = os.getenv("ENABLE_KNOWLEDGE_GRAPH", "true").lower() == "true"
    ENABLE_AUTO_GENERATION = os.getenv("ENABLE_AUTO_GENERATION", "true").lower() == "true"
    ENABLE_FEEDBACK_LOOP = os.getenv("ENABLE_FEEDBACK_LOOP", "true").lower() == "true"

    @classmethod
    def validate(cls):
        """Validate required configuration"""
        required = ['OPENAI_API_KEY', 'AWS_ACCESS_KEY', 'AWS_SECRET_KEY', 'AWS_BUCKET_NAME']
        missing = [field for field in required if not getattr(cls, field)]
        
        if missing:
            raise ValueError(f"Missing required configuration: {', '.join(missing)}")
        
        return True
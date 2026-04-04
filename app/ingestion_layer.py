"""
Ingestion Layer - Handles multiple data sources (PDF, Word, Confluence, DB, Code)
Converts all sources into structured JSON with metadata
"""

import os
import tempfile
import json
from typing import List, Optional
from datetime import datetime
import logging

from langchain.document_loaders import TextLoader, PyPDFLoader
try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None

from models.schemas import (
    IngestedDocument, Metadata, SourceType
)

logger = logging.getLogger(__name__)


class IngestionLayer:
    """Handles ingestion from multiple data sources"""

    def __init__(self):
        self.supported_formats = {'.pdf', '.txt', '.docx', '.json'}
        self.document_counter = 0

    def ingest_pdf(self, file_path: str, metadata: Optional[Metadata] = None) -> IngestedDocument:
        """Ingest PDF file"""
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            content = "\n".join([doc.page_content for doc in documents])
            
            if metadata is None:
                metadata = Metadata(
                    source=SourceType.PDF,
                    filename=os.path.basename(file_path),
                    timestamp=datetime.utcnow()
                )
            
            doc_id = f"doc_{self.document_counter}_{int(datetime.utcnow().timestamp())}"
            self.document_counter += 1
            
            return IngestedDocument(
                document_id=doc_id,
                content=content,
                metadata=metadata,
                chunks=[doc.page_content for doc in documents]
            )
        except Exception as e:
            logger.error(f"Error ingesting PDF {file_path}: {str(e)}")
            raise

    def ingest_text(self, file_path: str, metadata: Optional[Metadata] = None) -> IngestedDocument:
        """Ingest plain text file"""
        try:
            loader = TextLoader(file_path)
            documents = loader.load()
            content = "".join([doc.page_content for doc in documents])
            
            if metadata is None:
                metadata = Metadata(
                    source=SourceType.TEXT,
                    filename=os.path.basename(file_path),
                    timestamp=datetime.utcnow()
                )
            
            doc_id = f"doc_{self.document_counter}_{int(datetime.utcnow().timestamp())}"
            self.document_counter += 1
            
            return IngestedDocument(
                document_id=doc_id,
                content=content,
                metadata=metadata,
                chunks=[content]
            )
        except Exception as e:
            logger.error(f"Error ingesting text file {file_path}: {str(e)}")
            raise

    def ingest_docx(self, file_path: str, metadata: Optional[Metadata] = None) -> IngestedDocument:
        """Ingest Word document"""
        if DocxDocument is None:
            raise ImportError("python-docx is not installed. Install with: pip install python-docx")
        
        try:
            doc = DocxDocument(file_path)
            content = "\n".join([para.text for para in doc.paragraphs])
            
            if metadata is None:
                metadata = Metadata(
                    source=SourceType.WORD,
                    filename=os.path.basename(file_path),
                    timestamp=datetime.utcnow()
                )
            
            doc_id = f"doc_{self.document_counter}_{int(datetime.utcnow().timestamp())}"
            self.document_counter += 1
            
            return IngestedDocument(
                document_id=doc_id,
                content=content,
                metadata=metadata,
                chunks=[content]
            )
        except Exception as e:
            logger.error(f"Error ingesting DOCX {file_path}: {str(e)}")
            raise

    def ingest_json(self, file_path: str, metadata: Optional[Metadata] = None) -> IngestedDocument:
        """Ingest JSON file (structured data)"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            content = json.dumps(data, indent=2)
            
            if metadata is None:
                metadata = Metadata(
                    source=SourceType.TEXT,
                    filename=os.path.basename(file_path),
                    timestamp=datetime.utcnow()
                )
            
            doc_id = f"doc_{self.document_counter}_{int(datetime.utcnow().timestamp())}"
            self.document_counter += 1
            
            return IngestedDocument(
                document_id=doc_id,
                content=content,
                metadata=metadata,
                chunks=[content]
            )
        except Exception as e:
            logger.error(f"Error ingesting JSON {file_path}: {str(e)}")
            raise

    def ingest_file(self, file_path: str, metadata: Optional[Metadata] = None) -> IngestedDocument:
        """Ingest file based on extension"""
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            return self.ingest_pdf(file_path, metadata)
        elif file_ext == '.txt':
            return self.ingest_text(file_path, metadata)
        elif file_ext == '.docx':
            return self.ingest_docx(file_path, metadata)
        elif file_ext == '.json':
            return self.ingest_json(file_path, metadata)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}. Supported: {self.supported_formats}")

    def ingest_from_file_object(self, file_obj, filename: str, metadata: Optional[Metadata] = None) -> IngestedDocument:
        """Ingest from file object (via Flask upload)"""
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, filename)
        
        try:
            file_obj.save(temp_path)
            
            if metadata is None:
                file_ext = os.path.splitext(filename)[1].lower()
                source_type = SourceType.PDF if file_ext == '.pdf' else SourceType.TEXT
                metadata = Metadata(
                    source=source_type,
                    filename=filename,
                    timestamp=datetime.utcnow()
                )
            
            return self.ingest_file(temp_path, metadata)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)

    def ingest_text_content(self, content: str, source_name: str, metadata: Optional[Metadata] = None) -> IngestedDocument:
        """Ingest raw text content"""
        if metadata is None:
            metadata = Metadata(
                source=SourceType.TEXT,
                filename=source_name,
                timestamp=datetime.utcnow()
            )
        
        doc_id = f"doc_{self.document_counter}_{int(datetime.utcnow().timestamp())}"
        self.document_counter += 1
        
        return IngestedDocument(
            document_id=doc_id,
            content=content,
            metadata=metadata,
            chunks=[content]
        )

    def ingest_db_schema(self, db_url: str, metadata: Optional[Metadata] = None) -> IngestedDocument:
        """Ingest database schema"""
        try:
            import sqlalchemy as sa
            
            engine = sa.create_engine(db_url)
            inspector = sa.inspect(engine)
            
            schema_info = {}
            for table_name in inspector.get_table_names():
                columns = inspector.get_columns(table_name)
                schema_info[table_name] = [
                    {
                        'name': col['name'],
                        'type': str(col['type']),
                        'nullable': col['nullable']
                    }
                    for col in columns
                ]
            
            content = json.dumps(schema_info, indent=2)
            
            if metadata is None:
                metadata = Metadata(
                    source=SourceType.DATABASE,
                    filename="database_schema",
                    timestamp=datetime.utcnow()
                )
            
            doc_id = f"doc_{self.document_counter}_{int(datetime.utcnow().timestamp())}"
            self.document_counter += 1
            
            return IngestedDocument(
                document_id=doc_id,
                content=content,
                metadata=metadata,
                chunks=[content]
            )
        except ImportError:
            raise ImportError("sqlalchemy is not installed. Install with: pip install sqlalchemy")
        except Exception as e:
            logger.error(f"Error ingesting database schema: {str(e)}")
            raise

    def ingest_code_file(self, file_path: str, metadata: Optional[Metadata] = None) -> IngestedDocument:
        """Ingest source code file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if metadata is None:
                metadata = Metadata(
                    source=SourceType.CODE,
                    filename=os.path.basename(file_path),
                    timestamp=datetime.utcnow()
                )
            
            doc_id = f"doc_{self.document_counter}_{int(datetime.utcnow().timestamp())}"
            self.document_counter += 1
            
            return IngestedDocument(
                document_id=doc_id,
                content=content,
                metadata=metadata,
                chunks=[content]
            )
        except Exception as e:
            logger.error(f"Error ingesting code file {file_path}: {str(e)}")
            raise

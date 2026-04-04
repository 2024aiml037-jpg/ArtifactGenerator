"""
Main Flask Application - Enterprise Knowledge Platform
Orchestrates all layers: Ingestion, Extraction, Normalization, Validation, Knowledge Graph, Output
"""

import os
import logging
from datetime import datetime

from flask import Flask, request, render_template, jsonify
from langchain.text_splitter import RecursiveCharacterTextSplitter

from models.vector_store import VectorStore
from services.storage_service import S3Storage
from services.llm_service import LLMService
from config import Config
from ingestion_layer import IngestionLayer
from extraction_layer import ExtractionLayer
from normalization_engine import NormalizationEngine
from validation_engine import ValidationEngine
from knowledge_graph import KnowledgeGraph
from observability import Observability, FeedbackLoop
from decision_engine import CreditDecisionEngine
from models.schemas import GeneratedDocument, UserFeedback, EntityType

# Validate configuration
try:
    Config.validate()
except ValueError as e:
    print(f"Configuration Error: {e}")
    exit(1)

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_DOCUMENT_SIZE_MB * 1024 * 1024

# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize core services
vector_store = VectorStore(Config.VECTOR_DB_PATH)
storage_service = S3Storage()
llm_service = LLMService(vector_store)

# Initialize knowledge platform layers
ingestion_layer = IngestionLayer() if Config.ENABLE_INGESTION_LAYER else None
extraction_layer = ExtractionLayer(llm_service) if Config.ENABLE_EXTRACTION_LAYER else None
normalization_engine = NormalizationEngine() if Config.ENABLE_NORMALIZATION else None
validation_engine = ValidationEngine(llm_service) if Config.ENABLE_VALIDATION else None
knowledge_graph = KnowledgeGraph() if Config.ENABLE_KNOWLEDGE_GRAPH else None

# Initialize observability and feedback
observability = Observability()
feedback_loop = FeedbackLoop() if Config.ENABLE_FEEDBACK_LOOP else None

# Initialize credit decision engine
data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
decision_engine = CreditDecisionEngine(data_dir, vector_store=vector_store, ingestion_layer=ingestion_layer)

# Pipeline state tracker
pipeline_state = {
    'current_document_id': None,
    'ingested_document': None,
    'extraction_result': None,
    'normalization_result': None,
    'validation_result': None,
    'knowledge_graph_built': False
}


# ==================== UI Routes ====================

@app.route('/')
def index():
    """Serve the main UI"""
    return render_template('index.html')


@app.route('/api/status')
def get_status():
    """Get system status and configuration"""
    return jsonify({
        'status': 'operational',
        'timestamp': datetime.utcnow().isoformat(),
        'config': {
            'ingestion_enabled': Config.ENABLE_INGESTION_LAYER,
            'extraction_enabled': Config.ENABLE_EXTRACTION_LAYER,
            'normalization_enabled': Config.ENABLE_NORMALIZATION,
            'validation_enabled': Config.ENABLE_VALIDATION,
            'knowledge_graph_enabled': Config.ENABLE_KNOWLEDGE_GRAPH,
            'auto_generation_enabled': Config.ENABLE_AUTO_GENERATION,
            'feedback_enabled': Config.ENABLE_FEEDBACK_LOOP
        },
        'vector_store_stats': vector_store.get_statistics(),
        'knowledge_graph_stats': knowledge_graph.get_statistics() if knowledge_graph else {}
    })


# ==================== Ingestion Routes ====================

@app.route('/upload', methods=['POST'])
def upload_document():
    """Upload and ingest a document"""
    try:
        with observability.track_operation("document_upload"):
            logger.debug("Upload endpoint called")
            
            if 'file' not in request.files:
                logger.warning("No file in request")
                return jsonify({'error': 'No file provided'}), 400
            
            file = request.files['file']
            if file.filename == '':
                logger.warning("Empty filename")
                return jsonify({'error': 'No file selected'}), 400

            # Validate file extension
            supported_exts = {'.txt', '.pdf', '.docx', '.json', '.xlsx'}
            file_ext = os.path.splitext(file.filename)[1].lower()
            
            if file_ext not in supported_exts:
                logger.warning(f"Unsupported file type: {file.filename}")
                return jsonify({'error': f'Only {supported_exts} files are supported'}), 400

            logger.debug(f"Processing file: {file.filename}")
            
            # STEP 1: Ingestion
            if not Config.ENABLE_INGESTION_LAYER:
                return jsonify({'error': 'Ingestion layer is disabled'}), 503
            
            ingested_doc = ingestion_layer.ingest_from_file_object(
                file, file.filename
            )
            
            pipeline_state['current_document_id'] = ingested_doc.document_id
            
            # Upload to S3
            file.seek(0)
            storage_service.upload_file(file, file.filename)
            logger.debug("File uploaded to S3")
            
            # Store ingested document for pipeline
            pipeline_state['ingested_document'] = ingested_doc
            
            # Add to vector store
            vector_store.add_text_chunks(ingested_doc.chunks)
            logger.debug("Document chunks added to vector store")
            
            observability.record_ingestion(
                source_type=str(ingested_doc.metadata.source),
                duration=0.0,
                document_count=len(ingested_doc.chunks)
            )
            
            # Return structured JSON metadata
            return jsonify({
                'message': 'Document ingested successfully',
                'document_id': ingested_doc.document_id,
                'chunks_processed': len(ingested_doc.chunks),
                'metadata': {
                    'source': ingested_doc.metadata.source.value,
                    'filename': ingested_doc.metadata.filename,
                    'version': ingested_doc.metadata.version,
                    'timestamp': ingested_doc.metadata.timestamp.isoformat()
                },
                'content_preview': ingested_doc.content[:500],
                'next_step': '/api/extract'
            }), 200

    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500


@app.route('/api/ingest/sample-data', methods=['POST'])
def ingest_sample_data():
    """Ingest Zoot sample request and rules from data/ directory into Vector DB"""
    try:
        with observability.track_operation("sample_data_ingestion"):
            if not Config.ENABLE_INGESTION_LAYER:
                return jsonify({'error': 'Ingestion layer is disabled'}), 503

            data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
            
            # Define the Zoot sample files to ingest
            zoot_files = [
                'SampleZootRequest.json',
                'ZootRules.xlsx',
            ]
            
            # Also ingest response samples if present
            optional_files = [
                'SampleZootResponseApproved.json',
                'SampleZootResponseDecline.json',
                'SampleZootResponseRefer.json',
            ]
            
            ingested_docs = []
            all_chunks = []
            
            for filename in zoot_files + optional_files:
                file_path = os.path.join(data_dir, filename)
                if not os.path.exists(file_path):
                    if filename in zoot_files:
                        return jsonify({'error': f'Required file not found: {filename}'}), 404
                    continue
                
                logger.info(f"Ingesting sample file: {filename}")
                ingested_doc = ingestion_layer.ingest_file(file_path)
                ingested_docs.append(ingested_doc)
                all_chunks.extend(ingested_doc.chunks)
                
                # Store to vector DB
                vector_store.add_text_chunks(
                    ingested_doc.chunks,
                    metadatas=[{'source': filename, 'document_id': ingested_doc.document_id} for _ in ingested_doc.chunks]
                )
                
                observability.record_ingestion(
                    source_type=str(ingested_doc.metadata.source),
                    duration=0.0,
                    document_count=len(ingested_doc.chunks)
                )
            
            # Store the last ingested doc in pipeline state for further processing
            if ingested_docs:
                # Merge all docs into a combined document for pipeline
                combined_content = "\n\n".join(doc.content for doc in ingested_docs)
                from models.schemas import IngestedDocument, Metadata, SourceType
                combined_doc = IngestedDocument(
                    document_id=f"zoot_combined_{int(datetime.utcnow().timestamp())}",
                    content=combined_content,
                    metadata=Metadata(
                        source=SourceType.API,
                        filename="ZootCombined",
                        timestamp=datetime.utcnow(),
                        tags=["zoot", "sample-data"]
                    ),
                    chunks=all_chunks
                )
                pipeline_state['current_document_id'] = combined_doc.document_id
                pipeline_state['ingested_document'] = combined_doc
            
            results = []
            for doc in ingested_docs:
                results.append({
                    'document_id': doc.document_id,
                    'filename': doc.metadata.filename,
                    'source_type': doc.metadata.source.value,
                    'chunks_stored': len(doc.chunks),
                    'content_preview': doc.content[:300]
                })
            
            return jsonify({
                'message': f'Successfully ingested {len(ingested_docs)} Zoot data files into Vector DB',
                'total_chunks_in_vector_db': len(all_chunks),
                'combined_document_id': pipeline_state['current_document_id'],
                'documents': results,
                'next_steps': ['/api/pipeline/run', '/api/extract']
            }), 200

    except Exception as e:
        logger.error(f"Sample data ingestion error: {str(e)}")
        return jsonify({'error': f'Sample data ingestion failed: {str(e)}'}), 500


# ==================== Extraction Routes ====================

@app.route('/api/extract', methods=['POST'])
def extract_entities():
    """Extract requirements, rules, entities using LLMs"""
    try:
        with observability.track_operation("entity_extraction"):
            data = request.json or {}
            document_id = data.get('document_id') or pipeline_state['current_document_id']
            
            if not document_id:
                return jsonify({'error': 'No document to extract from'}), 400
            
            if not Config.ENABLE_EXTRACTION_LAYER:
                return jsonify({'error': 'Extraction layer is disabled'}), 503
            
            ingested_doc = pipeline_state.get('ingested_document')
            if not ingested_doc or ingested_doc.document_id != document_id:
                return jsonify({'error': f'Ingested document {document_id} not found in pipeline'}), 404
            
            logger.info(f"Extracting entities from document {document_id}")
            
            # STEP 3: AI EXTRACTION - Extract requirements, rules, entities using LLMs
            extraction_result = extraction_layer.extract_all_entities(ingested_doc)
            
            # Store in pipeline state
            pipeline_state['extraction_result'] = extraction_result
            
            # Build entity summary by type
            entities_by_type = {}
            for entity in extraction_result.extracted_entities:
                etype = entity.type.value
                if etype not in entities_by_type:
                    entities_by_type[etype] = []
                entities_by_type[etype].append({
                    'id': entity.id,
                    'text': entity.text[:200],
                    'confidence': entity.traceability.confidence_score,
                    'confidence_level': entity.traceability.confidence_level.value,
                    'source': entity.traceability.source_id
                })
            
            observability.record_extraction(
                extraction_type="all",
                duration=0.0,
                entity_types={k: len(v) for k, v in entities_by_type.items()},
                avg_confidence=sum(
                    e.traceability.confidence_score for e in extraction_result.extracted_entities
                ) / max(1, len(extraction_result.extracted_entities))
            )
            
            return jsonify({
                'message': 'Extraction completed successfully',
                'document_id': document_id,
                'total_entities': len(extraction_result.extracted_entities),
                'entities_by_type': {k: len(v) for k, v in entities_by_type.items()},
                'entities': entities_by_type,
                'errors': extraction_result.errors,
                'warnings': extraction_result.warnings,
                'next_step': '/api/normalize'
            }), 200

    except Exception as e:
        logger.error(f"Extraction error: {str(e)}")
        return jsonify({'error': f'Extraction failed: {str(e)}'}), 500


# ==================== Normalization Routes ====================

@app.route('/api/normalize', methods=['POST'])
def normalize_entities():
    """Remove duplicates using embeddings and semantic similarity"""
    try:
        with observability.track_operation("entity_normalization"):
            if not Config.ENABLE_NORMALIZATION:
                return jsonify({'error': 'Normalization is disabled'}), 503
            
            extraction_result = pipeline_state.get('extraction_result')
            if not extraction_result or not extraction_result.extracted_entities:
                return jsonify({'error': 'No extracted entities to normalize. Run /api/extract first.'}), 400
            
            logger.info(f"Normalizing {len(extraction_result.extracted_entities)} extracted entities")
            
            # STEP 4: NORMALIZATION - Remove duplicates using embeddings and semantic similarity
            normalization_result = normalization_engine.normalize(extraction_result.extracted_entities)
            
            # Store in pipeline state
            pipeline_state['normalization_result'] = normalization_result
            
            # Add normalized entities to vector store with traceability
            vector_store.add_normalized_entities(normalization_result.normalized_entities)
            
            normalized_summary = []
            for entity in normalization_result.normalized_entities:
                normalized_summary.append({
                    'canonical_id': entity.canonical_id,
                    'text': entity.canonical_text[:200],
                    'type': entity.entity_type.value,
                    'merged_from_count': len(entity.merged_from),
                    'synonyms': entity.synonyms,
                    'traceability': [{
                        'source': t.source_id,
                        'confidence': t.confidence_score,
                        'method': t.extraction_method
                    } for t in entity.traceability]
                })
            
            return jsonify({
                'message': 'Normalization completed successfully',
                'total_normalized': len(normalization_result.normalized_entities),
                'duplicates_removed': normalization_result.duplicates_removed,
                'merges_performed': normalization_result.merges_performed,
                'entities': normalized_summary,
                'next_step': '/api/validate'
            }), 200

    except Exception as e:
        logger.error(f"Normalization error: {str(e)}")
        return jsonify({'error': f'Normalization failed: {str(e)}'}), 500


# ==================== Validation Routes ====================

@app.route('/api/validate', methods=['POST'])
def validate_knowledge():
    """Detect conflicts, gaps, inconsistencies"""
    try:
        with observability.track_operation("knowledge_validation"):
            if not Config.ENABLE_VALIDATION:
                return jsonify({'error': 'Validation is disabled'}), 503
            
            normalization_result = pipeline_state.get('normalization_result')
            if not normalization_result or not normalization_result.normalized_entities:
                return jsonify({'error': 'No normalized entities to validate. Run /api/normalize first.'}), 400
            
            logger.info(f"Validating {len(normalization_result.normalized_entities)} normalized entities")
            
            # Gather knowledge graph edges if available
            kg_edges = None
            if knowledge_graph and pipeline_state.get('knowledge_graph_built'):
                kg_edges = [e.dict() for e in knowledge_graph.edges.values()]
            
            # STEP 6: VALIDATION - Detect conflicts, gaps, inconsistencies
            validation_result = validation_engine.validate(
                normalization_result.normalized_entities,
                knowledge_graph_edges=kg_edges
            )
            
            # Store in pipeline state
            pipeline_state['validation_result'] = validation_result
            
            observability.record_validation(
                duration=0.0,
                total_issues=len(validation_result.conflicts) + len(validation_result.gaps) + len(validation_result.inconsistencies),
                validation_score=validation_result.validation_score
            )
            
            conflicts_summary = [{
                'id': c.conflict_id,
                'type': c.type,
                'severity': c.severity,
                'description': c.description,
                'involved_entities': c.involved_entities
            } for c in validation_result.conflicts]
            
            return jsonify({
                'message': 'Validation completed',
                'is_valid': validation_result.is_valid,
                'validation_score': validation_result.validation_score,
                'conflicts': conflicts_summary,
                'gaps': validation_result.gaps,
                'inconsistencies': validation_result.inconsistencies,
                'next_step': '/api/knowledge-graph/build'
            }), 200

    except Exception as e:
        logger.error(f"Validation error: {str(e)}")
        return jsonify({'error': f'Validation failed: {str(e)}'}), 500


# ==================== Knowledge Graph Routes ====================

@app.route('/api/knowledge-graph/build', methods=['POST'])
def build_knowledge_graph():
    """Build relationships: Requirement -> Rule -> Code -> API -> DB"""
    try:
        with observability.track_operation("knowledge_graph_build"):
            if not Config.ENABLE_KNOWLEDGE_GRAPH:
                return jsonify({'error': 'Knowledge graph is disabled'}), 503
            
            normalization_result = pipeline_state.get('normalization_result')
            if not normalization_result or not normalization_result.normalized_entities:
                return jsonify({'error': 'No normalized entities. Run /api/normalize first.'}), 400
            
            entities = normalization_result.normalized_entities
            logger.info(f"Building knowledge graph from {len(entities)} entities")
            
            # STEP 5: KNOWLEDGE GRAPH - Add entities as nodes
            knowledge_graph.add_entities(entities)
            
            # Auto-discover relationships (Requirement -> Rule -> Code -> API -> DB)
            use_llm = Config.AUTO_DISCOVER_RELATIONSHIPS
            knowledge_graph.auto_discover_relationships(
                entities,
                llm_service=llm_service if use_llm else None
            )
            
            pipeline_state['knowledge_graph_built'] = True
            
            stats = knowledge_graph.get_statistics()
            orphaned = knowledge_graph.get_orphaned_entities()
            
            return jsonify({
                'message': 'Knowledge graph built successfully',
                'statistics': stats,
                'orphaned_entities': orphaned,
                'next_step': '/api/validate'
            }), 200

    except Exception as e:
        logger.error(f"Knowledge graph build error: {str(e)}")
        return jsonify({'error': f'Knowledge graph build failed: {str(e)}'}), 500


@app.route('/api/knowledge-graph', methods=['GET'])
def get_knowledge_graph():
    """Get knowledge graph statistics and visualization"""
    try:
        if not Config.ENABLE_KNOWLEDGE_GRAPH:
            return jsonify({'error': 'Knowledge graph is disabled'}), 503
        
        stats = knowledge_graph.get_statistics()
        return jsonify({
            'statistics': stats,
            'export_url': '/api/knowledge-graph/export'
        }), 200

    except Exception as e:
        logger.error(f"Knowledge graph error: {str(e)}")
        return jsonify({'error': f'Knowledge graph failed: {str(e)}'}), 500


@app.route('/api/knowledge-graph/export', methods=['GET'])
def export_knowledge_graph():
    """Export knowledge graph to JSON"""
    try:
        if not Config.ENABLE_KNOWLEDGE_GRAPH:
            return jsonify({'error': 'Knowledge graph is disabled'}), 503
        
        graph_json = knowledge_graph.export_to_json()
        return app.response_class(
            response=graph_json,
            status=200,
            mimetype='application/json'
        )
    except Exception as e:
        logger.error(f"Graph export error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/knowledge-graph/subgraph/<entity_id>', methods=['GET'])
def get_subgraph(entity_id):
    """Get visualization of subgraph around an entity"""
    try:
        if not Config.ENABLE_KNOWLEDGE_GRAPH:
            return jsonify({'error': 'Knowledge graph is disabled'}), 503
        
        depth = request.args.get('depth', default=1, type=int)
        subgraph = knowledge_graph.visualize_subgraph(entity_id, depth)
        return jsonify(subgraph), 200
    except Exception as e:
        logger.error(f"Subgraph error: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ==================== Document Generation Routes ====================

@app.route('/api/generate/requirements', methods=['POST'])
def generate_requirements():
    """Generate requirements document from extracted entities"""
    try:
        if not Config.ENABLE_AUTO_GENERATION:
            return jsonify({'error': 'Auto-generation is disabled'}), 503
        
        with observability.track_operation("requirements_generation"):
            normalization_result = pipeline_state.get('normalization_result')
            if not normalization_result:
                return jsonify({'error': 'No normalized entities. Run the pipeline first.'}), 400
            
            # Filter requirement entities
            req_entities = [
                e for e in normalization_result.normalized_entities
                if e.entity_type == EntityType.REQUIREMENT
            ]
            
            if not req_entities:
                return jsonify({'error': 'No requirement entities found'}), 400
            
            # Convert to dicts for LLM service
            entity_dicts = [{
                'text': e.canonical_text,
                'type': e.entity_type.value,
                'attributes': e.unified_attributes,
                'confidence': sum(t.confidence_score for t in e.traceability) / max(1, len(e.traceability))
            } for e in req_entities]
            
            logger.info(f"Generating requirements document from {len(entity_dicts)} entities")
            
            # STEP 8: OUTPUT GENERATION
            content = llm_service.generate_requirements_document(entity_dicts)
            
            doc = GeneratedDocument(
                document_id=f"req_{int(datetime.utcnow().timestamp())}",
                document_type="requirements",
                title="Generated Requirements Document",
                content=content,
                source_entities=[e.canonical_id for e in req_entities],
                requires_review=True
            )
            
            observability.record_generation("requirements", 0.0, len(entity_dicts))
            
            return jsonify({
                'document_id': doc.document_id,
                'type': doc.document_type,
                'title': doc.title,
                'content': doc.content,
                'source_entity_count': len(doc.source_entities),
                'source_entities': doc.source_entities,
                'requires_review': doc.requires_review,
                'generated_at': doc.generation_timestamp.isoformat()
            }), 200

    except Exception as e:
        logger.error(f"Requirements generation error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/generate/design', methods=['POST'])
def generate_design():
    """Generate design document from extracted entities"""
    try:
        if not Config.ENABLE_AUTO_GENERATION:
            return jsonify({'error': 'Auto-generation is disabled'}), 503
        
        with observability.track_operation("design_generation"):
            normalization_result = pipeline_state.get('normalization_result')
            if not normalization_result:
                return jsonify({'error': 'No normalized entities. Run the pipeline first.'}), 400
            
            # Use design, entity, and API types for design docs
            design_entities = [
                e for e in normalization_result.normalized_entities
                if e.entity_type in (EntityType.DESIGN, EntityType.ENTITY, EntityType.API, EntityType.DATABASE)
            ]
            
            if not design_entities:
                return jsonify({'error': 'No design-related entities found'}), 400
            
            entity_dicts = [{
                'text': e.canonical_text,
                'type': e.entity_type.value,
                'attributes': e.unified_attributes,
            } for e in design_entities]
            
            content = llm_service.generate_design_document(entity_dicts)
            
            doc = GeneratedDocument(
                document_id=f"design_{int(datetime.utcnow().timestamp())}",
                document_type="design",
                title="Generated Design Document",
                content=content,
                source_entities=[e.canonical_id for e in design_entities],
                requires_review=True
            )
            
            observability.record_generation("design", 0.0, len(entity_dicts))
            
            return jsonify({
                'document_id': doc.document_id,
                'type': doc.document_type,
                'title': doc.title,
                'content': doc.content,
                'source_entity_count': len(doc.source_entities),
                'requires_review': doc.requires_review,
                'generated_at': doc.generation_timestamp.isoformat()
            }), 200

    except Exception as e:
        logger.error(f"Design generation error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/generate/rules', methods=['POST'])
def generate_rules():
    """Generate business rules document"""
    try:
        if not Config.ENABLE_AUTO_GENERATION:
            return jsonify({'error': 'Auto-generation is disabled'}), 503
        
        with observability.track_operation("rules_generation"):
            normalization_result = pipeline_state.get('normalization_result')
            if not normalization_result:
                return jsonify({'error': 'No normalized entities. Run the pipeline first.'}), 400
            
            rule_entities = [
                e for e in normalization_result.normalized_entities
                if e.entity_type == EntityType.RULE
            ]
            
            if not rule_entities:
                return jsonify({'error': 'No rule entities found'}), 400
            
            entity_dicts = [{
                'text': e.canonical_text,
                'type': e.entity_type.value,
                'attributes': e.unified_attributes,
            } for e in rule_entities]
            
            content = llm_service.generate_rules_document(entity_dicts)
            
            doc = GeneratedDocument(
                document_id=f"rules_{int(datetime.utcnow().timestamp())}",
                document_type="rules",
                title="Generated Business Rules Document",
                content=content,
                source_entities=[e.canonical_id for e in rule_entities],
                requires_review=True
            )
            
            observability.record_generation("rules", 0.0, len(entity_dicts))
            
            return jsonify({
                'document_id': doc.document_id,
                'type': doc.document_type,
                'title': doc.title,
                'content': doc.content,
                'source_entity_count': len(doc.source_entities),
                'requires_review': doc.requires_review,
                'generated_at': doc.generation_timestamp.isoformat()
            }), 200

    except Exception as e:
        logger.error(f"Rules generation error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/pipeline/run', methods=['POST'])
def run_full_pipeline():
    """Run the full pipeline: Extract -> Normalize -> Knowledge Graph -> Validate -> Generate"""
    try:
        with observability.track_operation("full_pipeline"):
            ingested_doc = pipeline_state.get('ingested_document')
            if not ingested_doc:
                return jsonify({'error': 'No ingested document. Upload a file first via /upload'}), 400
            
            results = {'document_id': ingested_doc.document_id, 'steps': {}}
            
            # STEP 3: AI EXTRACTION
            if Config.ENABLE_EXTRACTION_LAYER and extraction_layer:
                logger.info("Pipeline Step: Extraction")
                extraction_result = extraction_layer.extract_all_entities(ingested_doc)
                pipeline_state['extraction_result'] = extraction_result
                results['steps']['extraction'] = {
                    'total_entities': len(extraction_result.extracted_entities),
                    'errors': extraction_result.errors,
                    'warnings': extraction_result.warnings
                }
            else:
                return jsonify({'error': 'Extraction layer is disabled'}), 503
            
            # STEP 4: NORMALIZATION
            if Config.ENABLE_NORMALIZATION and normalization_engine:
                logger.info("Pipeline Step: Normalization")
                normalization_result = normalization_engine.normalize(extraction_result.extracted_entities)
                pipeline_state['normalization_result'] = normalization_result
                vector_store.add_normalized_entities(normalization_result.normalized_entities)
                results['steps']['normalization'] = {
                    'total_normalized': len(normalization_result.normalized_entities),
                    'duplicates_removed': normalization_result.duplicates_removed,
                    'merges_performed': normalization_result.merges_performed
                }
            else:
                return jsonify({'error': 'Normalization is disabled'}), 503
            
            # STEP 5: KNOWLEDGE GRAPH
            if Config.ENABLE_KNOWLEDGE_GRAPH and knowledge_graph:
                logger.info("Pipeline Step: Knowledge Graph")
                knowledge_graph.add_entities(normalization_result.normalized_entities)
                knowledge_graph.auto_discover_relationships(
                    normalization_result.normalized_entities,
                    llm_service=llm_service if Config.AUTO_DISCOVER_RELATIONSHIPS else None
                )
                pipeline_state['knowledge_graph_built'] = True
                results['steps']['knowledge_graph'] = knowledge_graph.get_statistics()
            
            # STEP 6: VALIDATION
            if Config.ENABLE_VALIDATION and validation_engine:
                logger.info("Pipeline Step: Validation")
                kg_edges = [e.dict() for e in knowledge_graph.edges.values()] if knowledge_graph else None
                validation_result = validation_engine.validate(
                    normalization_result.normalized_entities,
                    knowledge_graph_edges=kg_edges
                )
                pipeline_state['validation_result'] = validation_result
                results['steps']['validation'] = {
                    'is_valid': validation_result.is_valid,
                    'score': validation_result.validation_score,
                    'conflicts': len(validation_result.conflicts),
                    'gaps': len(validation_result.gaps),
                    'inconsistencies': len(validation_result.inconsistencies)
                }
            
            # STEP 7: TRACEABILITY SUMMARY
            traceability_summary = []
            for entity in normalization_result.normalized_entities:
                for trace in entity.traceability:
                    traceability_summary.append({
                        'entity_id': entity.canonical_id,
                        'entity_type': entity.entity_type.value,
                        'source': trace.source_id,
                        'confidence': trace.confidence_score,
                        'confidence_level': trace.confidence_level.value,
                        'extraction_method': trace.extraction_method,
                        'version': trace.source_metadata.version
                    })
            results['steps']['traceability'] = {
                'total_traces': len(traceability_summary),
                'traces': traceability_summary[:50]  # First 50 for brevity
            }
            
            results['pipeline_status'] = 'completed'
            results['next_steps'] = [
                '/api/generate/requirements',
                '/api/generate/design',
                '/api/generate/rules'
            ]
            
            return jsonify(results), 200

    except Exception as e:
        logger.error(f"Pipeline error: {str(e)}")
        return jsonify({'error': f'Pipeline failed: {str(e)}'}), 500


# ==================== Query/Search Routes ====================

# Credit decision keywords for routing
CREDIT_DECISION_KEYWORDS = [
    'credit decision', 'credit score', 'bureau score', 'loan decision',
    'approved', 'declined', 'refer', 'zoot', 'risk score', 'risk band',
    'dti', 'debt to income', 'loan approval', 'credit limit', 'interest rate',
    'customer decision', 'evaluate customer', 'credit check'
]


def _is_credit_decision_query(question: str) -> bool:
    """Check if the question is about credit decisions"""
    q_lower = question.lower()
    return any(kw in q_lower for kw in CREDIT_DECISION_KEYWORDS)


def _extract_customer_id(question: str) -> str:
    """Try to extract a customer ID from the question text"""
    import re
    # Match patterns like CI98765432, CI-12345, or quoted IDs
    match = re.search(r'\b(CI[\-]?\d{4,})\b', question, re.IGNORECASE)
    if match:
        return match.group(1).upper().replace('-', '')
    return None


@app.route('/query', methods=['POST'])
def query():
    """
    Smart query endpoint:
    1. If credit decision query with customer ID -> decision engine
    2. Search vector store for relevant context
    3. Use LLM with vector store context to answer
    """
    data = request.json
    if 'question' not in data:
        return jsonify({'error': 'No question provided'}), 400

    question = data['question'].strip()
    if not question:
        return jsonify({'error': 'Question cannot be empty'}), 400

    try:
        # --- Route 1: Credit decision query ---
        if _is_credit_decision_query(question):
            customer_id = _extract_customer_id(question)

            if customer_id:
                # Evaluate specific customer
                try:
                    result = decision_engine.evaluate(customer_id)
                    return jsonify({
                        'response': f"Credit decision for customer {customer_id}:",
                        'type': 'credit_decision',
                        'decision': result['response'],
                        'traceability': result['traceability']
                    })
                except ValueError as e:
                    # Customer not found in vector store
                    vs_ids = decision_engine.find_customer_ids_in_vector_store()
                    available = vs_ids or decision_engine.get_customer_ids()
                    return jsonify({
                        'response': str(e),
                        'type': 'credit_decision_error',
                        'available_customers': available
                    }), 404

            # No specific customer ID in query — find customers from vector store
            vs_customer_ids = decision_engine.find_customer_ids_in_vector_store()
            if not vs_customer_ids:
                vs_customer_ids = decision_engine.get_customer_ids()

            if vs_customer_ids:
                # Evaluate first customer found in vector store
                result = decision_engine.evaluate(vs_customer_ids[0])
                return jsonify({
                    'response': f"Credit decision for customer {vs_customer_ids[0]}:",
                    'type': 'credit_decision',
                    'decision': result['response'],
                    'traceability': result['traceability'],
                    'available_customers': vs_customer_ids
                })
            else:
                return jsonify({
                    'response': 'No customer IDs found in vector store. Please ingest customer data first.',
                    'type': 'credit_decision_error',
                    'available_customers': []
                }), 404

        # --- Route 2: Vector store search + LLM ---
        vector_results = vector_store.similarity_search(question, k=5)
        context_chunks = [doc.page_content for doc in vector_results]

        if context_chunks:
            # Build context-augmented prompt
            context_text = "\n\n---\n\n".join(context_chunks)
            augmented_prompt = (
                f"Use the following context from the knowledge base to answer the question. "
                f"If the context doesn't contain relevant information, say so.\n\n"
                f"Context:\n{context_text}\n\n"
                f"Question: {question}"
            )
            llm_response = llm_service.llm.predict(input=augmented_prompt)

            return jsonify({
                'response': llm_response,
                'type': 'vector_search',
                'sources': [doc.metadata.get('source', 'unknown') for doc in vector_results],
                'chunks_found': len(context_chunks)
            })
        else:
            # No vector results — plain LLM
            response = llm_service.get_response(question)
            return jsonify({
                'response': response,
                'type': 'llm_only'
            })

    except Exception as e:
        logger.error(f"Query error: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ==================== Feedback Routes ====================

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """Submit user feedback for continuous improvement"""
    try:
        if not Config.ENABLE_FEEDBACK_LOOP:
            return jsonify({'error': 'Feedback loop is disabled'}), 503
        
        data = request.json or {}
        
        feedback_id = feedback_loop.submit_feedback(
            entity_id=data.get('entity_id'),
            feedback_type=data.get('type'),
            original_text=data.get('original_text'),
            corrected_text=data.get('corrected_text'),
            notes=data.get('notes'),
            user_id=data.get('user_id')
        )
        
        return jsonify({
            'feedback_id': feedback_id,
            'message': 'Feedback recorded successfully'
        }), 201

    except Exception as e:
        logger.error(f"Feedback error: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ==================== Observability Routes ====================

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Get system metrics"""
    try:
        if not Config.ENABLE_METRICS:
            return jsonify({'error': 'Metrics are disabled'}), 503
        
        summary = observability.get_performance_summary()
        return jsonify(summary), 200
    except Exception as e:
        logger.error(f"Metrics error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/events', methods=['GET'])
def get_events():
    """Get event log"""
    try:
        event_type = request.args.get('type')
        limit = request.args.get('limit', default=100, type=int)
        
        events = observability.get_event_log(event_type=event_type, limit=limit)
        return jsonify({'events': events, 'total': len(events)}), 200
    except Exception as e:
        logger.error(f"Events error: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ==================== Credit Decision Routes ====================

@app.route('/api/decision/customers', methods=['GET'])
def list_customers():
    """List all available customer IDs from vector store"""
    try:
        # Retrieve customer IDs dynamically from vector store
        vs_customer_ids = decision_engine.find_customer_ids_in_vector_store()
        # Fallback to in-memory if vector store has none
        if not vs_customer_ids:
            vs_customer_ids = decision_engine.get_customer_ids()
        return jsonify({
            'customers': vs_customer_ids,
            'total': len(vs_customer_ids),
            'source': 'vector_store'
        }), 200
    except Exception as e:
        logger.error(f"Error listing customers: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/decision/<customer_id>', methods=['GET'])
def get_credit_decision(customer_id):
    """
    Get credit decision for a customer by ID.
    Customer ID is validated against the vector store.
    Flow: Ingest request into Vector Store -> Retrieve rules -> Apply rules -> Return response
    Response matches SampleZootResponse templates (Approved/Declined/Refer).
    """
    try:
        result = decision_engine.evaluate(customer_id)
        # Return both the clean Zoot response and traceability info
        return jsonify({
            **result['response'],
            'traceability': result['traceability']
        }), 200
    except ValueError:
        # Customer ID not found — retrieve available IDs from vector store
        vs_ids = decision_engine.find_customer_ids_in_vector_store()
        available = vs_ids or decision_engine.get_customer_ids()
        return jsonify({
            'error': f"Customer ID '{customer_id}' not found in vector store.",
            'message': 'Customer ID not found. Please provide a valid customer ID from the vector store.',
            'available_customers': available
        }), 404
    except Exception as e:
        logger.error(f"Decision error for {customer_id}: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/decision/evaluate', methods=['POST'])
def evaluate_credit_request():
    """
    Evaluate a credit decision for an ad-hoc Zoot request payload.
    Accepts a full Zoot request JSON, ingests it into Vector Store,
    retrieves matching rules, applies them, and returns the decision.
    Response matches SampleZootResponse templates (Approved/Declined/Refer).
    """
    try:
        request_data = request.json
        if not request_data:
            return jsonify({'error': 'Request body is required'}), 400
        if not request_data.get('applicant', {}).get('customerId'):
            return jsonify({'error': 'applicant.customerId is required'}), 400

        result = decision_engine.evaluate_request(request_data)
        return jsonify({
            **result['response'],
            'traceability': result['traceability']
        }), 200
    except Exception as e:
        logger.error(f"Decision evaluation error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/decision/ingest', methods=['POST'])
def ingest_request_to_vectordb():
    """
    Ingest a Zoot request into the Vector Store without evaluating.
    Useful for pre-loading customer data before decision calls.
    """
    try:
        request_data = request.json
        if not request_data:
            return jsonify({'error': 'Request body is required'}), 400

        result = decision_engine.ingest_request(request_data)
        return jsonify({
            'message': 'Request ingested into Vector Store',
            **result
        }), 200
    except Exception as e:
        logger.error(f"Ingest error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/decision/reload', methods=['POST'])
def reload_customer_data():
    """Reload customer data from the data directory"""
    try:
        decision_engine.reload_data()
        return jsonify({
            'message': 'Customer data reloaded',
            'customers': decision_engine.get_customer_ids()
        }), 200
    except Exception as e:
        logger.error(f"Reload error: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ==================== Error Handlers ====================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500


# ==================== Application Startup ====================

if __name__ == '__main__':
    logger.info("Starting Enterprise Knowledge Platform")
    logger.info(f"Configuration validated. Running in {'DEBUG' if Config.DEBUG_MODE else 'PRODUCTION'} mode")
    
    app.run(
        host=Config.API_HOST,
        port=Config.API_PORT,
        debug=Config.DEBUG_MODE
    )
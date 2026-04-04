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
from models.schemas import GeneratedDocument, UserFeedback

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

# Pipeline state tracker
pipeline_state = {
    'current_document_id': None,
    'extraction_result': None,
    'normalization_result': None,
    'validation_result': None
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
            supported_exts = {'.txt', '.pdf', '.docx', '.json'}
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
            
            # Add to vector store
            vector_store.add_text_chunks(ingested_doc.chunks)
            logger.debug("Document chunks added to vector store")
            
            observability.record_ingestion(
                source_type=str(ingested_doc.metadata.source),
                duration=0.0,  # TODO: track actual duration
                document_count=len(ingested_doc.chunks)
            )
            
            return jsonify({
                'message': 'Document ingested successfully',
                'document_id': ingested_doc.document_id,
                'chunks_processed': len(ingested_doc.chunks),
                'next_step': '/api/extract'
            }), 200

    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500


# ==================== Extraction Routes ====================

@app.route('/api/extract', methods=['POST'])
def extract_entities():
    """Extract entities from ingested document"""
    try:
        with observability.track_operation("entity_extraction"):
            data = request.json or {}
            document_id = data.get('document_id') or pipeline_state['current_document_id']
            
            if not document_id:
                return jsonify({'error': 'No document to extract from'}), 400
            
            if not Config.ENABLE_EXTRACTION_LAYER:
                return jsonify({'error': 'Extraction layer is disabled'}), 503
            
            # TODO: Retrieve ingested document by ID
            # For now, use vector store search as proxy
            logger.debug(f"Extracting entities from document {document_id}")
            
            # Would implement full extraction here
            # This is a placeholder returning mock data
            return jsonify({
                'message': 'Extraction would be performed here',
                'document_id': document_id,
                'next_step': '/api/normalize'
            }), 200

    except Exception as e:
        logger.error(f"Extraction error: {str(e)}")
        return jsonify({'error': f'Extraction failed: {str(e)}'}), 500


# ==================== Normalization Routes ====================

@app.route('/api/normalize', methods=['POST'])
def normalize_entities():
    """Normalize and deduplicate entities"""
    try:
        with observability.track_operation("entity_normalization"):
            if not Config.ENABLE_NORMALIZATION:
                return jsonify({'error': 'Normalization is disabled'}), 503
            
            logger.debug("Performing entity normalization")
            
            # Would implement normalization here
            return jsonify({
                'message': 'Normalization would be performed here',
                'next_step': '/api/validate'
            }), 200

    except Exception as e:
        logger.error(f"Normalization error: {str(e)}")
        return jsonify({'error': f'Normalization failed: {str(e)}'}), 500


# ==================== Validation Routes ====================

@app.route('/api/validate', methods=['POST'])
def validate_knowledge():
    """Validate knowledge and detect conflicts"""
    try:
        with observability.track_operation("knowledge_validation"):
            if not Config.ENABLE_VALIDATION:
                return jsonify({'error': 'Validation is disabled'}), 503
            
            logger.debug("Performing knowledge validation")
            
            # Would implement validation here
            return jsonify({
                'message': 'Validation would be performed here',
                'next_step': '/api/knowledge-graph'
            }), 200

    except Exception as e:
        logger.error(f"Validation error: {str(e)}")
        return jsonify({'error': f'Validation failed: {str(e)}'}), 500


# ==================== Knowledge Graph Routes ====================

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
    """Generate requirements document"""
    try:
        if not Config.ENABLE_AUTO_GENERATION:
            return jsonify({'error': 'Auto-generation is disabled'}), 503
        
        with observability.track_operation("requirements_generation"):
            data = request.json or {}
            entity_ids = data.get('entity_ids', [])
            
            logger.debug(f"Generating requirements from {len(entity_ids)} entities")
            
            # Would generate requirements here
            doc = GeneratedDocument(
                document_id=f"req_{datetime.utcnow().timestamp()}",
                document_type="requirements",
                title="Generated Requirements Document",
                content="Would be generated from extracted entities",
                source_entities=entity_ids
            )
            
            observability.record_generation("requirements", 0.0, len(entity_ids))
            
            return jsonify({
                'document_id': doc.document_id,
                'type': doc.document_type,
                'requires_review': doc.requires_review
            }), 200

    except Exception as e:
        logger.error(f"Requirements generation error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/generate/design', methods=['POST'])
def generate_design():
    """Generate design document"""
    try:
        if not Config.ENABLE_AUTO_GENERATION:
            return jsonify({'error': 'Auto-generation is disabled'}), 503
        
        # Similar to requirements generation
        return jsonify({'message': 'Design generation would be performed here'}), 200

    except Exception as e:
        logger.error(f"Design generation error: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ==================== Query/Search Routes ====================

@app.route('/query', methods=['POST'])
def query():
    """Query the knowledge base"""
    data = request.json
    if 'question' not in data:
        return jsonify({'error': 'No question provided'}), 400

    try:
        response = llm_service.get_response(data['question'])
        return jsonify({'response': response})
    except Exception as e:
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
"""
Observability Module - Tracks metrics, logging, and system performance
"""

import logging
import time
from typing import Dict, List, Any, Callable
from datetime import datetime
import json

from prometheus_client import Counter, Histogram, Gauge

logger = logging.getLogger(__name__)


class Observability:
    """Handles metrics, monitoring, and observability"""

    def __init__(self):
        """Initialize observability metrics"""
        
        # Counters
        self.documents_ingested = Counter(
            'documents_ingested_total',
            'Total documents ingested',
            ['source_type']
        )
        
        self.entities_extracted = Counter(
            'entities_extracted_total',
            'Total entities extracted',
            ['entity_type']
        )
        
        self.conflicts_detected = Counter(
            'conflicts_detected_total',
            'Total conflicts detected',
            ['conflict_type']
        )
        
        self.documents_generated = Counter(
            'documents_generated_total',
            'Total documents generated',
            ['document_type']
        )
        
        # Histograms for timing
        self.ingestion_duration = Histogram(
            'ingestion_duration_seconds',
            'Time spent on ingestion',
            ['source_type']
        )
        
        self.extraction_duration = Histogram(
            'extraction_duration_seconds',
            'Time spent on extraction',
            ['extraction_type']
        )
        
        self.validation_duration = Histogram(
            'validation_duration_seconds',
            'Time spent on validation'
        )
        
        # Gauges
        self.active_documents = Gauge(
            'active_documents',
            'Number of documents being processed'
        )
        
        self.vector_store_size = Gauge(
            'vector_store_size',
            'Number of vectors in store'
        )
        
        self.confidence_average = Gauge(
            'confidence_average',
            'Average confidence score',
            ['entity_type']
        )
        
        # Performance tracking
        self.operation_times = {}
        self.event_log = []

    def track_operation(self, operation_name: str) -> 'OperationTracker':
        """
        Context manager to track operation timing
        
        Args:
            operation_name: Name of operation to track
            
        Returns:
            OperationTracker context manager
        """
        return OperationTracker(self, operation_name)

    def log_event(self, event_type: str, details: Dict[str, Any], level: str = "INFO"):
        """
        Log an event for auditing
        
        Args:
            event_type: Type of event
            details: Event details
            level: Log level
        """
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": event_type,
            "details": details
        }
        
        self.event_log.append(event)
        
        log_func = getattr(logger, level.lower(), logger.info)
        log_func(f"{event_type}: {json.dumps(details)}")

    def record_ingestion(self, source_type: str, duration: float, document_count: int = 1):
        """Record ingestion metrics"""
        self.documents_ingested.labels(source_type=source_type).inc(document_count)
        self.ingestion_duration.labels(source_type=source_type).observe(duration)
        self.log_event("INGESTION", {
            "source_type": source_type,
            "duration_seconds": duration,
            "document_count": document_count
        })

    def record_extraction(self, extraction_type: str, duration: float, 
                         entity_types: Dict[str, int], avg_confidence: float):
        """Record extraction metrics"""
        for entity_type, count in entity_types.items():
            self.entities_extracted.labels(entity_type=entity_type).inc(count)
            self.confidence_average.labels(entity_type=entity_type).set(avg_confidence)
        
        self.extraction_duration.labels(extraction_type=extraction_type).observe(duration)
        self.log_event("EXTRACTION", {
            "extraction_type": extraction_type,
            "duration_seconds": duration,
            "entity_types": entity_types,
            "average_confidence": avg_confidence
        })

    def record_validation(self, duration: float, total_issues: int, 
                         validation_score: float):
        """Record validation metrics"""
        self.validation_duration.observe(duration)
        self.log_event("VALIDATION", {
            "duration_seconds": duration,
            "total_issues": total_issues,
            "validation_score": validation_score
        })

    def record_conflict(self, conflict_type: str, count: int):
        """Record detected conflicts"""
        for _ in range(count):
            self.conflicts_detected.labels(conflict_type=conflict_type).inc()
        
        self.log_event("CONFLICT", {
            "conflict_type": conflict_type,
            "count": count
        })

    def record_generation(self, document_type: str, duration: float, from_entity_count: int):
        """Record document generation"""
        self.documents_generated.labels(document_type=document_type).inc()
        self.log_event("GENERATION", {
            "document_type": document_type,
            "duration_seconds": duration,
            "source_entity_count": from_entity_count
        })

    def update_vector_store_size(self, size: int):
        """Update vector store monitoring"""
        self.vector_store_size.set(size)

    def update_active_documents(self, count: int):
        """Update active document count"""
        self.active_documents.set(count)

    def get_event_log(self, event_type: str = None, limit: int = 100) -> List[Dict]:
        """Get event log, optionally filtered by type"""
        events = self.event_log
        
        if event_type:
            events = [e for e in events if e['type'] == event_type]
        
        return events[-limit:]

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        return {
            "event_log_size": len(self.event_log),
            "total_events_by_type": self._count_events_by_type(),
            "operation_times": self.operation_times
        }

    def _count_events_by_type(self) -> Dict[str, int]:
        """Count events by type"""
        counts = {}
        for event in self.event_log:
            event_type = event['type']
            counts[event_type] = counts.get(event_type, 0) + 1
        return counts

    def export_metrics_json(self) -> str:
        """Export metrics as JSON"""
        return json.dumps({
            "event_log_size": len(self.event_log),
            "events_by_type": self._count_events_by_type(),
            "recent_events": self.get_event_log(limit=20)
        }, indent=2, default=str)


class OperationTracker:
    """Context manager for tracking operation timing and metrics"""

    def __init__(self, observability: Observability, operation_name: str):
        self.observability = observability
        self.operation_name = operation_name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        logger.debug(f"Starting operation: {self.operation_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        if exc_type:
            logger.error(f"Operation {self.operation_name} failed after {duration:.2f}s: {exc_val}")
            self.observability.log_event(
                "OPERATION_ERROR",
                {
                    "operation": self.operation_name,
                    "duration_seconds": duration,
                    "error": str(exc_val)
                },
                level="ERROR"
            )
        else:
            logger.debug(f"Operation {self.operation_name} completed in {duration:.2f}s")
            self.observability.log_event(
                "OPERATION_SUCCESS",
                {
                    "operation": self.operation_name,
                    "duration_seconds": duration
                }
            )
        
        self.observability.operation_times[self.operation_name] = duration
        return False  # Don't suppress exceptions


class FeedbackLoop:
    """Manages user feedback for continuous improvement"""

    def __init__(self):
        self.feedback_items = []
        self.feedback_counter = 0

    def submit_feedback(self, entity_id: str, feedback_type: str, 
                       original_text: str, corrected_text: str = None, 
                       notes: str = None, user_id: str = None) -> str:
        """
        Submit user feedback for an entity
        
        Args:
            entity_id: ID of entity being reviewed
            feedback_type: Type of feedback (correction, suggestion, validation, edit)
            original_text: Original extracted text
            corrected_text: Corrected text (if applicable)
            notes: Additional notes
            user_id: ID of user providing feedback
            
        Returns:
            Feedback ID
        """
        feedback_id = f"feedback_{self.feedback_counter}"
        self.feedback_counter += 1
        
        feedback = {
            "feedback_id": feedback_id,
            "entity_id": entity_id,
            "type": feedback_type,
            "original_text": original_text,
            "corrected_text": corrected_text,
            "notes": notes,
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.feedback_items.append(feedback)
        logger.info(f"Feedback submitted: {feedback_id} for entity {entity_id}")
        
        return feedback_id

    def get_feedback_for_entity(self, entity_id: str) -> List[Dict]:
        """Get all feedback for an entity"""
        return [f for f in self.feedback_items if f['entity_id'] == entity_id]

    def generate_improvement_suggestions(self) -> List[str]:
        """
        Analyze feedback to generate improvement suggestions
        
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        
        # Count feedback types
        correction_count = sum(1 for f in self.feedback_items if f['type'] == 'correction')
        suggestion_count = sum(1 for f in self.feedback_items if f['type'] == 'suggestion')
        
        if correction_count > len(self.feedback_items) * 0.3:
            suggestions.append("High volume of corrections detected. Consider reviewing extraction prompts.")
        
        if suggestion_count > 0:
            suggestions.append(f"Received {suggestion_count} suggestions for improvement. Review feedback for patterns.")
        
        return suggestions

    def export_feedback_json(self) -> str:
        """Export all feedback as JSON"""
        return json.dumps(self.feedback_items, indent=2, default=str)

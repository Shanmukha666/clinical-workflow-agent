"""
Ingestion Agent - Entry point for all data
Learns from past ingestion patterns and communicates with other agents
"""

from __future__ import annotations
import json
import os
import sys
import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional
import re

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
from .base_agent import DynamicAgent, LearningSignal as AgentMessage
from .communication import MessageBroker
from .memory import PersistentMemory
logger = logging.getLogger(__name__)
# Import services (will be implemented later)
try:
    from services.ocr import extract_text_from_image
    from services.parser import extract_text_from_pdf
except ImportError:
    # Placeholder for now
    def extract_text_from_image(path): return f"OCR text from {path}"
    def extract_text_from_pdf(path): return f"PDF text from {path}"


class IngestionAgent(DynamicAgent):
    """
    Ingestion Agent:
    - Accepts various input formats (text, PDF, images)
    - Normalizes data for downstream agents
    - Learns from past ingestion patterns
    - Communicates with other agents about new cases
    """
    
    def __init__(self, memory: Optional[PersistentMemory] = None, broker: Optional[MessageBroker] = None):
        super().__init__("IngestionAgent", memory)
        self.broker = broker or MessageBroker()
        self.supported_formats = ['txt', 'pdf', 'png', 'jpg', 'jpeg', 'json']
        
        # Learning parameters
        self.pattern_cache = {}  # Cache of learned extraction patterns
        self.extraction_success_rate = {}  # Track extraction success
        
        # Subscribe to relevant events
        self.subscribe("feedback_received")
        self.subscribe("agent_error")
        
        # Register with broker if available
        if self.broker:
            self.broker.register_agent(self)
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing method for ingestion
        """
        # Extract input
        file_path = data.get("file")
        raw_text = data.get("raw_data")
        patient_id = data.get("patient_id")
        case_id = data.get("case_id") or self._generate_case_id(patient_id)
        
        # Determine input type and extract content
        content = None
        input_type = None
        confidence = 0.0
        
        if file_path:
            # File-based ingestion
            content, input_type, confidence = self._ingest_file(file_path)
        elif raw_text:
            # Text-based ingestion
            content = raw_text
            input_type = "text"
            confidence = 1.0
        else:
            raise ValueError("No input source provided (file or raw_data required)")
        
        # Check for similar historical cases (learning)
        similar_cases = self._find_similar_cases(content)
        if similar_cases:
            data["similar_cases"] = similar_cases
            data["similarity_alert"] = f"Found {len(similar_cases)} similar historical cases"
        
        # Validate and clean content
        cleaned_content = self._clean_text(content)
        
        # Create ingestion record
        ingestion_record = {
            "case_id": case_id,
            "patient_id": patient_id,
            "input_type": input_type,
            "raw_data": cleaned_content,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
            "file_info": {
                "original_file": file_path,
                "size_bytes": os.path.getsize(file_path) if file_path and os.path.exists(file_path) else 0
            } if file_path else None
        }
        
        # Store in memory
        if self.memory:
            self.memory.store_case({
                **ingestion_record,
                "status": "ingested"
            })
        
        # Notify other agents about new case
        self._broadcast_new_case(case_id, ingestion_record)
        
        # Query other agents for relevant info
        self._query_historical_context(case_id, patient_id)
        
        return {
            "status": "success",
            "case_id": case_id,
            "patient_id": patient_id,
            "input_type": input_type,
            "content_length": len(cleaned_content),
            "confidence": confidence,
            "similar_cases_found": len(similar_cases) if similar_cases else 0,
            "next_agent": "ExtractionAgent"
        }
    
    def _ingest_file(self, file_path: str) -> tuple:
        """Ingest file and extract text"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_extension = file_path.split('.')[-1].lower()
        
        if file_extension not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        # Track success rate for learning
        if file_extension not in self.extraction_success_rate:
            self.extraction_success_rate[file_extension] = {"success": 0, "total": 0}
        
        try:
            if file_extension in ['png', 'jpg', 'jpeg']:
                text = extract_text_from_image(file_path)
                confidence = self._calculate_ocr_confidence(text, file_path)
                input_type = "image"
            elif file_extension == 'pdf':
                text = extract_text_from_pdf(file_path)
                confidence = 0.9 if len(text) > 100 else 0.5
                input_type = "pdf"
            elif file_extension == 'txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                confidence = 1.0
                input_type = "text"
            else:
                text = json.dumps(json.load(open(file_path)))
                confidence = 1.0
                input_type = "json"
            
            # Update success tracking
            self.extraction_success_rate[file_extension]["success"] += 1
            
            return text, input_type, confidence
            
        except Exception as e:
            # Track failure for learning
            self.extraction_success_rate[file_extension]["total"] += 1
            raise Exception(f"Failed to ingest {file_path}: {str(e)}")
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize line breaks
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remove special characters but keep clinical notation
        text = re.sub(r'[^\w\s\-:.,/()%]', '', text)
        
        return text.strip()
    
    def _find_similar_cases(self, content: str) -> List[Dict]:
        """Find similar cases in memory (learning)"""
        if not self.memory:
            return []
        
        similar = []
        content_hash = hashlib.md5(content[:500].encode()).hexdigest()
        
        # Simple similarity check - in production use embeddings
        for case_id, case in self.memory.cases.items():
            case_content = case.get("data", {}).get("raw_data", "")
            if case_content and len(case_content) > 100:
                case_hash = hashlib.md5(case_content[:500].encode()).hexdigest()
                if case_hash == content_hash:
                    similar.append({
                        "case_id": case_id,
                        "similarity": 1.0,
                        "outcome": case.get("data", {}).get("decision", {})
                    })
        
        return similar[:3]  # Return top 3
    
    def _calculate_ocr_confidence(self, text: str, file_path: str) -> float:
        """Calculate confidence in OCR extraction"""
        # Simple heuristic - in production use OCR engine's confidence
        words = len(text.split())
        
        if words < 10:
            return 0.3
        elif words < 50:
            return 0.6
        else:
            return 0.9
    
    def _broadcast_new_case(self, case_id: str, case_data: Dict):
        """Notify other agents about new case"""
        message = AgentMessage(
            sender=self.name,
            receiver="broadcast",
            message_type="new_case",
            payload={
                "case_id": case_id,
                "case_data": case_data
            }
        )
        
        if self.broker:
            self.broker.send(message)
    
    def _query_historical_context(self, case_id: str, patient_id: str):
        """Query other agents for historical context"""
        if patient_id and self.memory:
            # Get patient history
            history = self.memory.get_case_history(patient_id)
            
            if history:
                # Send to decision agent for context
                message = AgentMessage(
                    sender=self.name,
                    receiver="DecisionAgent",
                    message_type="historical_context",
                    payload={
                        "case_id": case_id,
                        "patient_id": patient_id,
                        "history": history
                    }
                )
                
                if self.broker:
                    self.broker.send(message)
    
    def _generate_case_id(self, patient_id: Optional[str]) -> str:
        """Generate unique case ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        if patient_id:
            return f"CASE-{patient_id}-{timestamp}"
        return f"CASE-{timestamp}-{os.urandom(4).hex()}"
    
    def _handle_broadcast(self, message: AgentMessage):
        """Handle broadcast messages"""
        if message.message_type == "feedback_received":
            # Learn from feedback
            feedback = message.payload
            self._learn_from_feedback(feedback)
        elif message.message_type == "agent_error":
            # Handle errors from other agents
            error = message.payload
            logger.warning(f"Received error from {error['agent']}: {error['error']}")
    
    def _learn_from_feedback(self, feedback: Dict):
        """Learn from feedback to improve ingestion"""
        if feedback.get("doctor_feedback") == "incorrect":
            # Track what went wrong
            extraction_confidence = feedback.get("extraction_confidence", 0)
            if extraction_confidence < 0.8:
                # Low extraction confidence may indicate ingestion issues
                self.pattern_cache["low_confidence_trigger"] = self.pattern_cache.get("low_confidence_trigger", 0) + 1
                
                if self.pattern_cache["low_confidence_trigger"] > 5:
                    logger.info("Learning: Multiple low confidence extractions - consider improving OCR")

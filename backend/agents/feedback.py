"""
Feedback Agent - Collects feedback and enables learning
Drives system improvement through continuous learning
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np

from .base_agent import DynamicAgent, LearningSignal as AgentMessage
from .communication import MessageBroker
from .memory import PersistentMemory, LearningEngine

logger = logging.getLogger(__name__)


class FeedbackAgent(DynamicAgent):
    """
    Feedback Agent:
    - Collects and processes doctor feedback
    - Extracts learning patterns
    - Drives system adaptation
    - Tracks performance metrics
    """
    
    def __init__(self, memory: Optional[PersistentMemory] = None, broker: Optional[MessageBroker] = None):
        super().__init__("FeedbackAgent", memory)
        self.broker = broker or MessageBroker()
        
        # Learning components
        self.learning_engine = LearningEngine(memory) if memory else None
        self.feedback_buffer: List[Dict] = []
        self.performance_metrics: Dict[str, List[float]] = {
            "accuracy": [],
            "precision": [],
            "recall": []
        }
        
        # Register with broker
        if self.broker:
            self.broker.register_agent(self)
        
        logger.info("FeedbackAgent initialized")
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process feedback and generate learning signals
        """
        # Validate feedback
        if not self._validate_feedback(data):
            return {
                "status": "error",
                "error": "Invalid feedback format"
            }
        
        # Enrich feedback with context
        enriched_feedback = self._enrich_feedback(data)
        
        # Store feedback
        if self.memory:
            self.memory.store_feedback(enriched_feedback)
        
        # Add to buffer for batch processing
        self.feedback_buffer.append(enriched_feedback)
        
        # Extract learning patterns
        learning_signals = self._extract_learning_signals(enriched_feedback)
        
        # Generate performance metrics
        metrics = self._update_performance_metrics(enriched_feedback)
        
        # Check if batch learning should trigger
        if len(self.feedback_buffer) >= 10:  # Batch size
            self._trigger_batch_learning()
        
        # Broadcast feedback for other agents
        self._broadcast_feedback(enriched_feedback)
        
        return {
            "status": "success",
            "feedback_id": enriched_feedback["feedback_id"],
            "learning_signals": learning_signals,
            "current_accuracy": metrics.get("accuracy", 0),
            "suggestions": self._generate_suggestions(enriched_feedback)
        }
    
    def _validate_feedback(self, feedback: Dict) -> bool:
        """Validate feedback format"""
        required = ["decision_id", "doctor_feedback"]
        
        for field in required:
            if field not in feedback:
                logger.error(f"Missing required field: {field}")
                return False
        
        if feedback["doctor_feedback"] not in ["correct", "incorrect"]:
            logger.error(f"Invalid doctor_feedback: {feedback['doctor_feedback']}")
            return False
        
        return True
    
    def _enrich_feedback(self, feedback: Dict) -> Dict:
        """Enrich feedback with additional context"""
        # Get original decision from memory
        original_decision = None
        hemoglobin = None
        risk_level = None
        
        if self.memory:
            case = self.memory.get_case(feedback["decision_id"])
            if case:
                decision = case.get("data", {}).get("decision", {})
                original_decision = decision
                hemoglobin = decision.get("hemoglobin_value")
                risk_level = decision.get("risk_level")
        
        return {
            "feedback_id": f"FB-{datetime.now().timestamp()}",
            "decision_id": feedback["decision_id"],
            "doctor_feedback": feedback["doctor_feedback"],
            "correct_label": feedback.get("correct_label"),
            "notes": feedback.get("notes"),
            "original_risk": risk_level,
            "hemoglobin_value": hemoglobin,
            "original_decision": original_decision,
            "timestamp": datetime.now().isoformat()
        }
    
    def _extract_learning_signals(self, feedback: Dict) -> List[Dict]:
        """Extract learning signals from feedback"""
        signals = []
        
        if feedback["doctor_feedback"] == "incorrect":
            # Signal 1: Incorrect risk classification
            signals.append({
                "type": "misclassification",
                "severity": "high",
                "original_risk": feedback["original_risk"],
                "correct_risk": feedback["correct_label"],
                "hemoglobin": feedback["hemoglobin_value"],
                "suggestion": f"Review thresholds for {feedback['original_risk']} risk level"
            })
            
            # Check if this is a threshold boundary error
            if feedback["hemoglobin_value"]:
                hb = feedback["hemoglobin_value"]
                if feedback["original_risk"] == "LOW" and feedback["correct_label"] == "MODERATE":
                    # Under-classified
                    signals.append({
                        "type": "threshold_boundary",
                        "direction": "increase_sensitivity",
                        "current_threshold": self._get_current_threshold("moderate"),
                        "suggested_threshold": hb + 0.5,
                        "confidence": 0.7
                    })
                elif feedback["original_risk"] == "HIGH" and feedback["correct_label"] == "MODERATE":
                    # Over-classified
                    signals.append({
                        "type": "threshold_boundary",
                        "direction": "decrease_sensitivity",
                        "current_threshold": self._get_current_threshold("critical"),
                        "suggested_threshold": hb - 0.5,
                        "confidence": 0.7
                    })
        
        return signals
    
    def _update_performance_metrics(self, feedback: Dict) -> Dict[str, float]:
        """Update running performance metrics"""
        # Track accuracy over time
        correct = feedback["doctor_feedback"] == "correct"
        self.performance_metrics["accuracy"].append(1.0 if correct else 0.0)
        
        # Keep only last 100
        if len(self.performance_metrics["accuracy"]) > 100:
            self.performance_metrics["accuracy"] = self.performance_metrics["accuracy"][-100:]
        
        # Calculate rolling accuracy
        recent_accuracy = np.mean(self.performance_metrics["accuracy"][-20:]) if self.performance_metrics["accuracy"] else 0
        
        return {
            "accuracy": recent_accuracy,
            "total_feedback": len(self.performance_metrics["accuracy"]),
            "trend": self._calculate_trend()
        }
    
    def _calculate_trend(self) -> str:
        """Calculate performance trend"""
        if len(self.performance_metrics["accuracy"]) < 10:
            return "insufficient_data"
        
        recent = np.mean(self.performance_metrics["accuracy"][-10:])
        older = np.mean(self.performance_metrics["accuracy"][-20:-10]) if len(self.performance_metrics["accuracy"]) >= 20 else recent
        
        if recent > older + 0.05:
            return "improving"
        elif recent < older - 0.05:
            return "declining"
        else:
            return "stable"
    
    def _get_current_threshold(self, threshold_type: str) -> float:
        """Get current threshold value"""
        # In production, this would come from config
        thresholds = {
            "critical": 7.0,
            "moderate": 10.0
        }
        return thresholds.get(threshold_type, 0)
    
    def _trigger_batch_learning(self):
        """Trigger batch learning from accumulated feedback"""
        if len(self.feedback_buffer) < 10:
            return
        
        logger.info(f"Triggering batch learning with {len(self.feedback_buffer)} feedback items")
        
        # Analyze patterns
        incorrect = [f for f in self.feedback_buffer if f["doctor_feedback"] == "incorrect"]
        
        if len(incorrect) >= 3:
            # Look for patterns
            by_risk = {}
            for f in incorrect:
                risk = f["original_risk"]
                if risk not in by_risk:
                    by_risk[risk] = []
                by_risk[risk].append(f)
            
            # Suggest adjustments for each risk level
            for risk, cases in by_risk.items():
                if len(cases) >= 3:
                    hb_values = [c["hemoglobin_value"] for c in cases if c["hemoglobin_value"]]
                    if hb_values:
                        avg_hb = np.mean(hb_values)
                        logger.info(f"Learning: Risk {risk} misclassified. Avg Hb: {avg_hb:.1f}")
                        
                        # Broadcast learning signal
                        self._broadcast_learning_signal({
                            "type": "threshold_adjustment",
                            "risk_level": risk,
                            "suggested_threshold": avg_hb,
                            "confidence": min(0.9, len(cases) / 10)
                        })
        
        # Clear buffer
        self.feedback_buffer = []
    
    def _generate_suggestions(self, feedback: Dict) -> List[str]:
        """Generate actionable suggestions"""
        suggestions = []
        
        if feedback["doctor_feedback"] == "incorrect":
            if feedback["original_risk"] == "LOW" and feedback["correct_label"] == "MODERATE":
                suggestions.append("Consider lowering MODERATE risk threshold")
            elif feedback["original_risk"] == "HIGH" and feedback["correct_label"] == "MODERATE":
                suggestions.append("Consider raising CRITICAL risk threshold")
            elif feedback["original_risk"] == "MODERATE" and feedback["correct_label"] == "LOW":
                suggestions.append("Consider raising MODERATE risk threshold")
            elif feedback["original_risk"] == "MODERATE" and feedback["correct_label"] == "HIGH":
                suggestions.append("Consider lowering MODERATE risk threshold")
        
        return suggestions
    
    def _broadcast_feedback(self, feedback: Dict):
        """Broadcast feedback to all agents for learning"""
        message = AgentMessage(
            sender=self.name,
            receiver="broadcast",
            message_type="feedback_received",
            payload=feedback
        )
        
        if self.broker:
            self.broker.send(message)
    
    def _broadcast_learning_signal(self, signal: Dict):
        """Broadcast learning signal to decision agent"""
        message = AgentMessage(
            sender=self.name,
            receiver="DecisionAgent",
            message_type="learning_signal",
            payload=signal
        )
        
        if self.broker:
            self.broker.send(message)

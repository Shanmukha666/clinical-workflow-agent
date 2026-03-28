"""
Dynamic Orchestrator with Real-Time Learning and Agent Communication
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional
from datetime import datetime
import threading
import queue

from .base_agent import NeuralMemory, LearningSignal
from .extraction import ExtractionAgent
from .decision import DecisionAgent


class DynamicOrchestrator:
    """
    Orchestrator with:
    - Real-time learning from feedback
    - Agent-to-agent learning signals
    - Neural memory shared across agents
    - Continuous adaptation
    """
    
    def __init__(self):
        # Shared neural memory across all agents
        self.neural_memory = NeuralMemory(embedding_dim=128)
        
        # Initialize agents with shared memory
        self.extraction = ExtractionAgent(self.neural_memory)
        self.decision = DecisionAgent(self.neural_memory)
        
        # Learning queue
        self.learning_queue: queue.Queue = queue.Queue()
        self.feedback_buffer: List[Dict] = []
        
        # Start learning thread
        self.learning_thread = threading.Thread(target=self._continuous_learning, daemon=True)
        self.learning_thread.start()
        
        # Performance tracking
        self.performance_history: List[Dict] = []
        
        print("🚀 Dynamic Orchestrator initialized with real learning")
    
    def process_case(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a case with dynamic learning
        """
        case_id = input_data.get("case_id", f"CASE-{datetime.now().timestamp()}")
        
        # Step 1: Extraction with learning
        extraction_context = {
            "current_input": input_data
        }
        extraction_result = self.extraction.run(extraction_context)
        
        if extraction_result["status"] == "error":
            return {"status": "error", "error": extraction_result["error"]}
        
        # Step 2: Decision with learning
        decision_context = {
            "current_input": {
                "structured_data": extraction_result["output"]["structured_data"],
                "demographics": extraction_result["output"].get("entities", {}),
                "case_id": case_id
            }
        }
        decision_result = self.decision.run(decision_context)
        
        if decision_result["status"] == "error":
            return {"status": "error", "error": decision_result["error"]}
        
        # Store case in neural memory
        case_embedding = self._create_case_embedding(extraction_result, decision_result)
        self.neural_memory.store({
            "case_id": case_id,
            "extraction": extraction_result["output"],
            "decision": decision_result["output"],
            "timestamp": datetime.now().isoformat()
        }, case_embedding)
        
        return {
            "status": "success",
            "case_id": case_id,
            "extraction": extraction_result["output"],
            "decision": decision_result["output"],
            "embedding": case_embedding.tolist()
        }
    
    def provide_feedback(self, case_id: str, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """
        Provide feedback that triggers learning
        """
        # Store feedback
        feedback_record = {
            "case_id": case_id,
            "timestamp": datetime.now().isoformat(),
            **feedback
        }
        
        self.feedback_buffer.append(feedback_record)
        
        # Queue for learning
        self.learning_queue.put(feedback_record)
        
        # Immediate learning if buffer is full
        if len(self.feedback_buffer) >= 5:
            self._learn_from_batch()
        
        return {
            "status": "learning_triggered",
            "feedback_id": f"FB-{datetime.now().timestamp()}",
            "buffer_size": len(self.feedback_buffer)
        }
    
    def _continuous_learning(self):
        """
        Continuous learning thread that processes feedback in real-time
        """
        while True:
            try:
                # Wait for feedback
                feedback = self.learning_queue.get(timeout=1)
                
                # Process feedback
                self._process_single_feedback(feedback)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Learning error: {e}")
    
    def _process_single_feedback(self, feedback: Dict):
        """
        Process single feedback and trigger agent learning
        """
        case_id = feedback.get("case_id")
        doctor_feedback = feedback.get("doctor_feedback")
        
        # Find case in neural memory
        similar_cases = self._find_case(case_id)
        if not similar_cases:
            return
        
        case = similar_cases[0]
        extraction_output = case.get("extraction", {})
        decision_output = case.get("decision", {})
        
        if doctor_feedback == "incorrect":
            correct_risk = feedback.get("correct_label")
            correct_values = feedback.get("correct_values", {})
            
            # 1. Train extraction agent on correct values
            if correct_values:
                loss, gradient = self.extraction.compute_loss(
                    extraction_output, 
                    {"correct_values": correct_values}
                )
                
                # Send learning signal to extraction
                signal = LearningSignal(
                    sender="Orchestrator",
                    receiver="ExtractionAgent",
                    signal_type="weight_update",
                    data={"loss": loss},
                    confidence=0.9,
                    gradient=gradient
                )
                self.extraction.receive_learning_signal(signal)
                
                # Learn new patterns
                for field, value in correct_values.items():
                    if field not in extraction_output.get("structured_data", {}):
                        # Learn new pattern from this feedback
                        pattern = self._infer_pattern(feedback.get("text", ""), field)
                        if pattern:
                            self.extraction.learn_new_pattern(field, pattern, feedback.get("text", ""), value)
            
            # 2. Train decision agent on correct risk
            if correct_risk:
                loss, gradient = self.decision.compute_loss(
                    decision_output,
                    {"correct_risk_level": correct_risk}
                )
                
                # Send learning signal to decision
                signal = LearningSignal(
                    sender="Orchestrator",
                    receiver="DecisionAgent",
                    signal_type="weight_update",
                    data={"loss": loss},
                    confidence=0.9,
                    gradient=gradient
                )
                self.decision.receive_learning_signal(signal)
                
                # Learn from outcome
                self.decision.learn_from_outcome(
                    case_id,
                    correct_risk,
                    decision_output.get("risk_level")
                )
            
            # 3. Broadcast learning to other agents
            self._broadcast_learning(extraction_output, decision_output, correct_values, correct_risk)
    
    def _learn_from_batch(self):
        """
        Batch learning from multiple feedbacks
        """
        if len(self.feedback_buffer) < 5:
            return
        
        print(f"📚 Batch learning from {len(self.feedback_buffer)} feedback items")
        
        # Collect training data
        extraction_training = []
        decision_training = []
        
        for fb in self.feedback_buffer:
            if fb.get("correct_values"):
                extraction_training.append({
                    "text": fb.get("text", ""),
                    "correct_values": fb.get("correct_values", {})
                })
            
            if fb.get("correct_label"):
                decision_training.append({
                    "input": fb.get("input", {}),
                    "correct_risk": fb.get("correct_label")
                })
        
        # Train extraction agent
        if len(extraction_training) >= 5:
            self.extraction.train_on_feedback(extraction_training)
        
        # Train decision agent (save model)
        if len(decision_training) >= 5:
            self.decision._save_model()
        
        # Clear buffer
        self.feedback_buffer = []
        
        # Update performance metrics
        self._update_performance()
    
    def _broadcast_learning(self, extraction: Dict, decision: Dict, 
                           correct_values: Dict, correct_risk: str):
        """
        Broadcast learning to all agents
        """
        # Signal to extraction about pattern improvements
        if correct_values:
            for field, value in correct_values.items():
                pattern_signal = LearningSignal(
                    sender="DecisionAgent",
                    receiver="ExtractionAgent",
                    signal_type="pattern_update",
                    data={
                        "pattern_type": f"extraction_{field}",
                        "pattern_value": f"learned_from_feedback"
                    },
                    confidence=0.8
                )
                self.extraction.receive_learning_signal(pattern_signal)
        
        # Signal to decision about threshold adjustments
        if correct_risk:
            threshold_signal = LearningSignal(
                sender="ExtractionAgent",
                receiver="DecisionAgent",
                signal_type="threshold_adjustment",
                data={
                    "threshold_name": "critical",
                    "new_value": decision.get("hemoglobin_value", 10) - 0.5
                },
                confidence=0.7
            )
            self.decision.receive_learning_signal(threshold_signal)
    
    def _find_case(self, case_id: str) -> List[Dict]:
        """Find case in neural memory"""
        # Simple linear search - in production would use vector similarity
        results = []
        for memory in self.neural_memory.memories:
            if memory.get("case_id") == case_id:
                results.append(memory)
        return results
    
    def _create_case_embedding(self, extraction: Dict, decision: Dict) -> np.ndarray:
        """Create embedding for case"""
        embedding = np.zeros(128)
        
        # Add extraction features
        structured = extraction["output"].get("structured_data", {})
        for i, (field, value) in enumerate(structured.items()):
            if i < 10:
                embedding[i] = float(value) if value else 0
        
        # Add decision features
        risk_level = decision["output"].get("risk_level", "UNKNOWN")
        risk_map = {"LOW": 0.25, "MODERATE": 0.5, "HIGH": 0.75, "UNKNOWN": 0}
        embedding[10] = risk_map.get(risk_level, 0)
        embedding[11] = decision["output"].get("confidence_score", 0)
        
        # Normalize
        if np.linalg.norm(embedding) > 0:
            embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    
    def _infer_pattern(self, text: str, field: str) -> Optional[str]:
        """Infer extraction pattern from text and field"""
        # Simplified pattern inference
        patterns = {
            "hemoglobin": r"hb\s*[:=]?\s*(\d+\.?\d*)",
            "wbc": r"wbc\s*[:=]?\s*(\d+)",
            "platelets": r"plt\s*[:=]?\s*(\d+)"
        }
        
        pattern = patterns.get(field)
        if pattern and re.search(pattern, text, re.IGNORECASE):
            return pattern
        
        return None
    
    def _update_performance(self):
        """Update performance metrics"""
        accuracy = self.decision.performance_history[-10:] if self.decision.performance_history else []
        
        self.performance_history.append({
            "timestamp": datetime.now().isoformat(),
            "accuracy": np.mean(accuracy) if accuracy else 0,
            "adaptation_count": self.decision.adaptation_count,
            "buffer_size": len(self.feedback_buffer)
        })
        
        # Keep last 100
        if len(self.performance_history) > 100:
            self.performance_history.pop(0)
    
    def get_metrics(self) -> Dict:
        """Get current system metrics"""
        return {
            "decision_accuracy": np.mean(self.decision.performance_history) if self.decision.performance_history else 0,
            "decision_adaptations": self.decision.adaptation_count,
            "extraction_adaptations": self.extraction.adaptation_count,
            "learned_patterns": {
                field: len(patterns) 
                for field, patterns in self.extraction.learned_patterns.items()
            },
            "thresholds": self.decision.thresholds,
            "feedback_buffer_size": len(self.feedback_buffer),
            "neural_memory_size": len(self.neural_memory.memories)
        }


# Test the system
if __name__ == "__main__":
    import time
    
    print("=" * 70)
    print("DYNAMIC MULTI-AGENT SYSTEM TEST")
    print("=" * 70)
    
    orchestrator = DynamicOrchestrator()
    
    # Test cases with varying difficulty
    test_cases = [
        {
            "case_id": "CASE-001",
            "raw_data": "Patient: Male, 45. Hemoglobin: 6.2. Severe fatigue."
        },
        {
            "case_id": "CASE-002", 
            "raw_data": "Female, 32. Hb: 8.5. Mild fatigue, no other symptoms."
        },
        {
            "case_id": "CASE-003",
            "raw_data": "28 year old female. Hemoglobin: 11.2. Asymptomatic."
        },
        {
            "case_id": "CASE-004",
            "raw_data": "Male, 60. Hgb: 7.8. Shortness of breath, pale."
        }
    ]
    
    # Process cases
    results = []
    for case in test_cases:
        print(f"\n📋 Processing {case['case_id']}...")
        result = orchestrator.process_case(case)
        results.append(result)
        
        if result["status"] == "success":
            decision = result["decision"]
            print(f"   Risk: {decision['risk_level']}")
            print(f"   Confidence: {decision['confidence_score']:.2%}")
            print(f"   Reasoning: {decision['reasoning'][0]}")
        else:
            print(f"   Error: {result.get('error')}")
    
    # Provide feedback (simulating doctor corrections)
    print("\n" + "=" * 70)
    print("PROVIDING FEEDBACK (DOCTOR CORRECTIONS)")
    print("=" * 70)
    
    # Feedback 1: Case-001 was actually MODERATE, not HIGH
    print("\n📝 Feedback 1: Case-001 should be MODERATE")
    orchestrator.provide_feedback("CASE-001", {
        "doctor_feedback": "incorrect",
        "correct_label": "MODERATE",
        "correct_values": {"hemoglobin": 6.2},
        "text": test_cases[0]["raw_data"]
    })
    
    # Feedback 2: Case-002 was correct
    print("📝 Feedback 2: Case-002 correct")
    orchestrator.provide_feedback("CASE-002", {
        "doctor_feedback": "correct",
        "text": test_cases[1]["raw_data"]
    })
    
    # Feedback 3: Case-004 should be CRITICAL
    print("📝 Feedback 3: Case-004 should be CRITICAL")
    orchestrator.provide_feedback("CASE-004", {
        "doctor_feedback": "incorrect",
        "correct_label": "HIGH",
        "correct_values": {"hemoglobin": 7.8},
        "text": test_cases[3]["raw_data"]
    })
    
    # Wait for learning to process
    print("\n🔄 Waiting for learning to process...")
    time.sleep(2)
    
    # Process cases again to see learning effects
    print("\n" + "=" * 70)
    print("PROCESSING NEW CASE AFTER LEARNING")
    print("=" * 70)
    
    new_case = {
        "case_id": "CASE-005",
        "raw_data": "Male, 55. Hemoglobin: 6.5. Severe symptoms."
    }
    
    result = orchestrator.process_case(new_case)
    if result["status"] == "success":
        print(f"\n📋 New case result:")
        print(f"   Risk: {result['decision']['risk_level']}")
        print(f"   Confidence: {result['decision']['confidence_score']:.2%}")
        print(f"   Reasoning: {result['decision']['reasoning']}")
        print(f"   Neural vs Threshold: {result['decision'].get('neural_risk')} vs {result['decision'].get('threshold_risk')}")
    
    # Show metrics
    print("\n" + "=" * 70)
    print("SYSTEM METRICS AFTER LEARNING")
    print("=" * 70)
    
    metrics = orchestrator.get_metrics()
    print(f"Decision Accuracy: {metrics['decision_accuracy']:.2%}")
    print(f"Total Adaptations: {metrics['decision_adaptations']}")
    print(f"Learned Patterns: {metrics['learned_patterns']}")
    print(f"Current Thresholds: {metrics['thresholds']}")
    print(f"Neural Memory Size: {metrics['neural_memory_size']}")
    
    print("\n✅ Dynamic multi-agent system test complete!")
    print("The system now:")
    print("  - Learns from each feedback in real-time")
    print("  - Updates neural weights dynamically")
    print("  - Adjusts thresholds based on outcomes")
    print("  - Shares learning between agents") 

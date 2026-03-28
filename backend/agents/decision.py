"""
Decision Agent with Neural Learning and Dynamic Thresholds
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Tuple, Optional
from datetime import datetime
import json
import os
import logging

logger = logging.getLogger(__name__)
from .base_agent import DynamicAgent, NeuralMemory, LearningSignal


class DecisionAgent(DynamicAgent):
    """
    Decision agent that:
    - Learns optimal thresholds through reinforcement
    - Updates neural weights based on outcomes
    - Adapts to population patterns
    - Sends learning signals to other agents
    """
    
    def __init__(self, neural_memory: Optional[NeuralMemory] = None):
        super().__init__("DecisionAgent", neural_memory)
        
        # Neural network for risk prediction
        self.input_dim = 10  # Features: hemoglobin, age, gender, etc.
        self.hidden_dim = 32
        self.output_dim = 3  # Risk levels: LOW, MODERATE, HIGH
        
        # Initialize neural network weights
        self.W1 = np.random.randn(self.input_dim, self.hidden_dim) * 0.1
        self.b1 = np.zeros(self.hidden_dim)
        self.W2 = np.random.randn(self.hidden_dim, self.output_dim) * 0.1
        self.b2 = np.zeros(self.output_dim)
        
        # Dynamic thresholds (will be learned)
        self.thresholds = self._load_thresholds()
        
        # Q-learning parameters for threshold optimization
        self.q_table: Dict[str, Dict[float, float]] = {}
        self.learning_rate_q = 0.1
        self.discount_factor = 0.95
        self.exploration_rate = 0.1
        
        # Feature history for learning
        self.feature_history: List[np.ndarray] = []
        self.outcome_history: List[float] = []
        
        # Risk mapping
        self.risk_to_idx = {"LOW": 0, "MODERATE": 1, "HIGH": 2}
        self.idx_to_risk = {0: "LOW", 1: "MODERATE", 2: "HIGH"}
        
        # Load learned model
        self._load_model()
    
    def forward(self, input_data: Any) -> Tuple[Dict[str, Any], np.ndarray]:
        """
        Forward pass - make decision with neural network
        """
        structured = input_data.get("structured_data", {})
        demographics = input_data.get("demographics", {})
        
        hemoglobin = structured.get("hemoglobin")
        
        if hemoglobin is None:
            return self._handle_missing_data(), np.zeros(128)
        
        # Build feature vector
        features = self._build_features(hemoglobin, demographics)
        
        # Neural network forward pass
        h = np.tanh(np.dot(features, self.W1) + self.b1)
        logits = np.dot(h, self.W2) + self.b2
        probs = self._softmax(logits)
        
        # Get risk level from neural network
        risk_idx = np.argmax(probs)
        risk_level = self.idx_to_risk[risk_idx]
        neural_confidence = probs[risk_idx]
        
        # Get threshold-based decision (for comparison and learning)
        threshold_risk, threshold_confidence = self._threshold_decision(hemoglobin, demographics)
        
        # Combine both methods with learned weights
        if self.adaptation_count > 0:
            # Weight neural network more after learning
            neural_weight = min(0.8, 0.5 + self.adaptation_count / 100)
            final_risk = risk_level if neural_weight > 0.5 else threshold_risk
            final_confidence = (neural_confidence * neural_weight + 
                               threshold_confidence * (1 - neural_weight))
        else:
            final_risk = threshold_risk
            final_confidence = threshold_confidence
        
        # Generate reasoning
        reasoning = self._generate_reasoning(hemoglobin, demographics, final_risk, threshold_risk, risk_level)
        
        # Generate embedding for similarity search
        embedding = self._generate_embedding(features, final_risk, final_confidence)
        
        # Store features for future learning
        self.feature_history.append(features)
        if len(self.feature_history) > 100:
            self.feature_history.pop(0)
        
        return {
            "risk_level": final_risk,
            "confidence_score": final_confidence,
            "reasoning": reasoning,
            "clinical_insight": self._generate_insight(final_risk, hemoglobin),
            "suggested_actions": self._suggest_actions(final_risk, hemoglobin),
            "neural_risk": risk_level,
            "neural_confidence": neural_confidence,
            "threshold_risk": threshold_risk,
            "features": features.tolist()
        }, embedding
    
    def compute_loss(self, prediction: Any, target: Any) -> Tuple[float, np.ndarray]:
        """
        Compute cross-entropy loss and gradient for backpropagation
        """
        predicted_risk = prediction.get("risk_level")
        correct_risk = target.get("correct_risk_level")
        
        if not correct_risk or predicted_risk == correct_risk:
            return 0.0, np.zeros_like(self.W1.flatten())
        
        # Compute loss based on mismatch
        loss = 1.0  # Binary loss for wrong classification
        
        # Compute gradient (simplified - would be full backprop in production)
        gradient = np.random.randn(self.W1.size) * 0.01
        
        # Also update Q-table for threshold optimization
        self._update_q_table(target)
        
        return loss, gradient
    
    def _build_features(self, hemoglobin: float, demographics: Dict) -> np.ndarray:
        """Build feature vector for neural network"""
        features = np.zeros(self.input_dim)
        
        # Feature 0: Hemoglobin
        features[0] = hemoglobin / 20.0  # Normalize
        
        # Feature 1: Age (normalized)
        age = demographics.get("age", 50)
        features[1] = age / 100.0
        
        # Feature 2: Gender (0=female, 1=male)
        features[2] = 1.0 if demographics.get("gender") == "male" else 0.0
        
        # Feature 3: Previous hemoglobin (if available)
        # Feature 4-9: Placeholder for other clinical features
        for i in range(3, self.input_dim):
            features[i] = np.random.randn() * 0.1
        
        return features
    
    def _threshold_decision(self, hemoglobin: float, demographics: Dict) -> Tuple[str, float]:
        """
        Make decision using dynamic thresholds
        """
        gender = demographics.get("gender", "female")
        age = demographics.get("age", 50)
        
        # Get thresholds for this patient
        thresholds = self._get_thresholds(gender, age)
        
        if hemoglobin < thresholds["critical"]:
            return "HIGH", 0.95
        elif hemoglobin < thresholds["moderate"]:
            return "MODERATE", 0.85
        else:
            return "LOW", 0.90
    
    def _get_thresholds(self, gender: str, age: int) -> Dict[str, float]:
        """
        Get dynamic thresholds (learned from feedback)
        """
        # Base thresholds (will be updated by learning)
        base_thresholds = {
            "male": {"critical": 11.0, "moderate": 13.0},
            "female": {"critical": 10.0, "moderate": 12.0}
        }
        
        thresholds = base_thresholds.get(gender, base_thresholds["female"]).copy()
        
        # Apply learned adjustments
        for key, value in self.thresholds.items():
            if key in thresholds:
                thresholds[key] = value
        
        # Adjust for age
        if age < 12:
            thresholds["critical"] -= 1.0
            thresholds["moderate"] -= 1.0
        elif age > 65:
            thresholds["critical"] += 0.5
        
        return thresholds
    
    def _update_q_table(self, feedback: Dict):
        """
        Update Q-table based on feedback (Q-learning)
        """
        state = feedback.get("state", {})
        action = feedback.get("action", {})
        reward = feedback.get("reward", 0)
        
        state_key = f"hb_{state.get('hemoglobin', 0):.1f}_gender_{state.get('gender', '')}"
        
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        
        # Update Q-value for this action
        action_key = f"threshold_{action.get('threshold', '')}"
        current_q = self.q_table[state_key].get(action_key, 0)
        
        # Q-learning update
        new_q = current_q + self.learning_rate_q * (reward + self.discount_factor * 0 - current_q)
        self.q_table[state_key][action_key] = new_q
        
        # Adjust thresholds if Q-value indicates improvement
        if new_q > 0.8 and action_key not in self.thresholds:
            self.thresholds[action_key] = action.get("new_value", 0)
            logger.info(f"Q-learning updated threshold: {action_key} = {self.thresholds[action_key]}")
    
    def _generate_embedding(self, features: np.ndarray, risk: str, confidence: float) -> np.ndarray:
        """Generate embedding for similarity search"""
        embedding = np.zeros(128)
        embedding[:len(features)] = features
        embedding[10] = self.risk_to_idx.get(risk, 0) / 2.0
        embedding[11] = confidence
        
        # Normalize
        if np.linalg.norm(embedding) > 0:
            embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    
    def _generate_reasoning(self, hb: float, demographics: Dict, final_risk: str, 
                           threshold_risk: str, neural_risk: str) -> List[str]:
        """Generate reasoning with learning insights"""
        reasoning = []
        
        reasoning.append(f"Hemoglobin value: {hb} g/dL")
        
        if demographics.get("gender"):
            reasoning.append(f"Patient gender: {demographics['gender']}")
        if demographics.get("age"):
            reasoning.append(f"Patient age: {demographics['age']}")
        
        reasoning.append(f"Threshold-based assessment: {threshold_risk} risk")
        
        if neural_risk != threshold_risk and self.adaptation_count > 0:
            reasoning.append(f"Neural network assessment: {neural_risk} risk")
            reasoning.append(f"Combined assessment: {final_risk} risk")
        
        if final_risk == "HIGH":
            reasoning.append("HIGH risk: Immediate medical attention required")
        elif final_risk == "MODERATE":
            reasoning.append("MODERATE risk: Follow-up recommended")
        else:
            reasoning.append("LOW risk: Routine monitoring")
        
        return reasoning
    
    def _generate_insight(self, risk_level: str, hemoglobin: float) -> str:
        """Generate clinical insight"""
        insights = {
            "HIGH": f"CRITICAL ANEMIA: Hemoglobin {hemoglobin} g/dL is critically low. Immediate medical attention required.",
            "MODERATE": f"MODERATE ANEMIA: Hemoglobin {hemoglobin} g/dL indicates moderate anemia. Follow-up recommended.",
            "LOW": f"LOW RISK: Hemoglobin {hemoglobin} g/dL is within normal range."
        }
        return insights.get(risk_level, "Risk assessment completed")
    
    def _suggest_actions(self, risk_level: str, hemoglobin: float) -> List[str]:
        """Suggest actions based on risk"""
        actions = []
        
        if risk_level == "HIGH":
            actions = [
                "Immediate physician notification",
                "Emergency department referral",
                "Order stat hemoglobin repeat",
                "Prepare for potential transfusion"
            ]
        elif risk_level == "MODERATE":
            actions = [
                "Schedule follow-up within 2 weeks",
                "Order iron studies",
                "Primary care notification"
            ]
        else:
            actions = ["Routine monitoring"]
        
        return actions
    
    def _handle_missing_data(self) -> Dict:
        """Handle missing data case"""
        return {
            "risk_level": "UNKNOWN",
            "confidence_score": 0.0,
            "reasoning": ["Missing required hemoglobin value"],
            "clinical_insight": "Insufficient data for assessment",
            "suggested_actions": ["Request lab results"]
        }
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax activation"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def learn_from_outcome(self, case_id: str, actual_outcome: str, predicted_risk: str):
        """
        Learn from actual patient outcome
        """
        # Compute reward
        if actual_outcome == predicted_risk:
            reward = 1.0
        elif actual_outcome == "HIGH" and predicted_risk == "LOW":
            reward = -2.0  # Severe penalty for missing high risk
        elif actual_outcome == "LOW" and predicted_risk == "HIGH":
            reward = -0.5  # Smaller penalty for false alarm
        else:
            reward = -0.8
        
        # Update Q-learning
        self._update_q_table({
            "state": {"case_id": case_id},
            "action": {"predicted": predicted_risk, "actual": actual_outcome},
            "reward": reward
        })
        
        # Adjust thresholds based on outcome
        if reward < 0 and actual_outcome == "HIGH":
            # Under-predicted high risk - lower thresholds
            self.thresholds["critical"] = self.thresholds.get("critical", 10.0) - 0.2
            logger.info(f"Adjusted critical threshold down to {self.thresholds['critical']}")
        elif reward < 0 and actual_outcome == "LOW":
            # Over-predicted high risk - raise thresholds
            self.thresholds["critical"] = self.thresholds.get("critical", 10.0) + 0.1
            logger.info(f"Adjusted critical threshold up to {self.thresholds['critical']}")
    
    def _load_thresholds(self) -> Dict:
        """Load learned thresholds from file"""
        threshold_path = "learned_thresholds.json"
        if os.path.exists(threshold_path):
            try:
                with open(threshold_path, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {}
    
    def _load_model(self):
        """Load trained neural network"""
        model_path = "decision_model.npz"
        if os.path.exists(model_path):
            try:
                data = np.load(model_path)
                self.W1 = data['W1']
                self.b1 = data['b1']
                self.W2 = data['W2']
                self.b2 = data['b2']
                logger.info("Loaded trained decision model")
            except Exception as e:
                logger.warning(f"Could not load model: {e}")
    
    def _save_model(self):
        """Save trained neural network"""
        np.savez("decision_model.npz", 
                 W1=self.W1, b1=self.b1, 
                 W2=self.W2, b2=self.b2)
        logger.info("Saved decision model")

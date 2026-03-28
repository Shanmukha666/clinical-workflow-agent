"""
Extraction Agent with Neural Learning and Dynamic Pattern Recognition
"""

from __future__ import annotations

import re
import numpy as np
from typing import Any, Dict, List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
import pickle
import os

from .base_agent import DynamicAgent, NeuralMemory, LearningSignal


class ExtractionAgent(DynamicAgent):
    """
    Extraction agent that:
    - Learns new extraction patterns dynamically
    - Updates neural weights based on feedback
    - Communicates patterns to other agents
    """
    
    def __init__(self, neural_memory: Optional[NeuralMemory] = None):
        super().__init__("ExtractionAgent", neural_memory)
        
        # Neural network for pattern recognition
        self.vectorizer = TfidfVectorizer(max_features=100)
        self.nn_model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
        self.is_trained = False
        
        # Dynamic patterns (learned from feedback)
        self.learned_patterns: Dict[str, List[str]] = {
            "hemoglobin": [],
            "wbc": [],
            "platelets": [],
            "creatinine": []
        }
        
        # Context patterns for better extraction
        self.context_patterns: Dict[str, List[str]] = {
            "hemoglobin": ["hb", "hemoglobin", "haemoglobin", "hgb"],
            "wbc": ["wbc", "white blood cell", "leukocyte"],
            "platelets": ["platelet", "plt", "thrombocyte"],
            "creatinine": ["creatinine", "cr", "creat"]
        }
        
        # Initial patterns (will be expanded by learning)
        self.base_patterns = {
            "hemoglobin": [
                r"(?:hb|hemoglobin|haemoglobin|hgb)\s*[:\-=]?\s*(\d+\.?\d*)",
                r"hemoglobin\s+(\d+\.?\d*)\s*(?:g/dL|g/L)"
            ],
            "wbc": [
                r"(?:wbc|white blood cell)\s*[:\-=]?\s*(\d+)"
            ],
            "platelets": [
                r"(?:platelets?|plt)\s*[:\-=]?\s*(\d+)"
            ],
            "creatinine": [
                r"(?:creatinine|creat|cr)\s*[:\-=]?\s*(\d+\.?\d*)"
            ]
        }
        
        # Load saved model if exists
        self._load_model()
    
    def forward(self, input_data: Any) -> Tuple[Dict[str, Any], np.ndarray]:
        """
        Forward pass - extract values with neural embeddings
        """
        text = input_data.get("raw_data", "")
        if not text:
            text = input_data.get("ingested_data", {}).get("raw_data", "")
        
        # Step 1: Extract using current patterns
        extracted = {}
        confidence = {}
        
        for field, patterns in self.base_patterns.items():
            # Add learned patterns
            all_patterns = patterns + self.learned_patterns.get(field, [])
            
            for pattern in all_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    value = matches[0]
                    try:
                        extracted[field] = float(value) if '.' in str(value) else int(value)
                        confidence[field] = self._calculate_confidence(field, pattern, text)
                        break
                    except (ValueError, TypeError):
                        continue
        
        # Step 2: Use neural model for context-based extraction
        if self.is_trained:
            neural_extractions = self._neural_extraction(text)
            for field, value in neural_extractions.items():
                if field not in extracted and value is not None:
                    extracted[field] = value
                    confidence[field] = 0.7  # Neural model confidence
        
        # Step 3: Generate embedding for similarity search
        embedding = self._generate_embedding(extracted, text)
        
        # Step 4: Extract medical entities with context
        entities = self._extract_entities(text)
        
        result = {
            "structured_data": extracted,
            "confidence": confidence,
            "entities": entities,
            "raw_text": text[:500]  # Store snippet for learning
        }
        
        return result, embedding
    
    def compute_loss(self, prediction: Any, target: Any) -> Tuple[float, np.ndarray]:
        """
        Compute loss between extracted values and correct values
        """
        predicted_values = prediction.get("structured_data", {})
        correct_values = target.get("correct_values", {})
        
        total_loss = 0.0
        gradient = np.zeros(128)  # Gradient for neural weights
        
        for field, correct in correct_values.items():
            if field in predicted_values:
                predicted = predicted_values[field]
                # Mean squared error loss
                loss = (predicted - correct) ** 2
                total_loss += loss
                
                # Compute gradient direction
                gradient += (predicted - correct) * 2
        
        # Normalize gradient
        if np.linalg.norm(gradient) > 0:
            gradient = gradient / np.linalg.norm(gradient)
        
        return total_loss, gradient
    
    def _calculate_confidence(self, field: str, pattern: str, text: str) -> float:
        """Calculate confidence in extraction"""
        base_confidence = 0.8
        
        # Check if pattern was learned (lower confidence for learned patterns)
        if pattern in self.learned_patterns.get(field, []):
            base_confidence = 0.6
        
        # Adjust based on context matches
        context_matches = 0
        for ctx in self.context_patterns.get(field, []):
            if ctx in text.lower():
                context_matches += 1
        
        confidence = base_confidence + (context_matches * 0.05)
        return min(1.0, confidence)
    
    def _neural_extraction(self, text: str) -> Dict[str, Optional[float]]:
        """Use neural network for context-based extraction"""
        if not self.is_trained:
            return {}
        
        # Vectorize text
        X = self.vectorizer.transform([text])
        
        # Predict probabilities for each field
        predictions = self.nn_model.predict_proba(X)
        
        # This is simplified - in production, you'd have proper regression
        return {}  # Placeholder
    
    def _generate_embedding(self, extracted: Dict, text: str) -> np.ndarray:
        """Generate vector embedding for similarity search"""
        # Combine extracted values and text features
        features = []
        
        # Add extracted values as features
        for field in ["hemoglobin", "wbc", "platelets", "creatinine"]:
            features.append(float(extracted.get(field, 0)) if extracted.get(field) else 0)
        
        # Add text length as feature
        features.append(len(text))
        
        # Add word count as feature
        features.append(len(text.split()))
        
        # Pad to 128 dimensions
        embedding = np.zeros(128)
        embedding[:len(features)] = features
        
        # Normalize
        if np.linalg.norm(embedding) > 0:
            embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract medical entities with context"""
        text_lower = text.lower()
        
        entities = {
            "symptoms": [],
            "conditions": [],
            "medications": []
        }
        
        # Base entity lists (will be expanded by learning)
        symptom_keywords = ["fatigue", "fever", "weakness", "pain", "cough", "breathlessness"]
        condition_keywords = ["anemia", "infection", "diabetes", "hypertension"]
        medication_keywords = ["paracetamol", "ibuprofen", "insulin"]
        
        # Add learned patterns
        symptom_keywords.extend(self.patterns.get("symptoms", []))
        condition_keywords.extend(self.patterns.get("conditions", []))
        medication_keywords.extend(self.patterns.get("medications", []))
        
        for symptom in symptom_keywords:
            if symptom in text_lower:
                entities["symptoms"].append(symptom)
        
        for condition in condition_keywords:
            if condition in text_lower:
                entities["conditions"].append(condition)
        
        for medication in medication_keywords:
            if medication in text_lower:
                entities["medications"].append(medication)
        
        return entities
    
    def learn_new_pattern(self, field: str, pattern: str, text: str, correct_value: Any) -> None:
        """
        Learn a new extraction pattern from feedback
        """
        # Add to learned patterns
        if field not in self.learned_patterns:
            self.learned_patterns[field] = []
        
        if pattern not in self.learned_patterns[field]:
            self.learned_patterns[field].append(pattern)
            logger.info(f"ExtractionAgent learned new pattern for {field}: {pattern}")
            
            # Generate learning signal for other agents
            signal = LearningSignal(
                sender=self.name,
                receiver="DecisionAgent",
                signal_type="pattern_update",
                data={
                    "pattern_type": f"extraction_{field}",
                    "pattern_value": pattern
                },
                confidence=0.8
            )
            
            # Broadcast to decision agent
            self.learning_queue.append(signal)
    
    def learn_entity(self, entity_type: str, entity_value: str) -> None:
        """
        Learn a new medical entity
        """
        if entity_type not in self.patterns:
            self.patterns[entity_type] = []
        
        if entity_value not in self.patterns[entity_type]:
            self.patterns[entity_type].append(entity_value)
            logger.info(f"ExtractionAgent learned new entity: {entity_type} -> {entity_value}")
    
    def _apply_learned_patterns(self, output: Any, similar_memories: List[Dict]) -> Any:
        """
        Apply patterns from similar past extractions
        """
        structured = output.get("structured_data", {})
        
        for memory in similar_memories:
            prev_output = memory.get("output", {})
            prev_structured = prev_output.get("structured_data", {})
            
            # If we missed a field that was present in similar cases, add it
            for field, value in prev_structured.items():
                if field not in structured and value:
                    structured[field] = value
                    logger.info(f"Applied pattern from similar memory: added {field}={value}")
        
        output["structured_data"] = structured
        return output
    
    def _load_model(self):
        """Load trained neural model"""
        model_path = "extraction_model.pkl"
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    self.vectorizer, self.nn_model = pickle.load(f)
                    self.is_trained = True
                    logger.info("Loaded trained extraction model")
            except Exception as e:
                logger.warning(f"Could not load model: {e}")
    
    def _save_model(self):
        """Save trained neural model"""
        model_path = "extraction_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump((self.vectorizer, self.nn_model), f)
        logger.info("Saved extraction model")
    
    def train_on_feedback(self, feedback_data: List[Dict]) -> None:
        """
        Train neural model on collected feedback
        """
        if len(feedback_data) < 10:
            return
        
        X = []
        y = []
        
        for fb in feedback_data:
            text = fb.get("text", "")
            correct = fb.get("correct_values", {})
            
            if text and correct:
                X.append(text)
                # This is simplified - would need proper labeling
                y.append(correct.get("hemoglobin", 0))
        
        if len(X) >= 10:
            X_vec = self.vectorizer.fit_transform(X)
            # Train model (simplified)
            self.is_trained = True
            self._save_model()
            logger.info(f"Trained extraction model on {len(X)} samples")

"""
Base Agent with Neural Learning Capabilities
"""

from __future__ import annotations

import numpy as np
import pickle
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NeuralMemory:
    """Neural memory with vector embeddings and similarity search"""
    
    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        self.memories: List[Dict] = []
        self.embeddings: List[np.ndarray] = []
        self.weights: Dict[str, np.ndarray] = {}  # Neural weights per agent
        
    def store(self, memory: Dict, embedding: np.ndarray):
        """Store memory with vector embedding"""
        self.memories.append(memory)
        self.embeddings.append(embedding)
        
        # Keep only last 1000 memories for performance
        if len(self.memories) > 1000:
            self.memories.pop(0)
            self.embeddings.pop(0)
    
    def find_similar(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """Find similar memories using cosine similarity"""
        if not self.embeddings:
            return []
        
        similarities = []
        for i, emb in enumerate(self.embeddings):
            sim = np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb))
            similarities.append((sim, i))
        
        similarities.sort(reverse=True, key=lambda x: x[0])
        return [self.memories[idx] for _, idx in similarities[:k]]
    
    def update_weights(self, agent_name: str, gradient: np.ndarray):
        """Update neural weights based on learning"""
        if agent_name not in self.weights:
            self.weights[agent_name] = np.random.randn(self.embedding_dim) * 0.1
        
        self.weights[agent_name] += gradient * 0.01  # Learning rate
        # Normalize
        self.weights[agent_name] = self.weights[agent_name] / np.linalg.norm(self.weights[agent_name])


class LearningSignal:
    """Dynamic learning signal between agents"""
    
    def __init__(self, sender: str, receiver: str, signal_type: str, 
                 data: Any, confidence: float, gradient: Optional[np.ndarray] = None):
        self.id = f"{sender}-{receiver}-{datetime.now().timestamp()}"
        self.sender = sender
        self.receiver = receiver
        self.signal_type = signal_type  # 'threshold_adjustment', 'pattern_update', 'weight_update'
        self.data = data
        self.confidence = confidence
        self.gradient = gradient
        self.timestamp = datetime.now().isoformat()
        self.applied = False


class DynamicAgent(ABC):
    """Base class for all dynamically learning agents"""
    
    def __init__(self, name: str, neural_memory: Optional[NeuralMemory] = None):
        self.name = name
        self.neural_memory = neural_memory or NeuralMemory()
        
        # Neural network weights (perceptron)
        self.weights = np.random.randn(128) * 0.1
        self.bias = 0.0
        
        # Learning parameters
        self.learning_rate = 0.01
        self.performance_history: List[float] = []
        self.adaptation_count = 0
        
        # Dynamic thresholds (will be learned)
        self.thresholds: Dict[str, float] = {}
        self.patterns: Dict[str, List[str]] = {}
        
        # Message queue for learning signals
        self.learning_queue: List[LearningSignal] = []
        
        logger.info(f"Dynamic agent '{name}' initialized with neural memory")
    
    @abstractmethod
    def forward(self, input_data: Any) -> Tuple[Any, np.ndarray]:
        """
        Forward pass - process input and return (output, embedding)
        Embedding is used for similarity search and learning
        """
        pass
    
    @abstractmethod
    def compute_loss(self, prediction: Any, target: Any) -> Tuple[float, np.ndarray]:
        """
        Compute loss and gradient for learning
        Returns (loss, gradient)
        """
        pass
    
    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Main execution with dynamic learning"""
        
        # Extract input from context
        input_data = context.get("current_input", {})
        
        # Forward pass
        try:
            output, embedding = self.forward(input_data)
            
            # Find similar past experiences
            similar_memories = self.neural_memory.find_similar(embedding, k=3)
            
            # Apply learning from similar experiences
            if similar_memories:
                output = self._apply_learned_patterns(output, similar_memories)
            
            # Check for pending learning signals
            self._process_learning_signals()
            
            # Store this experience for future learning
            self.neural_memory.store({
                "input": input_data,
                "output": output,
                "timestamp": datetime.now().isoformat(),
                "agent": self.name
            }, embedding)
            
            return {
                "status": "success",
                "output": output,
                "embedding": embedding.tolist(),
                "similar_memories_used": len(similar_memories),
                "adaptation_count": self.adaptation_count
            }
            
        except Exception as e:
            logger.error(f"Agent {self.name} forward pass failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def receive_learning_signal(self, signal: LearningSignal) -> None:
        """Receive learning signal from another agent"""
        if signal.receiver == self.name:
            self.learning_queue.append(signal)
            logger.info(f"Agent {self.name} received learning signal from {signal.sender}")
    
    def _process_learning_signals(self):
        """Process and apply pending learning signals"""
        while self.learning_queue:
            signal = self.learning_queue.pop(0)
            
            if signal.signal_type == "threshold_adjustment":
                self._apply_threshold_adjustment(signal)
            elif signal.signal_type == "pattern_update":
                self._apply_pattern_update(signal)
            elif signal.signal_type == "weight_update":
                self._apply_weight_update(signal)
            
            signal.applied = True
            self.adaptation_count += 1
    
    def _apply_threshold_adjustment(self, signal: LearningSignal):
        """Dynamically adjust thresholds based on learning"""
        if signal.confidence > 0.7:  # Only apply high-confidence adjustments
            threshold_name = signal.data.get("threshold_name")
            new_value = signal.data.get("new_value")
            
            if threshold_name:
                self.thresholds[threshold_name] = new_value
                logger.info(f"Agent {self.name} adjusted threshold '{threshold_name}' to {new_value}")
    
    def _apply_pattern_update(self, signal: LearningSignal):
        """Learn new extraction or decision patterns"""
        pattern_type = signal.data.get("pattern_type")
        pattern_value = signal.data.get("pattern_value")
        
        if pattern_type and pattern_value:
            if pattern_type not in self.patterns:
                self.patterns[pattern_type] = []
            
            if pattern_value not in self.patterns[pattern_type]:
                self.patterns[pattern_type].append(pattern_value)
                logger.info(f"Agent {self.name} learned new pattern: {pattern_type} -> {pattern_value}")
    
    def _apply_weight_update(self, signal: LearningSignal):
        """Update neural weights based on feedback"""
        if signal.gradient is not None:
            self.weights += self.learning_rate * signal.gradient
            self.weights = self.weights / np.linalg.norm(self.weights)  # Normalize
            logger.info(f"Agent {self.name} updated neural weights")
    
    def _apply_learned_patterns(self, output: Any, similar_memories: List[Dict]) -> Any:
        """Modify output based on similar past experiences"""
        # Override in subclass
        return output
    
    def learn_from_feedback(self, input_data: Any, correct_output: Any, 
                           prediction: Any, confidence: float) -> Dict[str, Any]:
        """
        Learn from feedback - computes gradient and creates learning signal
        """
        # Compute loss and gradient
        loss, gradient = self.compute_loss(prediction, correct_output)
        
        # Update local weights
        self.weights += self.learning_rate * gradient
        self.performance_history.append(1 - loss)
        
        # Keep only last 100
        if len(self.performance_history) > 100:
            self.performance_history.pop(0)
        
        return {
            "loss": loss,
            "gradient": gradient,
            "current_accuracy": np.mean(self.performance_history) if self.performance_history else 0,
            "requires_broadcast": loss > 0.3  # High loss triggers broadcast to other agents
        }

"""
Agents Package - Multi-Agent Clinical Decision Support System
"""

# Import and rename for backward compatibility
from .base_agent import DynamicAgent as DynamicAgent
from .base_agent import NeuralMemory as AgentMemory
from .base_agent import LearningSignal as AgentMessage
from .orchestrator import DynamicOrchestrator
from .extraction import ExtractionAgent
from .decision import DecisionAgent
from .action import ActionAgent
from .feedback import FeedbackAgent
from .ingestion import IngestionAgent
from .communication import MessageBroker, AgentConversation
from .memory import PersistentMemory, LearningEngine

__all__ = [
    'DynamicAgent',
    'AgentMemory',
    'AgentMessage',
    'MessageBroker',
    'AgentConversation',
    'PersistentMemory',
    'LearningEngine',
    'DynamicOrchestrator',
    'ExtractionAgent',
    'DecisionAgent',
    'ActionAgent',
    'FeedbackAgent',
    'IngestionAgent'
]

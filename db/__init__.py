"""
Database Package Initialization for Multi-Agent Clinical System
"""

from __future__ import annotations

from .database import (
    engine,
    SessionLocal,
    get_db,
    get_db_context,
    init_database,
    check_database_health,
    get_session_stats,
    close_all_sessions,
    QueryHelper,
    DatabaseConfig
)

from .models import (
    Base,
    ReportStatus,
    RiskLevel,
    FeedbackLabel,
    AgentType,
    LearningSignalType,
    Patient,
    Report,
    ExtractionResult,
    Decision,
    ActionLog,
    FeedbackLog,
    AgentState,
    LearningSignal,
    NeuralMemory,
    PatientEmbedding,
    LearningPattern,
    AgentConversation,
    PerformanceMetric
)

# Version
__version__ = "2.0.0"

# Package exports
__all__ = [
    # Database core
    "engine",
    "SessionLocal",
    "get_db",
    "get_db_context",
    "init_database",
    "check_database_health",
    "get_session_stats",
    "close_all_sessions",
    "QueryHelper",
    "DatabaseConfig",
    
    # Base and enums
    "Base",
    "ReportStatus",
    "RiskLevel",
    "FeedbackLabel",
    "AgentType",
    "LearningSignalType",
    
    # Models
    "Patient",
    "Report",
    "ExtractionResult",
    "Decision",
    "ActionLog",
    "FeedbackLog",
    "AgentState",
    "LearningSignal",
    "NeuralMemory",
    "PatientEmbedding",
    "LearningPattern",
    "AgentConversation",
    "PerformanceMetric",
]

# Package metadata
__author__ = "Clinical Multi-Agent System"
__description__ = "Database models and utilities for multi-agent clinical decision support system"
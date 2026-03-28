"""
Production Database Models for Dynamic Multi-Agent Clinical System
Supports neural learning, agent communication, and continuous adaptation
"""

from __future__ import annotations

import enum
import uuid
import json
from datetime import datetime
from typing import List, Optional, Dict, Any
from sqlalchemy import (
    DateTime, Float, ForeignKey, Integer, String, Text, 
    Boolean, JSON, Index, LargeBinary, func, UniqueConstraint
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import JSONB, ARRAY


class Base(DeclarativeBase):
    pass


# ==================== ENUMS ====================

class ReportStatus(str, enum.Enum):
    RECEIVED = "RECEIVED"
    PARSED = "PARSED"
    EXTRACTED = "EXTRACTED"
    DECIDED = "DECIDED"
    ACTIONED = "ACTIONED"
    FAILED = "FAILED"
    COMPLETED = "COMPLETED"


class RiskLevel(str, enum.Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LOW = "LOW"
    UNKNOWN = "UNKNOWN"


class FeedbackLabel(str, enum.Enum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    PARTIAL = "partial"


class AgentType(str, enum.Enum):
    INGESTION = "IngestionAgent"
    EXTRACTION = "ExtractionAgent"
    DECISION = "DecisionAgent"
    ACTION = "ActionAgent"
    FEEDBACK = "FeedbackAgent"
    ORCHESTRATOR = "Orchestrator"


class LearningSignalType(str, enum.Enum):
    THRESHOLD_ADJUSTMENT = "threshold_adjustment"
    PATTERN_UPDATE = "pattern_update"
    WEIGHT_UPDATE = "weight_update"
    MODEL_RETRAIN = "model_retrain"
    FEATURE_UPDATE = "feature_update"


# ==================== CORE TABLES ====================

class Patient(Base):
    __tablename__ = "patients"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    patient_uid: Mapped[str] = mapped_column(
        String(36),
        unique=True,
        nullable=False,
        default=lambda: str(uuid.uuid4()),
        index=True,
    )
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    age: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    gender: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    phone: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    email: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    
    # Medical history for learning context
    medical_history: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    chronic_conditions: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Baseline values for personalized thresholds
    baseline_hemoglobin: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    baseline_wbc: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    reports: Mapped[List["Report"]] = relationship(
        back_populates="patient",
        cascade="all, delete-orphan",
    )
    patient_embeddings: Mapped[List["PatientEmbedding"]] = relationship(
        back_populates="patient",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Patient id={self.id} name={self.full_name!r}>"


class Report(Base):
    __tablename__ = "reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    report_uid: Mapped[str] = mapped_column(
        String(36),
        unique=True,
        nullable=False,
        default=lambda: str(uuid.uuid4()),
        index=True,
    )

    patient_id: Mapped[int] = mapped_column(
        ForeignKey("patients.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )

    source_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    file_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    file_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    file_size_bytes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    raw_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    parsed_payload: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # OCR/Extraction confidence
    extraction_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    status: Mapped[str] = mapped_column(
        String(20),
        default=ReportStatus.RECEIVED.value,
        nullable=False,
    )
    
    processing_time_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    patient: Mapped["Patient"] = relationship(back_populates="reports")
    decisions: Mapped[List["Decision"]] = relationship(
        back_populates="report",
        cascade="all, delete-orphan",
    )
    extraction_results: Mapped[List["ExtractionResult"]] = relationship(
        back_populates="report",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Report id={self.id} file_name={self.file_name!r} status={self.status!r}>"


class ExtractionResult(Base):
    """Stores structured extraction results with embeddings"""
    __tablename__ = "extraction_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    extraction_uid: Mapped[str] = mapped_column(
        String(36),
        unique=True,
        nullable=False,
        default=lambda: str(uuid.uuid4()),
        index=True,
    )
    
    report_id: Mapped[int] = mapped_column(
        ForeignKey("reports.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    
    # Extracted values
    hemoglobin: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    wbc: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    platelets: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    creatinine: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sodium: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    potassium: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Additional extracted fields
    additional_values: Mapped[Optional[str]] = mapped_column(JSON, nullable=True)
    
    # Extracted entities
    symptoms: Mapped[Optional[str]] = mapped_column(JSON, nullable=True)
    conditions: Mapped[Optional[str]] = mapped_column(JSON, nullable=True)
    medications: Mapped[Optional[str]] = mapped_column(JSON, nullable=True)
    
    # Confidence scores per field
    confidence_scores: Mapped[Optional[str]] = mapped_column(JSON, nullable=True)
    
    # Vector embedding for similarity search
    embedding: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON array
    
    # Demographics extracted
    extracted_age: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    extracted_gender: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    
    report: Mapped["Report"] = relationship(back_populates="extraction_results")
    decisions: Mapped[List["Decision"]] = relationship(
        back_populates="extraction_result",
        cascade="all, delete-orphan",
    )


class Decision(Base):
    __tablename__ = "decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    decision_uid: Mapped[str] = mapped_column(
        String(36),
        unique=True,
        nullable=False,
        default=lambda: str(uuid.uuid4()),
        index=True,
    )

    report_id: Mapped[int] = mapped_column(
        ForeignKey("reports.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    extraction_result_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("extraction_results.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Decision output
    risk_level: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default=RiskLevel.UNKNOWN.value,
    )
    clinical_insight: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    reasoning: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    confidence_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Neural network predictions
    neural_risk: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    neural_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    threshold_risk: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    
    # Feature vector for learning
    feature_vector: Mapped[Optional[str]] = mapped_column(JSON, nullable=True)
    
    # Suggested actions
    suggested_actions: Mapped[Optional[str]] = mapped_column(JSON, nullable=True)
    
    # Agent metadata
    agent_status: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    processing_time_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    report: Mapped["Report"] = relationship(back_populates="decisions")
    extraction_result: Mapped[Optional["ExtractionResult"]] = relationship(back_populates="decisions")
    feedback_logs: Mapped[List["FeedbackLog"]] = relationship(
        back_populates="decision",
        cascade="all, delete-orphan",
    )
    actions: Mapped[List["ActionLog"]] = relationship(
        back_populates="decision",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Decision id={self.id} risk_level={self.risk_level!r}>"


class ActionLog(Base):
    """Tracks all actions taken by the system"""
    __tablename__ = "action_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    action_uid: Mapped[str] = mapped_column(
        String(36),
        unique=True,
        nullable=False,
        default=lambda: str(uuid.uuid4()),
        index=True,
    )
    
    decision_id: Mapped[int] = mapped_column(
        ForeignKey("decisions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    
    action_type: Mapped[str] = mapped_column(String(50), nullable=False)  # email, sms, api, escalation
    action_status: Mapped[str] = mapped_column(String(20), nullable=False)  # success, failed, pending, retrying
    action_data: Mapped[Optional[str]] = mapped_column(JSON, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    decision: Mapped["Decision"] = relationship(back_populates="actions")


class FeedbackLog(Base):
    __tablename__ = "feedback_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    feedback_uid: Mapped[str] = mapped_column(
        String(36),
        unique=True,
        nullable=False,
        default=lambda: str(uuid.uuid4()),
        index=True,
    )

    decision_id: Mapped[int] = mapped_column(
        ForeignKey("decisions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    doctor_feedback: Mapped[str] = mapped_column(String(20), nullable=False)
    correct_label: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Corrected values for learning
    corrected_hemoglobin: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    corrected_risk: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    corrected_values: Mapped[Optional[str]] = mapped_column(JSON, nullable=True)
    
    # Learning metrics
    loss_computed: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    gradient_norm: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Processing
    processed_for_learning: Mapped[bool] = mapped_column(Boolean, default=False)
    processed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    decision: Mapped["Decision"] = relationship(back_populates="feedback_logs")
    learning_signals: Mapped[List["LearningSignal"]] = relationship(
        back_populates="feedback",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<FeedbackLog id={self.id} doctor_feedback={self.doctor_feedback!r}>"


# ==================== LEARNING TABLES ====================

class AgentState(Base):
    """Stores dynamic state for each agent (weights, thresholds, patterns)"""
    __tablename__ = "agent_states"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    agent_name: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    
    # Neural network weights (serialized)
    weights: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON or pickle base64
    bias: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Dynamic thresholds
    thresholds: Mapped[Optional[str]] = mapped_column(JSON, nullable=True)
    
    # Learned patterns
    learned_patterns: Mapped[Optional[str]] = mapped_column(JSON, nullable=True)
    
    # Q-table for reinforcement learning
    q_table: Mapped[Optional[str]] = mapped_column(JSON, nullable=True)
    
    # Performance metrics
    accuracy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    total_predictions: Mapped[int] = mapped_column(Integer, default=0)
    correct_predictions: Mapped[int] = mapped_column(Integer, default=0)
    
    # Version for tracking updates
    version: Mapped[int] = mapped_column(Integer, default=1)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    
    __table_args__ = (
        UniqueConstraint('agent_name', name='uq_agent_name'),
    )


class LearningSignal(Base):
    """Tracks learning signals between agents"""
    __tablename__ = "learning_signals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    signal_uid: Mapped[str] = mapped_column(
        String(36),
        unique=True,
        nullable=False,
        default=lambda: str(uuid.uuid4()),
        index=True,
    )
    
    feedback_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("feedback_logs.id", ondelete="SET NULL"),
        nullable=True,
    )
    
    sender: Mapped[str] = mapped_column(String(50), nullable=False)
    receiver: Mapped[str] = mapped_column(String(50), nullable=False)
    signal_type: Mapped[str] = mapped_column(String(50), nullable=False)
    
    # Signal data
    data: Mapped[Optional[str]] = mapped_column(JSON, nullable=True)
    gradient: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # Serialized gradient
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    
    # Status
    applied: Mapped[bool] = mapped_column(Boolean, default=False)
    applied_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    
    feedback: Mapped[Optional["FeedbackLog"]] = relationship(back_populates="learning_signals")
    
    __table_args__ = (
        Index('idx_signal_sender_receiver', 'sender', 'receiver'),
        Index('idx_signal_unapplied', 'applied', 'created_at'),
    )


class NeuralMemory(Base):
    """Stores vector embeddings for similarity search"""
    __tablename__ = "neural_memories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    memory_uid: Mapped[str] = mapped_column(
        String(36),
        unique=True,
        nullable=False,
        default=lambda: str(uuid.uuid4()),
        index=True,
    )
    
    # Reference to case
    case_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True, index=True)
    decision_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("decisions.id", ondelete="SET NULL"),
        nullable=True,
    )
    
    # Vector embedding (128-dim)
    embedding: Mapped[str] = mapped_column(Text, nullable=False)  # JSON array
    
    # Context data
    input_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    context_data: Mapped[Optional[str]] = mapped_column(JSON, nullable=True)
    
    # For similarity search
    case_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True, index=True)
    outcome: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    
    # Usage tracking
    access_count: Mapped[int] = mapped_column(Integer, default=0)
    last_accessed: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    
    __table_args__ = (
        Index('idx_memory_case_type', 'case_type', 'outcome'),
        Index('idx_memory_similarity', 'case_type', 'created_at'),
    )


class PatientEmbedding(Base):
    """Patient-specific embeddings for personalized learning"""
    __tablename__ = "patient_embeddings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    
    patient_id: Mapped[int] = mapped_column(
        ForeignKey("patients.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    
    # Embedding for this patient's baseline
    embedding: Mapped[str] = mapped_column(Text, nullable=False)  # JSON array
    
    # Patient-specific thresholds
    personalized_thresholds: Mapped[Optional[str]] = mapped_column(JSON, nullable=True)
    
    # Statistics
    total_cases: Mapped[int] = mapped_column(Integer, default=0)
    avg_hemoglobin: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    hemoglobin_trend: Mapped[Optional[str]] = mapped_column(JSON, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    
    patient: Mapped["Patient"] = relationship(back_populates="patient_embeddings")


class LearningPattern(Base):
    """Detected learning patterns from feedback"""
    __tablename__ = "learning_patterns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    pattern_uid: Mapped[str] = mapped_column(
        String(36),
        unique=True,
        nullable=False,
        default=lambda: str(uuid.uuid4()),
        index=True,
    )
    
    pattern_type: Mapped[str] = mapped_column(String(50), nullable=False)  # threshold_adjustment, pattern_discovery
    description: Mapped[str] = mapped_column(Text, nullable=False)
    
    # Pattern data
    pattern_data: Mapped[Optional[str]] = mapped_column(JSON, nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    
    # Evidence
    evidence_count: Mapped[int] = mapped_column(Integer, default=1)
    supporting_feedback_ids: Mapped[Optional[str]] = mapped_column(JSON, nullable=True)
    
    # Application status
    applied: Mapped[bool] = mapped_column(Boolean, default=False)
    applied_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    effectiveness: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    
    __table_args__ = (
        Index('idx_pattern_applied', 'applied', 'confidence'),
    )


class AgentConversation(Base):
    """Tracks conversations and messages between agents"""
    __tablename__ = "agent_conversations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_uid: Mapped[str] = mapped_column(
        String(36),
        unique=True,
        nullable=False,
        default=lambda: str(uuid.uuid4()),
        index=True,
    )
    
    case_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True, index=True)
    
    # Conversation metadata
    initiator: Mapped[str] = mapped_column(String(50), nullable=False)
    participants: Mapped[Optional[str]] = mapped_column(JSON, nullable=True)
    topic: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    
    # Messages
    messages: Mapped[Optional[str]] = mapped_column(JSON, nullable=True)
    message_count: Mapped[int] = mapped_column(Integer, default=0)
    
    # Outcome
    status: Mapped[str] = mapped_column(String(20), default="active")
    outcome: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)


class PerformanceMetric(Base):
    """Tracks system performance over time"""
    __tablename__ = "performance_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    
    metric_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    metric_value: Mapped[float] = mapped_column(Float, nullable=False)
    
    # Context
    agent_name: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    case_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    
    # Additional data
    meta_data: Mapped[Optional[str]] = mapped_column(JSON, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )
    
    __table_args__ = (
        Index('idx_metric_time_type', 'metric_type', 'created_at'),
        Index('idx_metric_agent', 'agent_name', 'created_at'),
    )


# ==================== HELPER FUNCTIONS ====================

def serialize_embedding(embedding: List[float]) -> str:
    """Convert numpy array to JSON string for storage"""
    return json.dumps(embedding)


def deserialize_embedding(embedding_str: str) -> List[float]:
    """Convert JSON string back to list"""
    return json.loads(embedding_str)


def serialize_weights(weights: Any) -> str:
    """Serialize neural network weights for storage"""
    import base64
    import pickle
    return base64.b64encode(pickle.dumps(weights)).decode('utf-8')


def deserialize_weights(weights_str: str) -> Any:
    """Deserialize neural network weights from storage"""
    import base64
    import pickle
    return pickle.loads(base64.b64decode(weights_str.encode('utf-8')))


# ==================== DATABASE INITIALIZATION ====================

def init_database(engine):
    """Create all tables"""
    Base.metadata.create_all(engine)
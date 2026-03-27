from sqlalchemy import Column, Integer, String, Float, Text, DateTime, ForeignKey, JSON, Boolean, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

Base = declarative_base()


class Patient(Base):
    """
    Stores patient information.
    """
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String(50), unique=True, index=True)
    name = Column(String(255))
    age = Column(Integer)
    gender = Column(String(10))
    medical_history = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    reports = relationship("Report", back_populates="patient")
    decisions = relationship("Decision", back_populates="patient")


class Report(Base):
    """
    Stores uploaded reports and extracted data from the Ingestion → Extraction pipeline.
    """
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String(50), ForeignKey("patients.patient_id"), index=True)
    input_type = Column(String(20))  # pdf, image, text
    raw_data = Column(Text)  # Raw extracted text from OCR/parser
    structured_data = Column(JSON)  # Extracted lab values (hemoglobin, wbc, etc.)
    medical_entities = Column(JSON)  # Extracted symptoms, conditions, medications
    file_path = Column(String(500), nullable=True)  # Path to uploaded file
    processed_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    patient = relationship("Patient", back_populates="reports")
    decisions = relationship("Decision", back_populates="report")


class Decision(Base):
    """
    Stores clinical decisions made by the Decision Engine agent.
    """
    __tablename__ = "decisions"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String(50), ForeignKey("patients.patient_id"), index=True)
    report_id = Column(Integer, ForeignKey("reports.id"), index=True)
    risk_level = Column(String(20))  # LOW, MEDIUM, HIGH, CRITICAL
    reasoning = Column(JSON)  # List of reasoning steps
    clinical_insight = Column(Text)  # Human-readable clinical insight
    confidence_score = Column(Float)  # 0.0 to 1.0
    recommended_actions = Column(JSON)  # List of recommended actions
    is_reviewed = Column(Boolean, default=False)
    reviewed_by = Column(String(255), nullable=True)
    decision_timestamp = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    patient = relationship("Patient", back_populates="decisions")
    report = relationship("Report", back_populates="decisions")
    feedback_logs = relationship("FeedbackLog", back_populates="decision")


class FeedbackLog(Base):
    """
    Stores feedback from doctors on decisions for the learning loop.
    """
    __tablename__ = "feedback_logs"

    id = Column(Integer, primary_key=True, index=True)
    decision_id = Column(Integer, ForeignKey("decisions.id"), index=True)
    patient_id = Column(String(50), ForeignKey("patients.patient_id"), index=True)
    feedback_type = Column(String(20))  # correct, incorrect, partial
    feedback_comment = Column(Text, nullable=True)
    doctor_id = Column(String(100), nullable=True)
    confidence_adjustment = Column(Float, nullable=True)  # If feedback differs, adjust confidence
    original_decision = Column(JSON)  # Store original decision for comparison
    corrected_decision = Column(JSON, nullable=True)  # If feedback was incorrect
    is_used_for_training = Column(Boolean, default=False)
    feedback_timestamp = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    decision = relationship("Decision", back_populates="feedback_logs")
    patient = relationship("Patient")


# Database setup helper
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

def init_db(database_url="postgresql://user:password@localhost/clinical_db"):
    """
    Initialize the database with all tables.
    
    Example:
        database_url = "postgresql://user:password@localhost/clinical_db"
        engine = create_engine(database_url)
        SessionLocal = sessionmaker(bind=engine)
        init_db(database_url)
    """
    engine = create_engine(database_url)
    Base.metadata.create_all(bind=engine)
    return engine


def get_session(database_url="postgresql://user:password@localhost/clinical_db"):
    """
    Get a database session.
    """
    engine = create_engine(database_url)
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()

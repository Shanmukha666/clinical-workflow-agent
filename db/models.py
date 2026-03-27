from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class ReportStatus(str, enum.Enum):
    RECEIVED = "RECEIVED"
    PARSED = "PARSED"
    PROCESSED = "PROCESSED"
    FAILED = "FAILED"


class RiskLevel(str, enum.Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    UNKNOWN = "UNKNOWN"


class FeedbackLabel(str, enum.Enum):
    CORRECT = "correct"
    INCORRECT = "incorrect"


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
        nullable=False,
        index=True,
    )

    source_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    file_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    raw_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    parsed_payload: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(
        String(20),
        default=ReportStatus.RECEIVED.value,
        nullable=False,
    )

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

    def __repr__(self) -> str:
        return f"<Report id={self.id} file_name={self.file_name!r} status={self.status!r}>"


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

    risk_level: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default=RiskLevel.UNKNOWN.value,
    )
    clinical_insight: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    reasoning: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    confidence_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    agent_status: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    report: Mapped["Report"] = relationship(back_populates="decisions")
    feedback_logs: Mapped[List["FeedbackLog"]] = relationship(
        back_populates="decision",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Decision id={self.id} risk_level={self.risk_level!r}>"


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
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    decision: Mapped["Decision"] = relationship(back_populates="feedback_logs")

    def __repr__(self) -> str:
        return f"<FeedbackLog id={self.id} doctor_feedback={self.doctor_feedback!r}>"
"""
Production-Ready API Routes for Dynamic Multi-Agent Clinical System
FastAPI with async support, comprehensive error handling, and monitoring
"""

from __future__ import annotations

import os
import uuid
import logging
import asyncio
import tempfile
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path
import json

from fastapi import (
    FastAPI, 
    File, 
    UploadFile, 
    HTTPException, 
    BackgroundTasks,
    Depends,
    Query,
    status,
    Request
)
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, validator
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
import redis.asyncio as redis
from prometheus_fastapi_instrumentator import Instrumentator

# Local imports
from db.models import (
    Report, Patient, Decision, FeedbackLog, 
    ExtractionResult, ActionLog, LearningSignal, PerformanceMetric
)
from db.database import get_db, engine, init_database
from backend.agents.orchestrator import DynamicOrchestrator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== Configuration ====================

class Settings:
    """Application settings"""
    PROJECT_NAME: str = "Clinical Multi-Agent System"
    VERSION: str = "2.0.0"
    API_V1_PREFIX: str = "/api/v1"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # Security
    API_KEY: str = os.getenv("API_KEY", "default-dev-key-change-in-production")
    ALLOWED_ORIGINS: List[str] = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000").split(",")
    ALLOWED_HOSTS: List[str] = os.getenv("ALLOWED_HOSTS", "*").split(",")
    
    # File upload
    MAX_UPLOAD_SIZE: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS: set = {".txt", ".pdf", ".png", ".jpg", ".jpeg", ".json"}
    TEMP_DIR: Path = Path("/tmp/clinical_uploads")
    
    # Redis for caching
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # Processing limits
    MAX_CONCURRENT_CASES: int = 10
    REQUEST_TIMEOUT: int = 300  # 5 minutes
    
    # Rate limiting
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 60  # seconds

settings = Settings()

# ==================== Redis Client ====================
redis_client: Optional[redis.Redis] = None

# ==================== Pydantic Models ====================
# ... rest of your code ...
class PatientCreate(BaseModel):
    """Patient creation request"""
    full_name: str = Field(..., min_length=1, max_length=255)
    age: Optional[int] = Field(None, ge=0, le=150)
    gender: Optional[str] = Field(None, pattern="^(male|female|other)$")
    phone: Optional[str] = Field(None, pattern=r"^[0-9+\-\s]{10,20}$")
    email: Optional[str] = Field(None, pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
    medical_history: Optional[str] = None
    baseline_hemoglobin: Optional[float] = Field(None, ge=0, le=30)
    
    class Config:
        json_schema_extra = {
            "example": {
                "full_name": "John Doe",
                "age": 45,
                "gender": "male",
                "phone": "+1234567890",
                "email": "john@example.com"
            }
        }


class ProcessRequest(BaseModel):
    """Process request model"""
    case_id: Optional[str] = None
    patient_id: Optional[str] = None
    text: Optional[str] = None
    file_path: Optional[str] = None
    priority: str = Field(default="normal", pattern="^(low|normal|high|critical)$")
    
    @validator('text')
    def validate_text_or_file(cls, v, values):
        if not v and not values.get('file_path'):
            raise ValueError('Either text or file_path must be provided')
        return v


class FeedbackRequest(BaseModel):
    """Feedback request model"""
    decision_id: str = Field(..., min_length=1)
    doctor_feedback: str = Field(..., pattern="^(correct|incorrect|partial)$")
    correct_label: Optional[str] = Field(None, pattern="^(LOW|MODERATE|HIGH|CRITICAL)$")
    correct_values: Optional[Dict[str, float]] = None
    notes: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "decision_id": "abc-123",
                "doctor_feedback": "incorrect",
                "correct_label": "MODERATE",
                "correct_values": {"hemoglobin": 8.5},
                "notes": "Patient has moderate anemia"
            }
        }


class ResponseModel(BaseModel):
    """Standard response model"""
    status: str
    message: str
    data: Optional[Any] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    request_id: str


class ErrorResponse(BaseModel):
    """Error response model"""
    status: str = "error"
    error_code: str
    message: str
    details: Optional[Any] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    request_id: str


# ==================== Helper Functions ====================

def generate_request_id() -> str:
    """Generate unique request ID"""
    return f"req-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"


async def rate_limit_check(api_key: str) -> bool:
    """Check rate limit for API key"""
    if not redis_client:
        return True
    
    key = f"ratelimit:{api_key}"
    current = await redis_client.incr(key)
    if current == 1:
        await redis_client.expire(key, settings.RATE_LIMIT_WINDOW)
    
    if current > settings.RATE_LIMIT_REQUESTS:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Max {settings.RATE_LIMIT_REQUESTS} requests per {settings.RATE_LIMIT_WINDOW} seconds"
        )
    
    return True


async def save_upload_file(upload_file: UploadFile) -> Path:
    """Save uploaded file to temporary directory"""
    # Validate extension
    file_ext = Path(upload_file.filename).suffix.lower()
    if file_ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type {file_ext} not allowed. Allowed: {settings.ALLOWED_EXTENSIONS}"
        )
    
    # Generate unique filename
    unique_filename = f"{uuid.uuid4().hex}{file_ext}"
    file_path = settings.TEMP_DIR / unique_filename
    
    # Save file
    try:
        content = await upload_file.read()
        file_path.write_bytes(content)
        logger.info(f"File saved: {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Failed to save file: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save file: {str(e)}"
        )


def create_response(status: str, message: str, data: Any = None, request_id: str = None) -> ResponseModel:
    """Create standardized response"""
    return ResponseModel(
        status=status,
        message=message,
        data=data,
        request_id=request_id or generate_request_id()
    )


def create_error_response(error_code: str, message: str, details: Any = None, request_id: str = None) -> ErrorResponse:
    """Create standardized error response"""
    return ErrorResponse(
        error_code=error_code,
        message=message,
        details=details,
        request_id=request_id or generate_request_id()
    )


# ==================== Middleware & Dependencies ====================

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.ALLOWED_HOSTS,
)

# API Key security
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Depends(api_key_header)):
    """Verify API key"""
    if not api_key or api_key != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    return api_key

# Redis client
redis_client: Optional[redis.Redis] = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global redis_client
    
    # Create temp directory
    settings.TEMP_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize database
    init_database()
    logger.info("Database initialized")
    
    # Initialize Redis
    try:
        redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
        await redis_client.ping()
        logger.info("Redis connected")
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}")
        redis_client = None
    
    # Initialize orchestrator
    app.state.orchestrator = DynamicOrchestrator()
    
    # Setup metrics
    instrumentator = Instrumentator(
        should_group_status_codes=False,
        should_ignore_untemplated=True,
        should_respect_env_var=True,
        should_instrument_requests_inprogress=True,
        excluded_handlers=["/metrics"],
        env_var_name="ENABLE_METRICS",
    )
    instrumentator.instrument(app).expose(app, endpoint="/metrics")
    
    logger.info("API started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if redis_client:
        await redis_client.close()
    logger.info("API shutdown complete")


# ==================== API Endpoints ====================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "operational",
        "docs": "/api/docs"
    }


@app.get("/health")
async def health_check(db: Session = Depends(get_db)):
    """Health check endpoint"""
    request_id = generate_request_id()
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "database": "unknown",
            "redis": "unknown",
            "orchestrator": "unknown"
        }
    }
    
    # Check database
    try:
        from sqlalchemy import text
        db.execute(text("SELECT 1"))
        health_status["components"]["database"] = "healthy"
    except Exception as e:
        health_status["components"]["database"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check Redis
    if redis_client:
        try:
            await redis_client.ping()
            health_status["components"]["redis"] = "healthy"
        except Exception as e:
            health_status["components"]["redis"] = f"unhealthy: {str(e)}"
            health_status["status"] = "degraded"
    else:
        health_status["components"]["redis"] = "not_configured"
    
    # Check orchestrator
    if hasattr(app.state, 'orchestrator'):
        health_status["components"]["orchestrator"] = "healthy"
    else:
        health_status["components"]["orchestrator"] = "not_initialized"
        health_status["status"] = "degraded"
    
    return health_status


# ==================== 1. Upload Endpoint ====================

@app.post(
    "/upload",
    response_model=ResponseModel,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(verify_api_key)]
)
async def upload_file(
    file: UploadFile = File(...),
    patient_id: Optional[str] = None,
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db),
    request: Request = None
):
    """
    Upload a clinical document for processing
    
    - **file**: Clinical document (PDF, image, text file)
    - **patient_id**: Optional patient ID to associate with
    """
    request_id = generate_request_id()
    
    # Rate limiting
    api_key = request.headers.get("X-API-Key")
    await rate_limit_check(api_key)
    
    try:
        # Save uploaded file
        file_path = await save_upload_file(file)
        
        # Create or retrieve patient
        patient = None
        if patient_id:
            patient = db.query(Patient).filter(Patient.patient_uid == patient_id).first()
            if not patient:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Patient {patient_id} not found"
                )
        
        # Create report record
        report = Report(
            patient_id=patient.id if patient else None,
            source_type=file.content_type,
            file_name=file.filename,
            file_path=str(file_path),
            file_size_bytes=file_path.stat().st_size,
            status="RECEIVED"
        )
        db.add(report)
        db.commit()
        db.refresh(report)
        
        # Trigger processing in background
        background_tasks.add_task(
            process_uploaded_file,
            report.report_uid,
            str(file_path),
            request_id
        )
        
        return create_response(
            status="accepted",
            message="File uploaded successfully. Processing started.",
            data={
                "report_uid": report.report_uid,
                "patient_id": patient.patient_uid if patient else None,
                "file_name": file.filename,
                "status": "processing"
            },
            request_id=request_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {str(e)}"
        )


async def process_uploaded_file(report_uid: str, file_path: str, request_id: str):
    """Background task to process uploaded file"""
    db = next(get_db())
    
    try:
        # Update status
        report = db.query(Report).filter(Report.report_uid == report_uid).first()
        if report:
            report.status = "PARSED"
            db.commit()
        
        # Process with orchestrator
        orchestrator = app.state.orchestrator
        result = orchestrator.process_case({
            "case_id": report_uid,
            "file": file_path,
            "report_id": report.id if report else None
        })
        
        # Update with results
        if report:
            if result["status"] == "success":
                report.status = "COMPLETED"
                
                # Save extraction results
                if "extraction" in result:
                    extraction = ExtractionResult(
                        report_id=report.id,
                        hemoglobin=result["extraction"].get("structured_data", {}).get("hemoglobin"),
                        confidence_scores=json.dumps(result["extraction"].get("confidence", {})),
                        symptoms=json.dumps(result["extraction"].get("entities", {}).get("symptoms", [])),
                        conditions=json.dumps(result["extraction"].get("entities", {}).get("conditions", [])),
                        medications=json.dumps(result["extraction"].get("entities", {}).get("medications", []))
                    )
                    db.add(extraction)
                    db.flush()
                    
                    # Save decision
                    if "decision" in result:
                        decision = Decision(
                            report_id=report.id,
                            extraction_result_id=extraction.id,
                            risk_level=result["decision"].get("risk_level"),
                            clinical_insight=result["decision"].get("clinical_insight"),
                            reasoning=json.dumps(result["decision"].get("reasoning", [])),
                            confidence_score=result["decision"].get("confidence_score"),
                            neural_risk=result["decision"].get("neural_risk"),
                            neural_confidence=result["decision"].get("neural_confidence"),
                            threshold_risk=result["decision"].get("threshold_risk"),
                            suggested_actions=json.dumps(result["decision"].get("suggested_actions", []))
                        )
                        db.add(decision)
                        
                        # Save actions if any
                        if "action" in result and result["action"]:
                            for action in result["action"].get("actions_taken", []):
                                action_log = ActionLog(
                                    decision_id=decision.id,
                                    action_type=action.split(" ")[0].lower(),
                                    action_status="success",
                                    action_data=json.dumps({"description": action})
                                )
                                db.add(action_log)
                
                db.commit()
                
            else:
                report.status = "FAILED"
                report.error_message = result.get("error", "Unknown error")
                db.commit()
        
        # Clean up temp file
        try:
            Path(file_path).unlink()
        except Exception as e:
            logger.warning(f"Failed to delete temp file: {e}")
            
    except Exception as e:
        logger.error(f"Background processing failed: {e}", exc_info=True)
        if report:
            report.status = "FAILED"
            report.error_message = str(e)
            db.commit()


# ==================== 2. Process Endpoint ====================

@app.post(
    "/process",
    response_model=ResponseModel,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(verify_api_key)]
)
async def process_text(
    request: ProcessRequest,
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db),
    req: Request = None
):
    """
    Process clinical text directly
    
    - **case_id**: Optional case ID
    - **patient_id**: Optional patient ID
    - **text**: Clinical text to process
    - **file_path**: Optional file path (if already uploaded)
    - **priority**: Processing priority (low, normal, high, critical)
    """
    request_id = generate_request_id()
    
    # Rate limiting
    api_key = req.headers.get("X-API-Key")
    await rate_limit_check(api_key)
    
    try:
        # Create or retrieve patient
        patient = None
        if request.patient_id:
            patient = db.query(Patient).filter(Patient.patient_uid == request.patient_id).first()
            if not patient:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Patient {request.patient_id} not found"
                )
        
        # Create report record
        report = Report(
            patient_id=patient.id if patient else None,
            source_type="text" if request.text else "file",
            status="RECEIVED",
            raw_text=request.text if request.text else None
        )
        db.add(report)
        db.commit()
        db.refresh(report)
        
        # Process with orchestrator
        orchestrator = app.state.orchestrator
        
        input_data = {
            "case_id": request.case_id or report.report_uid,
            "patient_id": request.patient_id,
            "raw_data": request.text,
            "report_id": report.id,
            "priority": request.priority
        }
        
        if request.file_path:
            input_data["file"] = request.file_path
        
        result = orchestrator.process_case(input_data)
        
        if result["status"] == "success":
            report.status = "COMPLETED"
            
            # Save extraction results
            if "extraction" in result:
                extraction = ExtractionResult(
                    report_id=report.id,
                    hemoglobin=result["extraction"].get("structured_data", {}).get("hemoglobin"),
                    wbc=result["extraction"].get("structured_data", {}).get("wbc"),
                    platelets=result["extraction"].get("structured_data", {}).get("platelets"),
                    creatinine=result["extraction"].get("structured_data", {}).get("creatinine"),
                    confidence_scores=json.dumps(result["extraction"].get("confidence", {})),
                    symptoms=json.dumps(result["extraction"].get("entities", {}).get("symptoms", [])),
                    conditions=json.dumps(result["extraction"].get("entities", {}).get("conditions", [])),
                    medications=json.dumps(result["extraction"].get("entities", {}).get("medications", []))
                )
                db.add(extraction)
                db.flush()
                
                # Save decision
                if "decision" in result:
                    decision = Decision(
                        report_id=report.id,
                        extraction_result_id=extraction.id,
                        risk_level=result["decision"].get("risk_level"),
                        clinical_insight=result["decision"].get("clinical_insight"),
                        reasoning=json.dumps(result["decision"].get("reasoning", [])),
                        confidence_score=result["decision"].get("confidence_score"),
                        neural_risk=result["decision"].get("neural_risk"),
                        neural_confidence=result["decision"].get("neural_confidence"),
                        threshold_risk=result["decision"].get("threshold_risk"),
                        feature_vector=json.dumps(result["decision"].get("features", [])),
                        suggested_actions=json.dumps(result["decision"].get("suggested_actions", []))
                    )
                    db.add(decision)
                    db.flush()
                    
                    # Save actions
                    if "action" in result and result["action"]:
                        for action in result["action"].get("actions_taken", []):
                            action_log = ActionLog(
                                decision_id=decision.id,
                                action_type=action.split(" ")[0].lower(),
                                action_status="success",
                                action_data=json.dumps({"description": action})
                            )
                            db.add(action_log)
            
            db.commit()
            
            return create_response(
                status="success",
                message="Text processed successfully",
                data={
                    "report_uid": report.report_uid,
                    "patient_id": patient.patient_uid if patient else None,
                    "risk_level": result["decision"].get("risk_level"),
                    "confidence_score": result["decision"].get("confidence_score"),
                    "clinical_insight": result["decision"].get("clinical_insight"),
                    "reasoning": result["decision"].get("reasoning", []),
                    "actions": result.get("action", {}).get("actions_taken", []),
                    "processing_time_ms": result.get("processing_time_ms", 0)
                },
                request_id=request_id
            )
        else:
            report.status = "FAILED"
            report.error_message = result.get("error", "Processing failed")
            db.commit()
            
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=result.get("error", "Processing failed")
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing failed: {str(e)}"
        )


# ==================== Additional Endpoints ====================

@app.get(
    "/cases/{case_id}",
    response_model=ResponseModel,
    dependencies=[Depends(verify_api_key)]
)
async def get_case_status(
    case_id: str,
    db: Session = Depends(get_db),
    req: Request = None
):
    """Get status and results of a processed case"""
    request_id = generate_request_id()
    
    try:
        # Find report
        report = db.query(Report).filter(
            (Report.report_uid == case_id) | (Report.id == case_id if case_id.isdigit() else False)
        ).first()
        
        if not report:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Case {case_id} not found"
            )
        
        # Build response
        response_data = {
            "case_id": report.report_uid,
            "status": report.status,
            "created_at": report.created_at.isoformat(),
            "updated_at": report.updated_at.isoformat(),
            "file_name": report.file_name,
        }
        
        # Add extraction if exists
        if report.extraction_results:
            extraction = report.extraction_results[-1]
            response_data["extraction"] = {
                "hemoglobin": extraction.hemoglobin,
                "wbc": extraction.wbc,
                "platelets": extraction.platelets,
                "creatinine": extraction.creatinine,
                "confidence": json.loads(extraction.confidence_scores) if extraction.confidence_scores else {},
                "entities": {
                    "symptoms": json.loads(extraction.symptoms) if extraction.symptoms else [],
                    "conditions": json.loads(extraction.conditions) if extraction.conditions else [],
                    "medications": json.loads(extraction.medications) if extraction.medications else []
                }
            }
        
        # Add decision if exists
        if report.decisions:
            decision = report.decisions[-1]
            response_data["decision"] = {
                "risk_level": decision.risk_level,
                "confidence_score": decision.confidence_score,
                "clinical_insight": decision.clinical_insight,
                "reasoning": json.loads(decision.reasoning) if decision.reasoning else [],
                "suggested_actions": json.loads(decision.suggested_actions) if decision.suggested_actions else []
            }
            
            # Add feedback if exists
            if decision.feedback_logs:
                response_data["feedback"] = {
                    "has_feedback": True,
                    "feedback_count": len(decision.feedback_logs)
                }
        
        return create_response(
            status="success",
            message="Case retrieved successfully",
            data=response_data,
            request_id=request_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get case: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get case: {str(e)}"
        )


@app.post(
    "/feedback",
    response_model=ResponseModel,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(verify_api_key)]
)
async def submit_feedback(
    feedback: FeedbackRequest,
    db: Session = Depends(get_db),
    req: Request = None
):
    """Submit doctor feedback for learning"""
    request_id = generate_request_id()
    
    try:
        # Find decision
        decision = db.query(Decision).filter(Decision.decision_uid == feedback.decision_id).first()
        
        if not decision:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Decision {feedback.decision_id} not found"
            )
        
        # Create feedback record
        feedback_log = FeedbackLog(
            decision_id=decision.id,
            doctor_feedback=feedback.doctor_feedback,
            correct_label=feedback.correct_label,
            notes=feedback.notes,
            corrected_values=json.dumps(feedback.correct_values) if feedback.correct_values else None,
            processed_for_learning=False
        )
        db.add(feedback_log)
        db.commit()
        db.refresh(feedback_log)
        
        # Trigger orchestrator learning
        orchestrator = app.state.orchestrator
        
        # Convert to format expected by orchestrator
        learning_data = {
            "decision_id": decision.decision_uid,
            "doctor_feedback": feedback.doctor_feedback,
            "correct_label": feedback.correct_label,
            "correct_values": feedback.correct_values,
            "notes": feedback.notes
        }
        
        # Process feedback for learning
        learning_result = orchestrator.provide_feedback(decision.decision_uid, learning_data)
        
        # Update feedback record with learning status
        feedback_log.processed_for_learning = True
        feedback_log.processed_at = datetime.now()
        db.commit()
        
        return create_response(
            status="success",
            message="Feedback submitted successfully",
            data={
                "feedback_id": feedback_log.feedback_uid,
                "decision_id": decision.decision_uid,
                "learning_triggered": True,
                "learning_result": learning_result
            },
            request_id=request_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to submit feedback: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit feedback: {str(e)}"
        )


@app.get(
    "/metrics/system",
    response_model=ResponseModel,
    dependencies=[Depends(verify_api_key)]
)
async def get_system_metrics(
    db: Session = Depends(get_db),
    req: Request = None
):
    """Get system performance metrics"""
    request_id = generate_request_id()
    
    try:
        orchestrator = app.state.orchestrator
        agent_metrics = orchestrator.get_metrics()
        
        # Get database metrics
        total_patients = db.query(Patient).count()
        total_reports = db.query(Report).count()
        total_decisions = db.query(Decision).count()
        total_feedback = db.query(FeedbackLog).count()
        
        # Get accuracy from recent feedback
        recent_feedback = db.query(FeedbackLog).order_by(
            FeedbackLog.created_at.desc()
        ).limit(100).all()
        
        correct_feedback = sum(1 for f in recent_feedback if f.doctor_feedback == "correct")
        accuracy = correct_feedback / len(recent_feedback) if recent_feedback else 0
        
        return create_response(
            status="success",
            message="System metrics retrieved",
            data={
                "agents": agent_metrics,
                "database": {
                    "patients": total_patients,
                    "reports": total_reports,
                    "decisions": total_decisions,
                    "feedback": total_feedback
                },
                "performance": {
                    "recent_accuracy": accuracy,
                    "total_feedback_samples": len(recent_feedback)
                }
            },
            request_id=request_id
        )
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get metrics: {str(e)}"
        )


@app.get(
    "/patients/{patient_id}/history",
    response_model=ResponseModel,
    dependencies=[Depends(verify_api_key)]
)
async def get_patient_history(
    patient_id: str,
    limit: int = Query(default=10, ge=1, le=100),
    db: Session = Depends(get_db),
    req: Request = None
):
    """Get patient medical history"""
    request_id = generate_request_id()
    
    try:
        # Find patient
        patient = db.query(Patient).filter(Patient.patient_uid == patient_id).first()
        
        if not patient:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Patient {patient_id} not found"
            )
        
        # Get reports with decisions
        reports = db.query(Report).filter(
            Report.patient_id == patient.id
        ).order_by(
            Report.created_at.desc()
        ).limit(limit).all()
        
        history = []
        for report in reports:
            report_data = {
                "report_uid": report.report_uid,
                "date": report.created_at.isoformat(),
                "status": report.status
            }
            
            if report.decisions:
                decision = report.decisions[-1]
                report_data["risk_level"] = decision.risk_level
                report_data["confidence"] = decision.confidence_score
                report_data["clinical_insight"] = decision.clinical_insight
            
            history.append(report_data)
        
        return create_response(
            status="success",
            message="Patient history retrieved",
            data={
                "patient_id": patient.patient_uid,
                "patient_name": patient.full_name,
                "age": patient.age,
                "gender": patient.gender,
                "total_cases": len(history),
                "history": history
            },
            request_id=request_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get patient history: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get patient history: {str(e)}"
        )


# ==================== Exception Handlers ====================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    request_id = generate_request_id()
    return JSONResponse(
        status_code=exc.status_code,
        content=create_error_response(
            error_code=f"HTTP_{exc.status_code}",
            message=exc.detail,
            request_id=request_id
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    request_id = generate_request_id()
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=create_error_response(
            error_code="INTERNAL_ERROR",
            message="An unexpected error occurred",
            details=str(exc) if settings.DEBUG else None,
            request_id=request_id
        ).dict()
    )
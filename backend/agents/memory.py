"""
Shared Memory and Learning System
Persists knowledge across sessions and enables continuous learning
"""

from __future__ import annotations

import json
import pickle
import sqlite3
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from collections import defaultdict
import hashlib
import logging
logger = logging.getLogger(__name__)


class PersistentMemory:
    """
    Persistent memory with SQLite backend
    Stores cases, feedback, and learned patterns
    """
    
    def __init__(self, db_path: str = "agent_memory.db"):
        self.db_path = db_path
        self._init_database()
        self.cache: Dict[str, Any] = {}
    
    def _init_database(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Cases table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cases (
                case_id TEXT PRIMARY KEY,
                patient_id TEXT,
                timestamp TEXT,
                raw_data TEXT,
                structured_data TEXT,
                decision TEXT,
                actions TEXT,
                status TEXT,
                processing_time REAL
            )
        """)
        
        # Feedback table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                decision_id TEXT,
                doctor_feedback TEXT,
                correct_label TEXT,
                notes TEXT,
                original_risk TEXT,
                hemoglobin_value REAL,
                timestamp TEXT,
                processed BOOLEAN DEFAULT 0
            )
        """)
        
        # Learning patterns table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS learning_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT,
                description TEXT,
                confidence REAL,
                applied BOOLEAN DEFAULT 0,
                effectiveness REAL,
                created_at TEXT,
                last_used TEXT
            )
        """)
        
        # Agent conversations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                agents TEXT,
                messages TEXT,
                outcome TEXT,
                timestamp TEXT
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info(f"Database initialized at {self.db_path}")
    
    def store_case(self, case_data: Dict):
        """Store a processed case"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO cases 
            (case_id, patient_id, timestamp, raw_data, structured_data, decision, actions, status, processing_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            case_data.get("case_id"),
            case_data.get("patient_id"),
            case_data.get("timestamp"),
            json.dumps(case_data.get("raw_data", "")),
            json.dumps(case_data.get("structured_data", {})),
            json.dumps(case_data.get("decision", {})),
            json.dumps(case_data.get("actions", [])),
            case_data.get("status"),
            case_data.get("processing_time", 0)
        ))
        
        conn.commit()
        conn.close()
    
    def store_feedback(self, feedback: Dict):
        """Store doctor feedback"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO feedback 
            (decision_id, doctor_feedback, correct_label, notes, original_risk, hemoglobin_value, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            feedback.get("decision_id"),
            feedback.get("doctor_feedback"),
            feedback.get("correct_label"),
            feedback.get("notes"),
            feedback.get("original_risk"),
            feedback.get("hemoglobin_value"),
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        # Trigger learning from new feedback
        self._process_pending_feedback()
    
    def _process_pending_feedback(self):
        """Process unprocessed feedback for learning"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM feedback WHERE processed = 0
        """)
        
        pending = cursor.fetchall()
        
        for p in pending:
            self._learn_from_feedback({
                "id": p[0],
                "decision_id": p[1],
                "doctor_feedback": p[2],
                "correct_label": p[3],
                "original_risk": p[5],
                "hemoglobin_value": p[6]
            })
            
            # Mark as processed
            cursor.execute("UPDATE feedback SET processed = 1 WHERE id = ?", (p[0],))
        
        conn.commit()
        conn.close()
    
    def _learn_from_feedback(self, feedback: Dict):
        """Extract learning patterns from feedback"""
        if feedback["doctor_feedback"] == "incorrect":
            # Check for pattern in threshold errors
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Find similar cases
            cursor.execute("""
                SELECT COUNT(*) FROM feedback 
                WHERE original_risk = ? AND doctor_feedback = 'incorrect'
            """, (feedback["original_risk"],))
            
            similar_errors = cursor.fetchone()[0]
            
            # If enough patterns, suggest threshold adjustment
            if similar_errors >= 3:
                pattern = {
                    "pattern_type": "threshold_adjustment",
                    "description": f"Multiple incorrect {feedback['original_risk']} risk classifications",
                    "confidence": min(0.9, similar_errors / 10),
                    "created_at": datetime.now().isoformat()
                }
                
                cursor.execute("""
                    INSERT INTO learning_patterns 
                    (pattern_type, description, confidence, created_at)
                    VALUES (?, ?, ?, ?)
                """, (pattern["pattern_type"], pattern["description"], 
                      pattern["confidence"], pattern["created_at"]))
                
                logger.info(f"New learning pattern detected: {pattern['description']}")
            
            conn.commit()
            conn.close()
    
    def get_learning_patterns(self, min_confidence: float = 0.7) -> List[Dict]:
        """Get active learning patterns"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM learning_patterns 
            WHERE confidence >= ? AND applied = 0
            ORDER BY confidence DESC
        """, (min_confidence,))
        
        patterns = cursor.fetchall()
        conn.close()
        
        return [{
            "id": p[0],
            "type": p[1],
            "description": p[2],
            "confidence": p[3],
            "created_at": p[6]
        } for p in patterns]
    
    def get_case_history(self, patient_id: str, days: int = 30) -> List[Dict]:
        """Get case history for a patient"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        
        cursor.execute("""
            SELECT * FROM cases 
            WHERE patient_id = ? AND timestamp >= ?
            ORDER BY timestamp DESC
        """, (patient_id, cutoff))
        
        cases = cursor.fetchall()
        conn.close()
        
        return [{
            "case_id": c[0],
            "timestamp": c[2],
            "structured_data": json.loads(c[4]),
            "decision": json.loads(c[5]),
            "status": c[7]
        } for c in cases]
    
    def get_accuracy_metrics(self, days: int = 30) -> Dict[str, Any]:
        """Get accuracy metrics for the system"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        
        cursor.execute("""
            SELECT doctor_feedback, COUNT(*) FROM feedback 
            WHERE timestamp >= ?
            GROUP BY doctor_feedback
        """, (cutoff,))
        
        feedback_counts = dict(cursor.fetchall())
        
        total = sum(feedback_counts.values())
        correct = feedback_counts.get("correct", 0)
        
        conn.close()
        
        return {
            "accuracy": correct / total if total > 0 else 0,
            "total_feedback": total,
            "period_days": days,
            "correct_count": correct,
            "incorrect_count": feedback_counts.get("incorrect", 0)
        }


class LearningEngine:
    """
    Machine learning based learning engine
    Uses historical data to improve decisions
    """
    
    def __init__(self, memory: PersistentMemory):
        self.memory = memory
        self.threshold_adjustments: Dict[str, float] = {}
        self._load_adjustments()
    
    def _load_adjustments(self):
        """Load learned threshold adjustments"""
        patterns = self.memory.get_learning_patterns()
        
        for pattern in patterns:
            if pattern["type"] == "threshold_adjustment":
                # Extract adjustment value
                if "suggested_threshold" in pattern.get("description", ""):
                    # Parse and apply
                    pass
    
    def suggest_threshold_adjustment(self, risk_level: str, 
                                     incorrect_cases: List[Dict]) -> Optional[float]:
        """Suggest threshold adjustment based on incorrect cases"""
        if len(incorrect_cases) < 3:
            return None
        
        # Calculate average hemoglobin of incorrectly classified cases
        hb_values = []
        for case in incorrect_cases:
            hb = case.get("hemoglobin_value")
            if hb:
                hb_values.append(hb)
        
        if not hb_values:
            return None
        
        return sum(hb_values) / len(hb_values)
    
    def should_adapt(self, agent_name: str) -> bool:
        """Determine if agent should adapt based on performance"""
        metrics = self.memory.get_accuracy_metrics(days=7)
        
        # If accuracy below 80% in last 7 days, suggest adaptation
        return metrics["accuracy"] < 0.8 and metrics["total_feedback"] > 10
    
    def adapt_decision_agent(self, decision_agent) -> bool:
        """Adapt decision agent based on learned patterns"""
        if not self.should_adapt("decision"):
            return False
        
        # Get recent incorrect cases
        conn = sqlite3.connect(self.memory.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM feedback 
            WHERE doctor_feedback = 'incorrect' 
            AND timestamp > datetime('now', '-7 days')
        """)
        
        recent_errors = cursor.fetchall()
        conn.close()
        
        if len(recent_errors) < 3:
            return False
        
        # Suggest threshold adjustment
        hb_values = [e[6] for e in recent_errors if e[6]]
        
        if hb_values:
            suggested_threshold = sum(hb_values) / len(hb_values)
            logger.info(f"LEARNING: Suggesting threshold adjustment to {suggested_threshold}")
            
            # Update decision agent (would be done through configuration)
            return True
        
        return False

from datetime import datetime
from typing import Dict, Optional


class FeedbackAgent:
    def __init__(self, db_client=None):
        """
        db_client: optional database connector (MongoDB / SQLAlchemy etc.)
        """
        self.db = db_client

    # ---------- PUBLIC METHOD ----------
    def process_feedback(self, feedback: Dict) -> Dict:
        """
        Main entry point
        """
        self._validate_feedback(feedback)

        enriched_feedback = self._enrich_feedback(feedback)

        # Store feedback
        self._store_feedback(enriched_feedback)

        # Generate learning signal
        learning_signal = self._generate_learning_signal(enriched_feedback)

        return {
            "status": "SUCCESS",
            "feedback_id": enriched_feedback["feedback_id"],
            "learning_signal": learning_signal
        }

    # ---------- VALIDATION ----------
    def _validate_feedback(self, feedback: Dict):
        required_fields = ["decision_id", "doctor_feedback"]

        for field in required_fields:
            if field not in feedback:
                raise ValueError(f"Missing required field: {field}")

        if feedback["doctor_feedback"] not in ["correct", "incorrect"]:
            raise ValueError("doctor_feedback must be 'correct' or 'incorrect'")

    # ---------- ENRICHMENT ----------
    def _enrich_feedback(self, feedback: Dict) -> Dict:
        return {
            "feedback_id": f"FB-{datetime.utcnow().timestamp()}",
            "decision_id": feedback["decision_id"],
            "doctor_feedback": feedback["doctor_feedback"],
            "correct_label": feedback.get("correct_label"),
            "notes": feedback.get("notes"),
            "timestamp": datetime.utcnow().isoformat()
        }

    # ---------- STORAGE ----------
    def _store_feedback(self, feedback: Dict):
        if self.db:
            # Example for MongoDB
            self.db.feedback.insert_one(feedback)
        else:
            # Fallback: simple print/log (for hackathon)
            print("[FEEDBACK LOGGED]", feedback)

    # ---------- LEARNING SIGNAL ----------
    def _generate_learning_signal(self, feedback: Dict) -> Dict:
        """
        This is what makes your system 'learning-capable'
        """

        if feedback["doctor_feedback"] == "correct":
            return {
                "update_required": False,
                "message": "Decision confirmed"
            }

        return {
            "update_required": True,
            "message": "Incorrect decision - update rules/model",
            "suggested_label": feedback.get("correct_label")
        }
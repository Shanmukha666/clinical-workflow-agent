from __future__ import annotations

from typing import Any, Dict, List


class DecisionAgent:
    """
    Decision agent:
    Converts structured clinical data into explainable clinical insights.
    """

    CRITICAL_HB_THRESHOLD = 7.0
    MODERATE_HB_THRESHOLD = 10.0

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        structured_data = payload.get("structured_data", {})
        hemoglobin = structured_data.get("hemoglobin")
        case_id = payload.get("case_id")

        if hemoglobin is None:
            return self._build_response(
                status="error",
                case_id=case_id,
                decision={
                    "risk_level": "UNKNOWN",
                    "reasoning": ["Missing required field: hemoglobin"],
                    "clinical_insight": "Insufficient data to assess anemia risk",
                    "confidence_score": 0.0,
                },
            )

        try:
            hb = float(hemoglobin)
        except (TypeError, ValueError):
            return self._build_response(
                status="error",
                case_id=case_id,
                decision={
                    "risk_level": "UNKNOWN",
                    "reasoning": ["Hemoglobin must be a numeric value"],
                    "clinical_insight": "Invalid hemoglobin input",
                    "confidence_score": 0.0,
                },
            )

        if hb < 0:
            return self._build_response(
                status="error",
                case_id=case_id,
                decision={
                    "risk_level": "UNKNOWN",
                    "reasoning": ["Hemoglobin cannot be negative"],
                    "clinical_insight": "Invalid hemoglobin input",
                    "confidence_score": 0.0,
                },
            )

        return self._build_response(
            status="success",
            case_id=case_id,
            decision=self._evaluate_hemoglobin(hb),
        )

    def _evaluate_hemoglobin(self, hb: float) -> Dict[str, Any]:
        reasoning: List[str] = []

        if hb < self.CRITICAL_HB_THRESHOLD:
            reasoning.append("Hemoglobin < 7 (critical threshold)")
            return {
                "risk_level": "HIGH",
                "reasoning": reasoning,
                "clinical_insight": "Severe anemia risk",
                "confidence_score": 0.92,
            }

        if self.CRITICAL_HB_THRESHOLD <= hb < self.MODERATE_HB_THRESHOLD:
            reasoning.append("Hemoglobin between 7 and 10 (moderate anemia range)")
            return {
                "risk_level": "MODERATE",
                "reasoning": reasoning,
                "clinical_insight": "Moderate anemia risk",
                "confidence_score": 0.84,
            }

        reasoning.append("Hemoglobin >= 10 (not in severe/moderate anemia threshold)")
        return {
            "risk_level": "LOW",
            "reasoning": reasoning,
            "clinical_insight": "No severe anemia risk detected",
            "confidence_score": 0.95,
        }

    def _build_response(
        self,
        status: str,
        case_id: str | None,
        decision: Dict[str, Any],
    ) -> Dict[str, Any]:
        response = {
            "agent": "decision",
            "status": status,
            "decision": decision,
        }
        if case_id is not None:
            response["case_id"] = case_id
        return response

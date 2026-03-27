from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage
from typing import Any, Dict, List


class ActionAgent:
    """
    Action agent:
    Triggers downstream real-world outcomes based on decision output.

    Supported actions:
    - Mock alert
    - SMS via Twilio (optional)
    - Email via SMTP (optional)
    - Mock hospital API call

    Expected input:
    {
        "agent": "decision",
        "status": "success",
        "case_id": "CASE-001",
        "decision": {
            "risk_level": "HIGH",
            "reasoning": [...],
            "clinical_insight": "...",
            "confidence_score": 0.92
        }
    }
    """

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        case_id = payload.get("case_id")
        status = payload.get("status")
        decision = payload.get("decision", {})
        risk_level = decision.get("risk_level")

        if status != "success":
            return self._build_response(
                status="error",
                case_id=case_id,
                actions=["No action taken because decision step failed"],
            )

        if not risk_level:
            return self._build_response(
                status="error",
                case_id=case_id,
                actions=["No action taken because risk_level is missing"],
            )

        actions_taken: List[str] = []

        if risk_level == "HIGH":
            actions_taken.append("Send alert")
            actions_taken.append("Schedule appointment")

            self._send_sms_if_configured(
                message=f"[{case_id}] HIGH risk detected. Immediate clinical review recommended."
            )
            self._send_email_if_configured(
                subject=f"Clinical Alert for {case_id}",
                body=(
                    f"Case: {case_id}\n"
                    f"Risk Level: HIGH\n"
                    f"Insight: {decision.get('clinical_insight', 'N/A')}\n"
                    f"Reasoning: {', '.join(decision.get('reasoning', []))}\n"
                ),
            )
            self._notify_hospital_api_mock(case_id, "HIGH")

        elif risk_level == "MODERATE":
            actions_taken.append("Send follow-up notification")
            actions_taken.append("Recommend clinical follow-up")

            self._send_email_if_configured(
                subject=f"Follow-up Recommendation for {case_id}",
                body=(
                    f"Case: {case_id}\n"
                    f"Risk Level: MODERATE\n"
                    f"Insight: {decision.get('clinical_insight', 'N/A')}\n"
                ),
            )
            self._notify_hospital_api_mock(case_id, "MODERATE")

        elif risk_level == "LOW":
            actions_taken.append("Log for routine review")
            self._notify_hospital_api_mock(case_id, "LOW")

        else:
            return self._build_response(
                status="error",
                case_id=case_id,
                actions=[f"No action taken because risk_level '{risk_level}' is unsupported"],
            )

        return self._build_response(
            status="success",
            case_id=case_id,
            actions=actions_taken,
        )

    def _build_response(
        self,
        status: str,
        case_id: str | None,
        actions: List[str],
    ) -> Dict[str, Any]:
        response = {
            "agent": "action",
            "status": status,
            "actions": actions,
        }
        if case_id is not None:
            response["case_id"] = case_id
        return response

    def _send_sms_if_configured(self, message: str) -> None:
        """
        Sends SMS only if Twilio config is present.
        Safe no-op otherwise.
        """
        account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        from_number = os.getenv("TWILIO_FROM_NUMBER")
        to_number = os.getenv("ALERT_TO_PHONE")

        if not all([account_sid, auth_token, from_number, to_number]):
            print("[ACTION] Twilio not configured. Skipping SMS.")
            return

        try:
            from twilio.rest import Client

            client = Client(account_sid, auth_token)
            client.messages.create(
                body=message,
                from_=from_number,
                to=to_number,
            )
            print("[ACTION] SMS sent successfully.")
        except Exception as exc:
            print(f"[ACTION] SMS failed: {exc}")

    def _send_email_if_configured(self, subject: str, body: str) -> None:
        """
        Sends email only if SMTP config is present.
        Safe no-op otherwise.
        """
        smtp_host = os.getenv("SMTP_HOST")
        smtp_port = os.getenv("SMTP_PORT")
        smtp_user = os.getenv("SMTP_USER")
        smtp_password = os.getenv("SMTP_PASSWORD")
        from_email = os.getenv("ALERT_FROM_EMAIL")
        to_email = os.getenv("ALERT_TO_EMAIL")

        if not all([smtp_host, smtp_port, smtp_user, smtp_password, from_email, to_email]):
            print("[ACTION] SMTP not configured. Skipping email.")
            return

        try:
            msg = EmailMessage()
            msg["Subject"] = subject
            msg["From"] = from_email
            msg["To"] = to_email
            msg.set_content(body)

            with smtplib.SMTP(smtp_host, int(smtp_port)) as server:
                server.starttls()
                server.login(smtp_user, smtp_password)
                server.send_message(msg)

            print("[ACTION] Email sent successfully.")
        except Exception as exc:
            print(f"[ACTION] Email failed: {exc}")

    def _notify_hospital_api_mock(self, case_id: str | None, risk_level: str) -> None:
        """
        Placeholder for hospital API integration.
        Replace with requests.post(...) or your internal service client later.
        """
        print(f"[ACTION] Mock hospital API notified for case_id={case_id}, risk_level={risk_level}")


if __name__ == "__main__":
    agent = ActionAgent()

    test_payloads = [
        {
            "agent": "decision",
            "status": "success",
            "case_id": "CASE-001",
            "decision": {
                "risk_level": "HIGH",
                "reasoning": ["Hemoglobin < 7 (critical threshold)"],
                "clinical_insight": "Severe anemia risk",
                "confidence_score": 0.92,
            },
        },
        {
            "agent": "decision",
            "status": "success",
            "case_id": "CASE-002",
            "decision": {
                "risk_level": "MODERATE",
                "reasoning": ["Hemoglobin between 7 and 10 (moderate anemia range)"],
                "clinical_insight": "Moderate anemia risk",
                "confidence_score": 0.84,
            },
        },
        {
            "agent": "decision",
            "status": "success",
            "case_id": "CASE-003",
            "decision": {
                "risk_level": "LOW",
                "reasoning": ["Hemoglobin >= 10 (not in severe/moderate anemia threshold)"],
                "clinical_insight": "No severe anemia risk detected",
                "confidence_score": 0.95,
            },
        },
        {
            "agent": "decision",
            "status": "error",
            "case_id": "CASE-004",
            "decision": {
                "risk_level": "UNKNOWN",
                "reasoning": ["Missing required field: hemoglobin"],
                "clinical_insight": "Insufficient data to assess anemia risk",
                "confidence_score": 0.0,
            },
        },
    ]

    for payload in test_payloads:
        print("=" * 60)
        print("INPUT:")
        print(payload)
        print("OUTPUT:")
        print(agent.run(payload))
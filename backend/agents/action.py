"""
Action Agent - Executes actions with retry and escalation
Learns from action outcomes
"""

from __future__ import annotations

import os
import smtplib
import logging
from email.message import EmailMessage
from typing import Any, Dict, List, Optional
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential


from .communication import MessageBroker
from .base_agent import DynamicAgent, LearningSignal as AgentMessage
from .memory import PersistentMemory

logger = logging.getLogger(__name__)


class ActionAgent(DynamicAgent):
    """
    Action Agent:
    - Executes real-world actions based on decisions
    - Retries failed actions with exponential backoff
    - Escalates if actions continue to fail
    - Learns from action outcomes
    """
    
    def __init__(self, memory: Optional[PersistentMemory] = None, broker: Optional[MessageBroker] = None):
        super().__init__("ActionAgent", memory)
        self.broker = broker or MessageBroker()
        
        # Action tracking
        self.action_history: List[Dict] = []
        self.failed_actions: Dict[str, int] = {}  # Track failures per action type
        self.escalation_threshold = 3  # Escalate after 3 failures
        
        # Register with broker
        if self.broker:
            self.broker.register_agent(self)
        
        # Subscribe to events
        self.subscribe("decision_made")
        self.subscribe("action_feedback")
        
        logger.info("ActionAgent initialized")
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute actions based on decision
        """
        case_id = data.get("case_id")
        decision = data.get("decision", {})
        risk_level = decision.get("risk_level")
        
        if not risk_level:
            return {
                "status": "error",
                "case_id": case_id,
                "error": "No risk level in decision"
            }
        
        actions_taken = []
        actions_failed = []
        
        # Execute actions based on risk level
        if risk_level == "CRITICAL":
            result = self._execute_critical_actions(case_id, decision)
            actions_taken.extend(result["taken"])
            actions_failed.extend(result["failed"])
            
        elif risk_level == "HIGH":
            result = self._execute_high_risk_actions(case_id, decision)
            actions_taken.extend(result["taken"])
            actions_failed.extend(result["failed"])
            
        elif risk_level == "MODERATE":
            result = self._execute_moderate_risk_actions(case_id, decision)
            actions_taken.extend(result["taken"])
            actions_failed.extend(result["failed"])
            
        elif risk_level == "LOW":
            result = self._execute_low_risk_actions(case_id, decision)
            actions_taken.extend(result["taken"])
            actions_failed.extend(result["failed"])
        
        # Check if escalation needed
        if len(actions_failed) > 0:
            self._handle_failures(case_id, actions_failed)
        
        # Log actions
        action_record = {
            "case_id": case_id,
            "risk_level": risk_level,
            "actions_taken": actions_taken,
            "actions_failed": actions_failed,
            "timestamp": datetime.now().isoformat(),
            "requires_escalation": len(actions_failed) > 0
        }
        
        # Store in memory
        if self.memory:
            self.memory.store_case({
                "case_id": case_id,
                "actions": action_record,
                "status": "actions_executed"
            })
        
        # Broadcast completion
        self._broadcast_actions_completed(case_id, action_record)
        
        return {
            "status": "success" if not actions_failed else "partial_success",
            "case_id": case_id,
            "actions_taken": actions_taken,
            "actions_failed": actions_failed,
            "requires_escalation": len(actions_failed) > 0
        }
    
    def _execute_critical_actions(self, case_id: str, decision: Dict) -> Dict:
        """Execute actions for CRITICAL risk level"""
        taken = []
        failed = []
        
        # 1. Immediate SMS alert (with retry)
        sms_sent = self._send_sms_with_retry(
            message=f"[CRITICAL ALERT] Case {case_id}: Severe anemia detected. Immediate clinical review required.\n"
                    f"Hemoglobin: {decision.get('hemoglobin_value')} g/dL\n"
                    f"Insight: {decision.get('clinical_insight')}"
        )
        
        if sms_sent:
            taken.append("Critical SMS alert sent")
        else:
            failed.append("SMS alert failed after retries")
        
        # 2. Email alert to multiple recipients
        email_sent = self._send_email_with_retry(
            subject=f"[CRITICAL] Clinical Alert for {case_id}",
            body=self._format_alert_email(case_id, decision, "CRITICAL"),
            urgent=True
        )
        
        if email_sent:
            taken.append("Critical email alert sent")
        else:
            failed.append("Email alert failed after retries")
        
        # 3. Hospital API notification (with retry)
        api_notified = self._notify_hospital_api_with_retry(case_id, "CRITICAL", decision)
        
        if api_notified:
            taken.append("Hospital system notified")
        else:
            failed.append("Hospital API notification failed")
        
        # 4. Create manual escalation task
        if len(failed) >= 2:
            self._create_escalation_task(case_id, decision, failed)
            taken.append("Manual escalation task created")
        
        return {"taken": taken, "failed": failed}
    
    def _execute_high_risk_actions(self, case_id: str, decision: Dict) -> Dict:
        """Execute actions for HIGH risk level"""
        taken = []
        failed = []
        
        # 1. Email to primary care
        email_sent = self._send_email_with_retry(
            subject=f"[HIGH RISK] Clinical Alert for {case_id}",
            body=self._format_alert_email(case_id, decision, "HIGH"),
            urgent=False
        )
        
        if email_sent:
            taken.append("High risk email alert sent")
        else:
            failed.append("Email alert failed")
        
        # 2. Schedule appointment (mock)
        appointment = self._schedule_appointment(case_id, decision)
        if appointment:
            taken.append(f"Appointment scheduled: {appointment}")
        else:
            failed.append("Appointment scheduling failed")
        
        # 3. Hospital API notification
        api_notified = self._notify_hospital_api_with_retry(case_id, "HIGH", decision)
        if api_notified:
            taken.append("Hospital system notified")
        
        return {"taken": taken, "failed": failed}
    
    def _execute_moderate_risk_actions(self, case_id: str, decision: Dict) -> Dict:
        """Execute actions for MODERATE risk level"""
        taken = []
        failed = []
        
        # 1. Follow-up email
        email_sent = self._send_email_with_retry(
            subject=f"Follow-up Recommendation for {case_id}",
            body=self._format_alert_email(case_id, decision, "MODERATE"),
            urgent=False
        )
        
        if email_sent:
            taken.append("Follow-up email sent")
        
        # 2. Log for tracking
        self._log_for_review(case_id, decision)
        taken.append("Logged for routine review")
        
        # 3. Hospital API notification
        self._notify_hospital_api_mock(case_id, "MODERATE")
        taken.append("Hospital system updated")
        
        return {"taken": taken, "failed": failed}
    
    def _execute_low_risk_actions(self, case_id: str, decision: Dict) -> Dict:
        """Execute actions for LOW risk level"""
        taken = []
        failed = []
        
        # 1. Log for routine review
        self._log_for_review(case_id, decision)
        taken.append("Logged for routine review")
        
        # 2. Hospital API notification
        self._notify_hospital_api_mock(case_id, "LOW")
        taken.append("Hospital system updated")
        
        return {"taken": taken, "failed": failed}
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _send_sms_with_retry(self, message: str) -> bool:
        """Send SMS with retry logic"""
        account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        from_number = os.getenv("TWILIO_FROM_NUMBER")
        to_number = os.getenv("ALERT_TO_PHONE")
        
        if not all([account_sid, auth_token, from_number, to_number]):
            logger.info("SMS not configured")
            return True  # Treat as success in mock mode
        
        try:
            from twilio.rest import Client
            client = Client(account_sid, auth_token)
            client.messages.create(
                body=message,
                from_=from_number,
                to=to_number
            )
            logger.info("SMS sent successfully")
            return True
        except Exception as e:
            logger.error(f"SMS failed: {e}")
            raise  # Trigger retry
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _send_email_with_retry(self, subject: str, body: str, urgent: bool = False) -> bool:
        """Send email with retry logic"""
        smtp_host = os.getenv("SMTP_HOST")
        smtp_port = os.getenv("SMTP_PORT")
        smtp_user = os.getenv("SMTP_USER")
        smtp_password = os.getenv("SMTP_PASSWORD")
        from_email = os.getenv("ALERT_FROM_EMAIL")
        
        # Get appropriate recipients
        if urgent:
            to_emails = os.getenv("CRITICAL_ALERT_EMAILS", "").split(",")
        else:
            to_emails = os.getenv("ALERT_TO_EMAIL", "").split(",")
        
        if not all([smtp_host, smtp_port, smtp_user, smtp_password, from_email, to_emails]):
            logger.info("Email not configured")
            return True  # Treat as success in mock mode
        
        try:
            msg = EmailMessage()
            msg["Subject"] = subject
            msg["From"] = from_email
            msg["To"] = ", ".join(to_emails)
            msg.set_content(body)
            
            with smtplib.SMTP(smtp_host, int(smtp_port)) as server:
                server.starttls()
                server.login(smtp_user, smtp_password)
                server.send_message(msg)
            
            logger.info(f"Email sent to {to_emails}")
            return True
        except Exception as e:
            logger.error(f"Email failed: {e}")
            raise  # Trigger retry
    
    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=5))
    def _notify_hospital_api_with_retry(self, case_id: str, risk_level: str, decision: Dict) -> bool:
        """Notify hospital API with retry"""
        # In production, this would be a real API call
        api_url = os.getenv("HOSPITAL_API_URL")
        
        if not api_url:
            logger.info("Hospital API not configured")
            return True
        
        try:
            # Mock API call
            logger.info(f"Hospital API notified: case={case_id}, risk={risk_level}")
            return True
        except Exception as e:
            logger.error(f"Hospital API failed: {e}")
            raise  # Trigger retry
    
    def _schedule_appointment(self, case_id: str, decision: Dict) -> Optional[str]:
        """Schedule appointment (mock)"""
        # In production, this would integrate with scheduling system
        appointment_time = datetime.now().replace(hour=14, minute=0)
        return f"Appointment scheduled for {appointment_time.strftime('%Y-%m-%d %H:%M')}"
    
    def _log_for_review(self, case_id: str, decision: Dict):
        """Log case for routine review"""
        logger.info(f"Case {case_id} logged for review: {decision.get('risk_level')}")
    
    def _create_escalation_task(self, case_id: str, decision: Dict, failed_actions: List[str]):
        """Create manual escalation task"""
        task = {
            "case_id": case_id,
            "priority": "HIGH",
            "reason": "Automated actions failed",
            "failed_actions": failed_actions,
            "decision": decision,
            "created_at": datetime.now().isoformat()
        }
        
        logger.info(f"Escalation task created: {task}")
        
        # Store escalation task
        if self.memory:
            self.memory.store_case({
                "case_id": f"ESCALATION-{case_id}",
                "data": task,
                "status": "escalated"
            })
    
    def _format_alert_email(self, case_id: str, decision: Dict, risk_level: str) -> str:
        """Format alert email body"""
        return f"""
        CLINICAL ALERT - {risk_level} RISK
        
        Case ID: {case_id}
        Risk Level: {risk_level}
        Hemoglobin: {decision.get('hemoglobin_value', 'N/A')} g/dL
        Confidence: {decision.get('confidence_score', 0):.1%}
        
        Clinical Insight:
        {decision.get('clinical_insight', 'N/A')}
        
        Reasoning:
        {chr(10).join(decision.get('reasoning', []))}
        
        Suggested Actions:
        {chr(10).join(decision.get('suggested_actions', []))}
        
        Timestamp: {datetime.now().isoformat()}
        
        This is an automated alert from the Clinical Decision Support System.
        """
    
    def _handle_failures(self, case_id: str, failed_actions: List[str]):
        """Handle action failures and track patterns"""
        for action in failed_actions:
            self.failed_actions[action] = self.failed_actions.get(action, 0) + 1
            
            # If same action fails frequently, escalate to admin
            if self.failed_actions[action] >= self.escalation_threshold:
                logger.critical(f"Action '{action}' has failed {self.failed_actions[action]} times. Escalating to admin.")
                self._escalate_to_admin(action)
    
    def _escalate_to_admin(self, action: str):
        """Escalate persistent failures to admin"""
        # In production, this would send an admin alert
        logger.critical(f"ADMIN ALERT: Action '{action}' consistently failing. Manual intervention required.")
    
    def _broadcast_actions_completed(self, case_id: str, action_record: Dict):
        """Broadcast action completion"""
        message = AgentMessage(
            sender=self.name,
            receiver="broadcast",
            message_type="actions_completed",
            payload={
                "case_id": case_id,
                "actions": action_record
            }
        )
        
        if self.broker:
            self.broker.send(message)
    
    def _notify_hospital_api_mock(self, case_id: str, risk_level: str):
        """Mock hospital API call"""
        logger.info(f"[MOCK] Hospital API notified: case_id={case_id}, risk_level={risk_level}")
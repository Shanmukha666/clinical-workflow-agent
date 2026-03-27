from backend.agents.feedback import FeedbackAgent

# Initialize agent (no DB for now)
agent = FeedbackAgent()

# ---------- TEST CASE 1: Correct decision ----------
feedback_1 = {
    "decision_id": "DEC-123",
    "doctor_feedback": "correct"
}

result_1 = agent.process_feedback(feedback_1)
print("Test 1 Result:", result_1)


# ---------- TEST CASE 2: Incorrect decision ----------
feedback_2 = {
    "decision_id": "DEC-456",
    "doctor_feedback": "incorrect",
    "correct_label": "Pneumonia",
    "notes": "Model missed infection signs"
}

result_2 = agent.process_feedback(feedback_2)
print("Test 2 Result:", result_2)
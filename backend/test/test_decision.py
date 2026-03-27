from backend.agents.decision import DecisionAgent

decision_agent = DecisionAgent()

payload = {
    "structured_data": {
        "hemoglobin": 6.5
    }
}

result = decision_agent.run(payload)
print(result)
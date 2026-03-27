import re


class ExtractionAgent:
    def __init__(self):
        self.name = "ExtractionAgent"

    def run(self, data):
        """
        Main agent execution method.
        Takes global data, processes it, returns updated data.
        """

        raw_data = data.get("raw_data", "").lower()

        # Step 1: Extract structured lab data
        structured = self.extract_lab_values(raw_data)

        # Step 2: Extract medical entities
        entities = self.extract_medical_entities(raw_data)

        # Step 3: Update global contract
        data["structured_data"] = structured
        data["medical_entities"] = entities

        # Step 4: Add agent log
        data["logs"].append({
            "agent": self.name,
            "message": "Extracted structured medical data and entities"
        })

        return data

    def extract_lab_values(self, text):
        patterns = {
            "hemoglobin": r"(hb|hemoglobin)\s*[:\-]?\s*(\d+\.?\d*)",
            "wbc": r"(wbc)\s*[:\-]?\s*(\d+)",
            "platelet_count": r"(platelets?)\s*[:\-]?\s*(\d+)",
            "creatinine": r"(creatinine)\s*[:\-]?\s*(\d+\.?\d*)",
        }

        result = {}

        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = match.group(2)
                result[key] = float(value) if '.' in value else int(value)

        return result

    def extract_medical_entities(self, text):
        symptom_keywords = ["fatigue", "fever", "weakness", "pain", "cough", "breathlessness"]
        condition_keywords = ["anemia", "infection", "diabetes", "hypertension"]
        medication_keywords = ["paracetamol", "ibuprofen", "insulin"]

        symptoms = [w for w in symptom_keywords if w in text]
        conditions = [w for w in condition_keywords if w in text]
        medications = [w for w in medication_keywords if w in text]

        return {
            "symptoms": symptoms,
            "conditions": conditions,
            "medications": medications
        }


# ✅ TEST BLOCK
if __name__ == "__main__":
    print("🚀 Running Extraction Agent...\n")

    agent = ExtractionAgent()

    data = {
        "patient_id": "123",
        "input_type": "text",
        "raw_data": "Patient report: Hb: 6.5 g/dL, WBC: 12000, fatigue and shortness of breath. Condition: anemia.",
        "structured_data": {},
        "medical_entities": {},
        "decision": {},
        "actions": [],
        "logs": []
    }

    result = agent.run(data)

    print("✅ OUTPUT:\n")
    print(result)
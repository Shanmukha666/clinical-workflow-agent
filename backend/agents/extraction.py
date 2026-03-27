import re


class ExtractionAgent:
    def __init__(self):
        # No heavy models — simple logic (hackathon safe)
        pass

    def process(self, data):
        """
        Process the input data to extract structured medical data and entities.
        Updates the data dict with 'structured_data' and 'medical_entities'.
        """
        raw_data = data.get("raw_data", "").lower()

        data["structured_data"] = self.extract_lab_values(raw_data)
        data["medical_entities"] = self.extract_medical_entities(raw_data)

        # Add log entry
        data["logs"].append("Extraction agent processed raw data.")

        return data

    def extract_lab_values(self, text):
        """
        Extract lab values using regex patterns.
        """
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
        """
        Extract symptoms, conditions, medications using keyword matching.
        """

        # Basic keyword lists (expand if needed)
        symptom_keywords = ["fatigue", "fever", "weakness", "pain", "cough", "breathlessness"]
        condition_keywords = ["anemia", "infection", "diabetes", "hypertension"]
        medication_keywords = ["paracetamol", "ibuprofen", "insulin"]

        symptoms = []
        conditions = []
        medications = []

        for word in symptom_keywords:
            if word in text:
                symptoms.append(word)

        for word in condition_keywords:
            if word in text:
                conditions.append(word)

        for word in medication_keywords:
            if word in text:
                medications.append(word)

        return {
            "symptoms": symptoms,
            "conditions": conditions,
            "medications": medications
        }


# ✅ TEST BLOCK
if __name__ == "__main__":
    agent = ExtractionAgent()

    data = {
        "patient_id": "123",
        "input_type": "text",
        "raw_data": "Patient report: Hb: 6.5 g/dL, WBC: 12000, symptoms include fatigue and shortness of breath. Condition: anemia.",
        "structured_data": {},
        "medical_entities": {},
        "decision": {},
        "actions": [],
        "logs": []
    }

    updated_data = agent.process(data)

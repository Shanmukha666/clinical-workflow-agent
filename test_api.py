import requests
import json

payload = {
    "text": "Male patient, 45. Hemoglobin: 6.2. Severe fatigue.",
    "priority": "high"
}

headers = {
    "X-API-Key": "your-secret-api-key-change-in-production",
    "Content-Type": "application/json"
}

try:
    response = requests.post(
        "http://localhost:8000/process",
        json=payload,
        headers=headers
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response Headers: {dict(response.headers)}")
    print(f"Response Text: {response.text}")
    
    if response.status_code == 500:
        print("\n❌ Server Error - Check server terminal for traceback")
        
except Exception as e:
    print(f"Error: {e}")
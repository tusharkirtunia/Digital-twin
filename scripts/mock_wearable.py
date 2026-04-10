# scripts/mock_wearable.py
import requests, random, time

TOKEN = "paste-your-jwt-token-here"
URL = "http://localhost:8000/health/ingest"

while True:
    data = {
        "heart_rate": random.randint(60, 110),
        "steps": random.randint(200, 1500),
        "spo2": round(random.uniform(95, 99.5), 1),
        "sleep_hours": round(random.uniform(4.5, 9.0), 1),
        "stress_level": random.randint(1, 10),
    }
    r = requests.post(URL, json=data, headers={"Authorization": f"Bearer {TOKEN}"})
    print(r.json())
    time.sleep(5)   # sends data every 5 seconds
# scripts/seed_fake_data.py
from app.services.wearable_ingest import ingest_wearable_data
from datetime import datetime, timedelta
import random

USER_ID = "paste-a-real-user-uuid-here"

for i in range(60):   # 60 days of fake history
    ingest_wearable_data(USER_ID, {
        "heart_rate": random.randint(58, 115),
        "steps": random.randint(1000, 12000),
        "spo2": round(random.uniform(94, 99.9), 1),
        "sleep_hours": round(random.uniform(4.0, 9.5), 1),
        "stress_level": random.randint(1, 10),
    })

print("Seeded 60 days of data for", USER_ID)
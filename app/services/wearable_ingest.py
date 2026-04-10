# app/services/wearable_ingest.py
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from datetime import datetime

client = InfluxDBClient(url="http://localhost:8086", token="YOUR_TOKEN", org="health-twin")
write_api = client.write_api(write_options=SYNCHRONOUS)

def ingest_wearable_data(user_id: str, data: dict):
    """
    data = {
        "heart_rate": 72,
        "steps": 450,
        "spo2": 98.5,
        "sleep_hours": 7.2,
        "stress_level": 3
    }
    """
    point = (
        Point("wearable_metrics")
        .tag("user_id", user_id)
        .field("heart_rate", data.get("heart_rate"))
        .field("steps", data.get("steps"))
        .field("spo2", data.get("spo2"))
        .field("sleep_hours", data.get("sleep_hours"))
        .field("stress_level", data.get("stress_level"))
        .time(datetime.utcnow(), WritePrecision.NANOSECOND)
    )
    write_api.write(bucket="health_data", record=point)

def get_user_timeseries(user_id: str, days: int = 30):
    query_api = client.query_api()
    query = f'''
        from(bucket:"health_data")
        |> range(start: -{days}d)
        |> filter(fn: (r) => r["user_id"] == "{user_id}")
        |> pivot(rowKey:["_time"], columnKey:["_field"], valueColumn:"_value")
    '''
    return query_api.query_data_frame(query)
# app/services/wearable_ingest.py
client = InfluxDBClient(
    url="http://localhost:8086",   # ← self-hosted, no dependency
    token="YOUR_TOKEN",            # ← this token is just a local password
    org="health-twin"
)
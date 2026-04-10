from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str
    SECRET_KEY: str
    INFLUXDB_URL: str = "http://localhost:8086"
    INFLUXDB_TOKEN: str
    INFLUXDB_ORG: str = "health-twin"
    REDIS_URL: str = "redis://localhost:6379/0"
    RABBITMQ_URL: str = "amqp://guest@localhost//"

    class Config:
        env_file = ".env"

settings = Settings()
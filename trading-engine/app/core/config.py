import os
from pydantic import BaseSettings

class Settings(BaseSettings):
    # 데이터베이스 설정
    DATABASE_URL: str
    PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    STATIC_DIR: str = os.path.join(PROJECT_ROOT, "static")
    TEMPLATE_DIR: str = os.path.join(PROJECT_ROOT, "templates")
    KAFKA_BOOTSTRAP_SERVERS: str
    KAFKA_ORDERS_TOPIC: str

    class Config:
        env_file = ".env"  # .env 파일 경로 지정

# 환경 변수 로드
settings = Settings()

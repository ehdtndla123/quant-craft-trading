import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    USE_SSH_TUNNEL: bool
    # 데이터베이스 설정
    DB_HOST: str
    DB_PORT: int
    DB_USERNAME: str
    DB_PASSWORD: str
    DB_NAME: str
    DB_ECHO: bool
    DB_POOL_SIZE: int
    DB_MAX_OVERFLOW: int

    # SSH 터널링 설정
    SSH_HOST: str
    SSH_PORT: int
    SSH_USERNAME: str
    SSH_KEY_PATH: str

    PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    STATIC_DIR: str = os.path.join(PROJECT_ROOT, "static")
    TEMPLATE_DIR: str = os.path.join(PROJECT_ROOT, "templates")
    KAFKA_BOOTSTRAP_SERVERS: str
    KAFKA_ORDERS_TOPIC: str

    class Config:
        env_file = ".env"


# 환경 변수 로드
settings = Settings()

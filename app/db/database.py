from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.core.config import settings

engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    Base.metadata.drop_all(bind=engine)
    print("Existing tables dropped.")
    Base.metadata.create_all(bind=engine)
    print("Database tables created.")

# 애플리케이션 시작 시 이 함수를 호출하여 데이터베이스를 초기화합니다.
def initialize_database():
    init_db()
    print("Database initialized successfully.")
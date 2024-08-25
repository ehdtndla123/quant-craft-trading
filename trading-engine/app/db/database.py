from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from app.core.config import settings

engine = create_engine(settings.DATABASE_URL)
session_factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
SessionLocal = scoped_session(session_factory=session_factory)
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


def initialize_database():
    init_db()
    print("Database initialized successfully.")

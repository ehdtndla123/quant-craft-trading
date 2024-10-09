from app.db.database_manager import db_manager
from sqlalchemy.orm import Session


def get_db() -> Session:
    db = next(db_manager.get_db())
    try:
        yield db
    finally:
        db.close()


def initialize_database():
    db_manager.initialize_database()


Base = db_manager.Base
tunnel = db_manager.tunnel
SessionLocal = db_manager.SessionLocal
engine = db_manager.engine

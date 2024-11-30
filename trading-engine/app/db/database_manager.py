from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from app.core.config import settings
from sshtunnel import SSHTunnelForwarder
import logging
from urllib.parse import quote_plus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def initialize(self):
        self.use_ssh_tunnel = settings.USE_SSH_TUNNEL
        self.tunnel = self.create_ssh_tunnel() if self.use_ssh_tunnel else None
        self.engine = self.create_engine()
        self.SessionLocal = self.create_session()
        self.Base = declarative_base()

    def create_ssh_tunnel(self):
        if not self.use_ssh_tunnel:
            return None
        try:
            tunnel = SSHTunnelForwarder(
                (settings.SSH_HOST, settings.SSH_PORT),
                ssh_username=settings.SSH_USERNAME,
                ssh_pkey=settings.SSH_KEY_PATH,
                remote_bind_address=(settings.DB_HOST, settings.DB_PORT),
                local_bind_address=('127.0.0.1', 0),
            )
            tunnel.start()
            logger.info(f"SSH 터널 생성 완료. 로컬 바인딩 주소: 127.0.0.1:{tunnel.local_bind_port}")
            return tunnel
        except Exception as e:
            logger.error(f"SSH 터널 생성 중 오류 발생: {str(e)}")
            raise

    def create_engine(self):
        encoded_password = quote_plus(settings.DB_PASSWORD)
        if self.use_ssh_tunnel:
            DATABASE_URL = f"postgresql://{settings.DB_USERNAME}:{encoded_password}@{settings.DB_HOST}:{self.tunnel.local_bind_port}/{settings.DB_NAME}?options=-csearch_path%3Dtrade"
        else:
            DATABASE_URL = f"postgresql://{settings.DB_USERNAME}:{encoded_password}@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}?options=-csearch_path%3Dtrade"
        return create_engine(DATABASE_URL, echo=settings.DB_ECHO, pool_size=settings.DB_POOL_SIZE,
                             max_overflow=settings.DB_MAX_OVERFLOW)

    def create_session(self):
        session_factory = sessionmaker(bind=self.engine, autocommit=False, autoflush=False)
        return scoped_session(session_factory)

    def get_db(self):
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()

    def init_db(self):
        pass
        with self.engine.connect() as conn:
            conn.execute(text('CREATE SCHEMA IF NOT EXISTS trade'))
            conn.commit()
        self.Base.metadata.drop_all(bind=self.engine)
        self.Base.metadata.create_all(bind=self.engine)

    def initialize_database(self):
        self.init_db()
        logger.info("Database initialized successfully.")

    def stop_tunnel(self):
        if self.use_ssh_tunnel and self.tunnel:
            self.tunnel.stop()
            logger.info("SSH tunnel stopped.")


db_manager = DatabaseManager()

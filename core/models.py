"""
Database models (PostgreSQL + SQLAlchemy)
"""
import os
from sqlalchemy import create_engine, Column, String, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# PostgreSQL 연결 URL
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://user:password@localhost:5432/stt_db"
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# === ORM 모델 ===

class STTJob(Base):
    __tablename__ = "stt_job"

    stt_id = Column(String, primary_key=True, index=True)
    user_id = Column(Integer)
    status = Column(String, default="pending")

    # S3
    s3_audio_url = Column(String, nullable=True)
    s3_transcript_url = Column(String, nullable=True)

    # 결과
    symptoms = Column(String, nullable=True)
    diagnosis = Column(String, nullable=True)
    notes = Column(String, nullable=True)

    # 날짜 (YYYY-MM-DD 형식 문자열)
    date = Column(String, nullable=True)

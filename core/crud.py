"""
Database CRUD operations
"""
import time
from datetime import datetime
from sqlalchemy.orm import Session
from core.models import SessionLocal, STTJob, Base, engine


# === DB 세션 ===

def get_db():
    """
    FastAPI Depends에서 사용할 DB 세션 생성 함수
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# === CRUD 함수 ===

def create_stt_job(db: Session, user_id: int) -> STTJob:
    """
    초기 STT Job 생성 (status='pending')
    백엔드(port 8000)에서 사용
    """
    stt_id = f"{user_id}_{int(time.time() * 1000)}"

    item = STTJob(
        stt_id=stt_id,
        user_id=user_id,
        status="pending"
    )
    db.add(item)
    db.commit()
    db.refresh(item)
    return item


def update_stt_result(db: Session, stt_id: str, result: dict) -> STTJob | None:
    """
    STT 결과 업데이트
    POST /stt/{stt_id}/result에서 사용
    """
    job = db.query(STTJob).filter(STTJob.stt_id == stt_id).first()
    if not job:
        return None

    for key, value in result.items():
        if hasattr(job, key):
            setattr(job, key, value)

    db.commit()
    db.refresh(job)
    return job


def get_stt_job(db: Session, stt_id: str) -> STTJob | None:
    """
    STT Job 조회
    GET /stt/{stt_id}/status에서 사용
    """
    return db.query(STTJob).filter(STTJob.stt_id == stt_id).first()


# === CLI용 함수 (test_cli.py에서 사용) ===

def init_db():
    """
    DB 초기화 (테이블 생성)
    PostgreSQL에서는 자동으로 생성되므로 이 함수는 호환성을 위해 유지
    """
    Base.metadata.create_all(bind=engine)
    print(f"✅ PostgreSQL tables created")


def save_transcript(result: dict, processing_time: float, audio_length: float, rtf: float) -> str:
    """
    CLI용 STT 결과 저장
    tests/test_cli.py에서 사용
    """
    db = SessionLocal()
    try:
        user_id = 1  # CLI는 user_id=1 고정
        stt_id = f"{user_id}_{int(time.time() * 1000)}"

        item = STTJob(
            stt_id=stt_id,
            user_id=user_id,
            status="completed",
            # transcript_text는 나중에 추가 가능
        )
        db.add(item)
        db.commit()
        db.refresh(item)
        return stt_id
    finally:
        db.close()


def save_summary(stt_id: str, symptoms: str, diagnosis_name: str, notes: str) -> int:
    """
    CLI용 요약 저장
    tests/test_cli.py에서 사용
    """
    db = SessionLocal()
    try:
        job = db.query(STTJob).filter(STTJob.stt_id == stt_id).first()
        if not job:
            return None

        job.symptoms = symptoms
        job.diagnosis = diagnosis_name
        job.notes = notes
        job.date = datetime.now().strftime("%Y-%m-%d")  # 오늘 날짜 자동 설정
        job.status = "done"

        db.commit()
        return 1  # 성공
    finally:
        db.close()

"""
SQLite/PostgreSQL 데이터베이스 저장 유틸리티

주요 기능:
- init_db: 데이터베이스 및 테이블 초기화
- save_transcript: STT 변환 결과 저장
- save_summary: AI 요약 결과 저장

향후 PostgreSQL 마이그레이션 시 이 파일만 수정
"""
import sqlite3
from pathlib import Path
from datetime import datetime


def init_db(db_path: str):
    """DB 파일과 테이블을 생성(없으면)."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    
    # STT 변환 결과 테이블
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS STT_Transcript (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            audio_file TEXT,
            model TEXT,
            transcript_text TEXT,
            processing_time REAL,
            audio_duration REAL,
            rtf REAL,
            file_size INTEGER,
            noise_reduction INTEGER,
            s3_url TEXT,
            stt_error TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    # STT 요약 결과 테이블
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS STT_Summary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transcript_id INTEGER NOT NULL,
            chief_complaint TEXT,
            diagnosis TEXT,
            recommendation TEXT,
            model TEXT,
            summary_time REAL,
            summary_error TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(transcript_id) REFERENCES STT_Transcript(id)
        )
        """
    )
    
    con.commit()
    con.close()


def save_transcript(
    result: dict,
    processing_time: float,
    audio_duration: float,
    rtf: float,
    noise_reduction: bool,
    db_path: str,
    file_size: int = None,
    stt_error: str = None
) -> int:
    """STT_Transcript 테이블에 저장하고 id 반환."""
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute(
        """
        INSERT INTO STT_Transcript (
            audio_file, model, transcript_text, processing_time, audio_duration, rtf,
            file_size, noise_reduction, stt_error, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            result.get("audio_file"),
            result.get("model"),
            result.get("text"),
            float(processing_time) if processing_time is not None else None,
            float(audio_duration) if audio_duration is not None else None,
            float(rtf) if rtf is not None else None,
            file_size,
            1 if noise_reduction else 0,
            stt_error,
            datetime.now().isoformat(),
        ),
    )
    tid = cur.lastrowid
    con.commit()
    con.close()
    return tid


def save_summary(
    transcript_id: int,
    chief_complaint: str,
    diagnosis: str,
    recommendation: str,
    model: str,
    summary_time: float,
    db_path: str,
    summary_error: str = None
) -> int:
    """
    AI 요약 결과를 STT_Summary 테이블에 저장

    Args:
        transcript_id: STT_Transcript 테이블의 ID (외래키)
        chief_complaint: 증상
        diagnosis: 진단
        recommendation: 권고사항
        model: 사용한 GPT 모델 (예: gpt-4o-mini)
        summary_time: 요약 생성 시간(초)
        db_path: DB 파일 경로
        summary_error: 요약 생성 중 발생한 에러 메시지

    Returns:
        생성된 summary의 ID
    """
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    cur.execute(
        """
        INSERT INTO STT_Summary (
            transcript_id, chief_complaint, diagnosis, recommendation, model, summary_time, summary_error
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            transcript_id,
            chief_complaint,
            diagnosis,
            recommendation,
            model,
            float(summary_time) if summary_time is not None else None,
            summary_error,
        ),
    )

    summary_id = cur.lastrowid
    con.commit()
    con.close()

    return summary_id




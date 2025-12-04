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
            stt_id  INTEGER PRIMARY KEY AUTOINCREMENT,
            audio_file TEXT,
            transcript_text TEXT,
            processing_time REAL,
            audio_length REAL,
            rtf REAL,
            file_size INTEGER,
            s3_url TEXT,
            stt_status TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    # STT 요약 결과 테이블 (추후 retry 기능 추가) -> visit에 response
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS STT_Summary (
            summ_id INTEGER PRIMARY KEY AUTOINCREMENT,
            stt_id INTEGER NOT NULL,
            symptoms TEXT,
            diagnosis_name TEXT,
            notes TEXT,            
            summary_time REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(stt_id) REFERENCES STT_Transcript(stt_id)
        )
        """
    )
    
    con.commit()
    con.close()


def save_transcript(
    result: dict,
    processing_time: float,
    audio_length: float,
    rtf: float,
    db_path: str,
    file_size: int = None,
    s3_url: str = None,  # 프로덕션: S3 URL, 로컬 테스트: 로컬 경로
    stt_status: str = None
) -> int:
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute(
        """
        INSERT INTO STT_Transcript (
            audio_file, transcript_text, processing_time, audio_length, rtf,
            file_size, s3_url, stt_status, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            result.get("audio_file"),
            result.get("text"),
            float(processing_time) if processing_time is not None else None,
            float(audio_length) if audio_length is not None else None,
            float(rtf) if rtf is not None else None,
            file_size,
            s3_url,
            stt_status,
            datetime.now().isoformat(),
        ),
    )
    stt_id = cur.lastrowid #transcript id
    con.commit()
    con.close()
    return stt_id


def save_summary(
    stt_id: int,
    symptoms: str,
    diagnosis_name: str,
    notes: str,
    summary_time: float,
    db_path: str,
    summary_error: str = None
) -> int:
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    cur.execute(
        """
        INSERT INTO STT_Summary (
            stt_id, symptoms, diagnosis_name, notes, summary_time
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            stt_id,
            symptoms,
            diagnosis_name,
            notes,
            float(summary_time) if summary_time is not None else None,
        ),
    )

    summ_id = cur.lastrowid
    con.commit()
    con.close()

    return summ_id




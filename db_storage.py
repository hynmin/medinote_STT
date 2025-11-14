"""
SQLite/PostgreSQL 데이터베이스 저장 유틸리티

주요 기능:
- init_db: 데이터베이스 및 테이블 초기화
- save_transcript: STT 변환 결과 저장
- save_metrics: 평가 지표 저장 (개발/테스트용)
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
            text TEXT,
            processing_time REAL,
            audio_duration REAL,
            rtf REAL,
            noise_reduction INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    # 개발 평가 지표 테이블 (정답 스크립트 있을 때)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS Dev_Metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transcript_id INTEGER NOT NULL,
            wer REAL,
            cer REAL,
            ref_chars INTEGER,
            hyp_chars INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(transcript_id) REFERENCES STT_Transcript(id)
        )
        """
    )

    # 관리 평가 지표 테이블 (프로덕션 모니터링)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS STT_Metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transcript_id INTEGER NOT NULL,
            avg_confidence REAL,
            min_confidence REAL,
            low_confidence_ratio REAL,
            silence_ratio REAL,
            audio_rms_energy REAL,
            clipping_detected INTEGER,
            word_count INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(transcript_id) REFERENCES STT_Transcript(id)
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
            medication TEXT,
            lifestyle_management TEXT,
            model TEXT,
            summary_time REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(transcript_id) REFERENCES STT_Transcript(id)
        )
        """
    )
    
    con.commit()
    con.close()


def save_transcript(result: dict, processing_time: float, audio_duration: float, rtf: float, noise_reduction: bool, db_path: str) -> int:
    """STT_Transcript 테이블에 저장하고 id 반환."""
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute(
        """
        INSERT INTO STT_Transcript (
            audio_file, model, text, processing_time, audio_duration, rtf, noise_reduction, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            result.get("audio_file"),
            result.get("model"),
            result.get("text"),
            float(processing_time) if processing_time is not None else None,
            float(audio_duration) if audio_duration is not None else None,
            float(rtf) if rtf is not None else None,
            1 if noise_reduction else 0,
            datetime.now().isoformat(),
        ),
    )
    tid = cur.lastrowid
    con.commit()
    con.close()
    return tid


def save_dev_metrics(transcript_id: int, metrics: dict, db_path: str):
    """Dev_Metrics 테이블에 저장 (개발/테스트용)"""
    if not metrics:
        return
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute(
        """
        INSERT INTO Dev_Metrics (
            transcript_id, wer, cer, ref_chars, hyp_chars
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (
            transcript_id,
            float(metrics.get("wer", 0.0)),
            float(metrics.get("cer", 0.0)),
            int(metrics.get("ref_chars", 0)),
            int(metrics.get("hyp_chars", 0)),
        ),
    )
    con.commit()
    con.close()


def save_stt_metrics(transcript_id: int, metrics: dict, db_path: str):
    """STT_Metrics 테이블에 저장 (프로덕션 모니터링용)"""
    if not metrics:
        return
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute(
        """
        INSERT INTO STT_Metrics (
            transcript_id, avg_confidence, min_confidence, low_confidence_ratio,
            silence_ratio, audio_rms_energy, clipping_detected, word_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            transcript_id,
            float(metrics.get("avg_confidence", 0.0)),
            float(metrics.get("min_confidence", 0.0)),
            float(metrics.get("low_confidence_ratio", 0.0)),
            float(metrics.get("silence_ratio", 0.0)),
            float(metrics.get("audio_rms_energy", 0.0)),
            int(metrics.get("clipping_detected", 0)),
            int(metrics.get("word_count", 0)),
        ),
    )
    con.commit()
    con.close()


def save_summary(
    transcript_id: int,
    chief_complaint: str,
    diagnosis: str,
    medication: str,
    lifestyle_management: str,
    model: str,
    summary_time: float,
    db_path: str
) -> int:
    """
    AI 요약 결과를 STT_Summary 테이블에 저장

    Args:
        transcript_id: STT_Transcript 테이블의 ID (외래키)
        chief_complaint: 주요 증상
        diagnosis: 진단
        medication: 약물 처방
        lifestyle_management: 생활 관리 및 재방문
        model: 사용한 GPT 모델 (예: gpt-4o-mini)
        summary_time: 요약 생성 시간(초)
        db_path: DB 파일 경로

    Returns:
        생성된 summary의 ID
    """
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    cur.execute(
        """
        INSERT INTO STT_Summary (
            transcript_id, chief_complaint, diagnosis,
            medication, lifestyle_management, model, summary_time
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            transcript_id,
            chief_complaint,
            diagnosis,
            medication,
            lifestyle_management,
            model,
            float(summary_time) if summary_time is not None else None,
        ),
    )

    summary_id = cur.lastrowid
    con.commit()
    con.close()

    return summary_id




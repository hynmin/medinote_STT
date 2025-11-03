"""
개발용 SQLite 저장 유틸리티
"""
import sqlite3
from pathlib import Path
from datetime import datetime


def init_db(db_path: str):
    """DB 파일과 테이블을 생성(없으면)."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS transcripts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            audio_file TEXT,
            model TEXT,
            text TEXT,
            diarization_enabled INTEGER,
            num_speakers INTEGER,
            processing_time REAL,
            audio_duration REAL,
            rtf REAL,
            noise_reduction INTEGER,
            vad_filter INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS segments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transcript_id INTEGER NOT NULL,
            speaker TEXT,
            text TEXT,
            start_sec REAL,
            end_sec REAL,
            FOREIGN KEY(transcript_id) REFERENCES transcripts(id)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transcript_id INTEGER NOT NULL,
            wer REAL,
            cer REAL,
            ref_chars INTEGER,
            hyp_chars INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(transcript_id) REFERENCES transcripts(id)
        )
        """
    )
    con.commit()
    con.close()


def save_transcript(result: dict, processing_time: float, audio_duration: float, rtf: float, noise_reduction: bool, vad_filter: bool, db_path: str) -> int:
    """transcripts 테이블에 저장하고 id 반환."""
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute(
        """
        INSERT INTO transcripts (
            audio_file, model, text, diarization_enabled, num_speakers, processing_time, audio_duration, rtf, noise_reduction, vad_filter, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            result.get("audio_file"),
            result.get("model"),
            result.get("text"),
            1 if result.get("segments") else 0,
            int(result.get("num_speakers", 0)),
            float(processing_time) if processing_time is not None else None,
            float(audio_duration) if audio_duration is not None else None,
            float(rtf) if rtf is not None else None,
            1 if noise_reduction else 0,
            1 if vad_filter else 0,
            datetime.now().isoformat(),
        ),
    )
    tid = cur.lastrowid
    con.commit()
    con.close()
    return tid


def save_segments(transcript_id: int, segments: list, db_path: str):
    """segments 테이블에 일괄 저장."""
    if not segments:
        return
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    rows = [
        (
            transcript_id,
            s.get("speaker"),
            s.get("text"),
            float(s.get("start", 0.0)),
            float(s.get("end", 0.0)),
        )
        for s in segments
    ]
    cur.executemany(
        """
        INSERT INTO segments (
            transcript_id, speaker, text, start_sec, end_sec
        ) VALUES (?, ?, ?, ?, ?)
        """,
        rows,
    )
    con.commit()
    con.close()


def save_metrics(transcript_id: int, metrics: dict, db_path: str):
    """metrics 테이블에 저장."""
    if not metrics:
        return
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute(
        """
        INSERT INTO metrics (
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

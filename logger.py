# logger.py
# SQLite session telemetry: one table for sessions, one for each feedback moment.

import sqlite3
from datetime import datetime
import config


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def init_db():
    conn = sqlite3.connect(config.DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TEXT,
            ended_at TEXT,
            planned_routine TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            timestamp TEXT,
            pose TEXT,
            transcript TEXT,
            compound REAL,
            sentiment TEXT,
            action TEXT,
            next_pose TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """)
    conn.commit()
    conn.close()


def start_session(routine):
    """Creates a session row and returns its id."""
    conn = sqlite3.connect(config.DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO sessions (started_at, planned_routine) VALUES (?, ?)",
        (_now(), ",".join(routine)),
    )
    session_id = cursor.lastrowid
    conn.commit()
    conn.close()
    print(f"🗄️  Session {session_id} started")
    return session_id


def end_session(session_id):
    conn = sqlite3.connect(config.DB_PATH)
    cursor = conn.cursor()
    cursor.execute("UPDATE sessions SET ended_at = ? WHERE id = ?", (_now(), session_id))
    conn.commit()
    conn.close()
    print(f"🗄️  Session {session_id} ended")


def log_feedback(session_id, pose, transcript, compound, sentiment, action, next_pose):
    conn = sqlite3.connect(config.DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO feedback_events
           (session_id, timestamp, pose, transcript, compound, sentiment, action, next_pose)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (session_id, _now(), pose, transcript, compound, sentiment, action, next_pose),
    )
    conn.commit()
    conn.close()
    print(f"🗄️  Logged: {pose} | '{transcript}' | {sentiment} -> {action}")
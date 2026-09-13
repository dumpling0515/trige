import sqlite3
from pathlib import Path

DB_PATH = Path("data/trige.db")


def get_connection():
    DB_PATH.parent.mkdir(exist_ok=True)
    return sqlite3.connect(DB_PATH)


def initialize_database():
    conn = get_connection()

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            pneumonia_score REAL NOT NULL,
            alert_flag TEXT NOT NULL,
            model_version TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    conn.commit()
    conn.close()


def save_scan(filename, score, alert_flag):
    conn = get_connection()

    conn.execute(
        """
        INSERT INTO scans (
            filename,
            pneumonia_score,
            alert_flag,
            model_version
        )
        VALUES (?, ?, ?, ?)
        """,
        (
            filename,
            score,
            alert_flag,
            "densenet121-res224-all"
        ),
    )

    conn.commit()
    conn.close()


def get_scans():
    conn = get_connection()

    rows = conn.execute(
        """
        SELECT
            id,
            filename,
            pneumonia_score,
            alert_flag,
            model_version,
            created_at
        FROM scans
        ORDER BY id DESC
        """
    ).fetchall()

    conn.close()

    return rows
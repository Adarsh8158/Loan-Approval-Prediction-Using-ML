"""
SQLite Database Service for LoanIQ Prediction History.
Stores applicant records, prediction results, probability scores, and timestamps.
"""

import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).resolve().parent.parent / "database.sqlite3"


def get_connection():
    """Create and return a database connection with row factory enabled."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize predictions table if it does not already exist."""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                applicant_summary TEXT NOT NULL,
                cibil_score INTEGER NOT NULL,
                loan_amount REAL NOT NULL,
                loan_term REAL NOT NULL,
                predicted_status TEXT NOT NULL,
                confidence REAL NOT NULL,
                monthly_emi REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()


def save_prediction_record(data: dict, result: dict) -> int:
    """Save a successful prediction to SQLite database."""
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        income_val = float(data.get("income_annum", 0))
        summary = f"CIBIL: {data.get('cibil_score', 0)}, Income: {income_val:,.0f}, Edu: {data.get('education', 'Graduate')}"
        cursor.execute("""
            INSERT INTO predictions (
                applicant_summary, cibil_score, loan_amount, loan_term,
                predicted_status, confidence, monthly_emi, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            summary,
            int(data.get("cibil_score", 0)),
            float(data.get("loan_amount", 0)),
            float(data.get("loan_term", 1)),
            result.get("status", "Unknown"),
            float(result.get("confidence", 0)),
            float(result.get("monthly_emi", 0)),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))
        conn.commit()
        return cursor.lastrowid


def get_prediction_history(limit: int = 50):
    """Retrieve recent predictions from the database."""
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, applicant_summary, cibil_score, loan_amount, loan_term,
                   predicted_status, confidence, monthly_emi, created_at
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
        """, (limit,))
        return [dict(row) for row in cursor.fetchall()]


def clear_prediction_history():
    """Clear all records from the predictions table."""
    init_db()
    with get_connection() as conn:
        conn.execute("DELETE FROM predictions")
        conn.commit()

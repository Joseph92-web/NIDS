# db.py
import sqlite3
from datetime import datetime

DB_PATH = 'detections.db'

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            prediction INTEGER,
            class_name TEXT,
            confidence REAL,
            source_ip TEXT,
            details TEXT
        )
    ''')
    conn.commit()
    conn.close()

def insert_alert(prediction, class_name, confidence, source_ip=None, details=None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO alerts (prediction, class_name, confidence, source_ip, details, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (prediction, class_name, confidence, source_ip, details, datetime.now()))
    conn.commit()
    conn.close()

def get_recent_alerts(limit=100):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT timestamp, class_name, confidence, source_ip
        FROM alerts
        ORDER BY timestamp DESC
        LIMIT ?
    ''', (limit,))
    rows = cursor.fetchall()
    conn.close()
    return rows

# Initialize DB when module loads
init_db()
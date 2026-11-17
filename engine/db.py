"""The one SQLite database (plan 10.1). This file holds the Pangram tables;
later pieces add their own."""
import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS pangram_cache (
    sha TEXT NOT NULL,
    model TEXT NOT NULL,
    response_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (sha, model)
);
CREATE TABLE IF NOT EXISTS pangram_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    sha TEXT NOT NULL,
    model TEXT NOT NULL,
    words INTEGER NOT NULL,
    credits INTEGER NOT NULL,
    usd REAL NOT NULL,
    purpose TEXT NOT NULL,
    label TEXT,
    status TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS baseline_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    run TEXT NOT NULL,
    kind TEXT NOT NULL,
    label TEXT NOT NULL,
    sha TEXT NOT NULL,
    model TEXT NOT NULL,
    words INTEGER NOT NULL,
    prediction TEXT,
    fraction_ai REAL,
    fraction_ai_assisted REAL,
    fraction_human REAL,
    max_window_score REAL,
    humanized INTEGER
);
"""


def connect(path):
    conn = sqlite3.connect(str(path))
    conn.executescript(SCHEMA)
    return conn

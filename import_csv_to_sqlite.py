#!/usr/bin/env python3
"""Импортирует ratings.csv и watched.csv в SQLite базу user_data.db

Запуск из корня проекта (где находятся ratings.csv и watched.csv):
    python import_csv_to_sqlite.py
"""
import csv
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / 'user_data.db'
RATINGS_CSV = ROOT / 'ratings.csv'
WATCHED_CSV = ROOT / 'watched.csv'


def create_tables(conn):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS movies (
            id INTEGER PRIMARY KEY,
            tmdb_id INTEGER,
            imdb_id TEXT,
            title TEXT NOT NULL,
            year INTEGER,
            letterboxd_uri TEXT,
            tmdb_type TEXT,
            UNIQUE(tmdb_id, title)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ratings (
            id INTEGER PRIMARY KEY,
            movie_id INTEGER NOT NULL,
            rating REAL,
            rated_at TEXT,
            source TEXT,
            FOREIGN KEY(movie_id) REFERENCES movies(id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS watched (
            id INTEGER PRIMARY KEY,
            movie_id INTEGER NOT NULL,
            watched_at TEXT,
            source TEXT,
            FOREIGN KEY(movie_id) REFERENCES movies(id)
        )
        """
    )


def upsert_movie(conn, tmdb_id, imdb_id, title, year, letterboxd_uri, tmdb_type):
    # normalize tmdb_id
    try:
        tmdb_val = int(tmdb_id) if tmdb_id not in (None, '') else None
    except Exception:
        tmdb_val = None

    conn.execute(
        "INSERT OR IGNORE INTO movies (tmdb_id, imdb_id, title, year, letterboxd_uri, tmdb_type) VALUES (?, ?, ?, ?, ?, ?)",
        (tmdb_val, imdb_id or None, title, int(year) if year not in (None, '') else None, letterboxd_uri or None, tmdb_type or None),
    )
    cur = conn.execute("SELECT id FROM movies WHERE tmdb_id=? AND title=?", (tmdb_val, title))
    r = cur.fetchone()
    if r:
        return r[0]
    cur = conn.execute("SELECT id FROM movies WHERE title=?", (title,))
    r = cur.fetchone()
    return r[0] if r else None


def import_ratings(conn, path):
    if not path.exists():
        print(f"ratings file not found: {path}")
        return 0
    inserted = 0
    with path.open(encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            date = row.get('Date')
            title = row.get('Name') or ''
            year = row.get('Year') or None
            letterboxd = row.get('Letterboxd URI') or row.get('LetterboxdURI')
            rating = row.get('Rating')
            tmdbtype = row.get('TmdbIdType') or row.get('TmdbIdType')
            tmdb = row.get('TmdbId') or row.get('TmdbID') or row.get('tmdb_id')
            imdb = row.get('ImdbId') or row.get('ImdbID') or row.get('imdb_id')
            movie_id = upsert_movie(conn, tmdb, imdb, title, year, letterboxd, tmdbtype)
            if movie_id is None:
                continue
            try:
                rating_val = float(rating) if rating not in (None, '') else None
            except Exception:
                rating_val = None
            conn.execute("INSERT INTO ratings (movie_id, rating, rated_at, source) VALUES (?, ?, ?, ?)", (movie_id, rating_val, date, 'ratings.csv'))
            inserted += 1
    conn.commit()
    return inserted


def import_watched(conn, path):
    if not path.exists():
        print(f"watched file not found: {path}")
        return 0
    inserted = 0
    with path.open(encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            date = row.get('Date')
            title = row.get('Name') or ''
            year = row.get('Year') or None
            letterboxd = row.get('Letterboxd URI') or row.get('LetterboxdURI')
            tmdbtype = row.get('TmdbIdType') or row.get('TmdbIdType')
            tmdb = row.get('TmdbId') or row.get('TmdbID') or row.get('tmdb_id')
            imdb = row.get('ImdbId') or row.get('ImdbID') or row.get('imdb_id')
            movie_id = upsert_movie(conn, tmdb, imdb, title, year, letterboxd, tmdbtype)
            if movie_id is None:
                continue
            conn.execute("INSERT INTO watched (movie_id, watched_at, source) VALUES (?, ?, ?)", (movie_id, date, 'watched.csv'))
            inserted += 1
    conn.commit()
    return inserted


def main():
    print("DB path:", DB_PATH)
    conn = sqlite3.connect(str(DB_PATH))
    create_tables(conn)
    r_count = import_ratings(conn, RATINGS_CSV)
    w_count = import_watched(conn, WATCHED_CSV)
    cur = conn.execute("SELECT COUNT(*) FROM movies")
    m_count = cur.fetchone()[0]
    print(f"Imported: movies={m_count}, ratings={r_count}, watched={w_count}")
    conn.close()


if __name__ == '__main__':
    main()

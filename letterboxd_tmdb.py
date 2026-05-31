#!/usr/bin/env python3
"""Resolve Letterboxd entries to TMDB ids without any external tool.

A raw Letterboxd export only contains Date,Name,Year,Letterboxd URI(,Rating).
This script pulls the exact TMDB id / type straight from the film page (the one
the short boxd.it URI redirects to), the same way the external mapper did, and
fills the TmdbId column in place.

Project pipeline:
    1) python letterboxd_tmdb.py        # enrich: fill TmdbId in the csv files
    2) python import_csv_to_sqlite.py    # import: offline load of ready csv into DB

Use as a library:
    from letterboxd_tmdb import LetterboxdResolver
    r = LetterboxdResolver()
    r.resolve("https://boxd.it/hTha")  # -> {"tmdb_id": 496243, "tmdb_type": "movie"}
"""
from __future__ import annotations

import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

import requests

ROOT = Path(__file__).resolve().parent
CACHE_PATH = ROOT / ".letterboxd_cache.json"

# Patterns found on a Letterboxd film page.
_RE_TMDB_ID = re.compile(r'data-tmdb-id="(\d+)"')
_RE_TMDB_TYPE = re.compile(r'data-tmdb-type="(\w+)"')
_RE_TMDB_LINK = re.compile(r'themoviedb\.org/(movie|tv)/(\d+)')

_UA = "Mozilla/5.0 (compatible; KinoAgent/1.0; +personal-use)"


class LetterboxdResolver:
    """Resolve a Letterboxd URI to a TMDB id/type, with an on-disk cache."""

    def __init__(
        self,
        cache_path: Path = CACHE_PATH,
        delay: float = 0.34,
        tmdb_api_key: Optional[str] = None,
    ):
        self.cache_path = cache_path
        self.delay = delay  # pause between LIVE requests (be polite to Letterboxd)
        self.tmdb_api_key = tmdb_api_key
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": _UA})
        self._cache: dict[str, dict] = {}
        if cache_path.exists():
            try:
                self._cache = json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                self._cache = {}

    # ---- public API ------------------------------------------------------
    def resolve(
        self,
        uri: Optional[str],
        title: Optional[str] = None,
        year: Optional[str] = None,
    ) -> dict:
        """Return {'tmdb_id', 'tmdb_type'}; values may be None.

        Try the Letterboxd page first (exact match); on failure fall back to a
        TMDB search by title+year (only if an api key is set). Cache is keyed by
        URI, so the same film in watched.csv and ratings.csv is fetched once.
        """
        key = (uri or f"title::{title}::{year}").strip()
        if key in self._cache:
            return self._cache[key]

        info = {"tmdb_id": None, "tmdb_type": None}
        if uri:
            info = self._scrape(uri) or info
        if info["tmdb_id"] is None and title:
            tmdb_id = self._tmdb_search(title, year)
            if tmdb_id is not None:
                info = {"tmdb_id": tmdb_id, "tmdb_type": "movie"}

        self._cache[key] = info
        self._flush()
        return info

    # ---- internals -------------------------------------------------------
    def _scrape(self, uri: str, retries: int = 3) -> Optional[dict]:
        for attempt in range(retries):
            try:
                resp = self.session.get(uri, timeout=15, allow_redirects=True)
            except requests.RequestException:
                time.sleep(1.5 * (attempt + 1))
                continue
            if resp.status_code in (429, 500, 502, 503, 504):
                time.sleep(1.5 * (attempt + 1))
                continue
            if not resp.ok:
                return None
            time.sleep(self.delay)  # polite pause after a successful live request
            html = resp.text
            tmdb_id = _first(_RE_TMDB_ID, html)
            tmdb_type = _first(_RE_TMDB_TYPE, html)
            if tmdb_id is None:
                m = _RE_TMDB_LINK.search(html)
                if m:
                    tmdb_type, tmdb_id = m.group(1), m.group(2)
            return {
                "tmdb_id": int(tmdb_id) if tmdb_id else None,
                "tmdb_type": tmdb_type,
            }
        return None

    def _tmdb_search(self, title: str, year: Optional[str]) -> Optional[int]:
        if not self.tmdb_api_key:
            return None
        try:
            params = {"api_key": self.tmdb_api_key, "query": title}
            if year:
                params["year"] = year
            r = self.session.get(
                "https://api.themoviedb.org/3/search/movie", params=params, timeout=15
            )
            results = r.json().get("results", [])
            return int(results[0]["id"]) if results else None
        except Exception:
            return None

    def _flush(self):
        try:
            self.cache_path.write_text(
                json.dumps(self._cache, ensure_ascii=False, indent=0), encoding="utf-8"
            )
        except Exception:
            pass


def _first(pattern: re.Pattern, text: str) -> Optional[str]:
    m = pattern.search(text)
    return m.group(1) if m else None


# ---- enrich a csv in place ----------------------------------------------
def enrich_csv(path: Path, resolver: LetterboxdResolver) -> int:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if not rows:
        print(f"{path.name}: empty")
        return 0
    fieldnames = list(rows[0].keys())
    for col in ("TmdbIdType", "TmdbId"):
        if col not in fieldnames:
            fieldnames.append(col)

    filled = 0
    for i, row in enumerate(rows, 1):
        if (row.get("TmdbId") or "").strip():
            continue  # already filled — do not resolve again
        info = resolver.resolve(
            row.get("Letterboxd URI"), row.get("Name"), row.get("Year")
        )
        if info["tmdb_id"]:
            row["TmdbId"] = info["tmdb_id"]
            row["TmdbIdType"] = (info["tmdb_type"] or "movie").capitalize()
            filled += 1
        print(f"  {path.name} {i}/{len(rows)}: {row.get('Name')} -> {info['tmdb_id']}")

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{path.name}: filled {filled}/{len(rows)}")
    return filled


def main(argv: list[str]):
    import os

    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except Exception:
        pass

    resolver = LetterboxdResolver(tmdb_api_key=os.getenv("TMDB_API_KEY"))
    paths = [Path(a) for a in argv] or [ROOT / "ratings.csv", ROOT / "watched.csv"]
    for p in paths:
        if p.exists():
            enrich_csv(p, resolver)
        else:
            print(f"file not found: {p}")


if __name__ == "__main__":
    main(sys.argv[1:])

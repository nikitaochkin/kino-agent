"""TMDB helpers: resolution, context fetch, discover, population probe, keyword canon."""
import threading
import time
from functools import lru_cache

import nltk
from langchain_core.tools import tool

from .config import tmdb
from .state import MovieContext


def _tmdb_call(fn, default, attempts: int = 3):
    """Run a TMDB call with retries; return `default` if all attempts fail.

    TMDB occasionally times out / drops a connection, and under the parallel fan-out
    one failure would otherwise crash the whole node. Transient blips are retried so
    a real value still gets cached; only a persistent failure falls back to default.
    """
    for i in range(attempts):
        try:
            return fn()
        except Exception:
            if i == attempts - 1:
                return default
            time.sleep(0.5 * (i + 1))

IMPORTANT_CREW_JOBS = {
    "Director",
    "Screenplay",
    "Writer",
    "Novel",
    "Original Music Composer",
    "Director of Photography",
    "Producer",
    "Editor",
    "Production Design",
}

# Built once at import (single TMDB call). Requires tmdb.API_KEY (set in config).
GENRE_NAME_TO_ID: dict[str, int] = {
    genre["name"]: genre["id"]
    for genre in tmdb.Genres().movie_list()["genres"]
}


# --- resolution -------------------------------------------------------------
@lru_cache(maxsize=512)
def _search_keyword_id(name: str) -> int | None:
    hits = _tmdb_call(lambda: tmdb.Search().keyword(query=name, page=1)["results"], [])
    return hits[0]["id"] if hits else None


@lru_cache(maxsize=512)
def _search_person_id(name: str) -> int | None:
    hits = _tmdb_call(lambda: tmdb.Search().person(query=name, page=1)["results"], [])
    return hits[0]["id"] if hits else None


def resolve_genre_ids(genres: list[str]) -> list[dict]:
    """Resolve genre names to {name, id} dicts; unresolved are omitted."""
    result = []
    for genre in genres:
        if genre_id := GENRE_NAME_TO_ID.get(genre):
            result.append({"name": genre, "id": genre_id})
    return result


def resolve_keywords(names: list[str]) -> list[dict]:
    """Resolve keyword strings to {name, id} dicts; unresolved are omitted."""
    result = []
    for name in names:
        kid = _search_keyword_id(name)
        if kid is not None:
            result.append({"name": name, "id": kid})
    return result


def resolve_persons(names: list[str]) -> list[dict]:
    """Resolve person names to {name, id} dicts; unresolved are omitted."""
    result = []
    for name in names:
        pid = _search_person_id(name)
        if pid is not None:
            result.append({"name": name, "id": pid})
    return result


# --- movie context ----------------------------------------------------------
def fetch_movie_context(movie_id: int) -> MovieContext | None:
    """Full context for a seed movie (info + keywords + credits + recs + similar)."""
    try:
        movie = tmdb.Movies(movie_id)
        info = movie.info()
        keywords = movie.keywords()["keywords"]
        credits = movie.credits()
        recommendations = [m["id"] for m in movie.recommendations()["results"]]
        similar = [m["id"] for m in movie.similar_movies()["results"]]

        return MovieContext(
            tmdb_id=movie_id,
            title=info.get("title") or "",
            overview=info.get("overview") or "",
            tagline=info.get("tagline") or "",
            genres=info.get("genres") or [],
            keywords=keywords or [],
            crew=[
                {"id": c["id"], "name": c["name"], "job": c["job"]}
                for c in credits["crew"]
                if c["job"] in IMPORTANT_CREW_JOBS
            ],
            cast=[{"id": c["id"], "name": c["name"]} for c in credits["cast"][:10]],
            countries=info.get("production_countries") or [],
            language=info.get("original_language") or "",
            runtime=info.get("runtime") or 0,
            release_date=info.get("release_date") or "",
            similar_movies=similar,
            recommendations=recommendations,
            popularity=info.get("popularity") or 0.0,
            vote_average=info.get("vote_average") or 0.0,
            vote_count=info.get("vote_count") or 0,
            budget=info.get("budget") or 0,
            revenue=info.get("revenue") or 0,
        )
    except Exception as e:
        print(f"[fetch_movie_context] Error for ID {movie_id}: {e}")
        return None


def fetch_ranking_context(movie_id: int) -> MovieContext | None:
    """Lightweight context for ranking: info + keywords only (2 TMDB calls)."""
    try:
        movie = tmdb.Movies(movie_id)
        info = movie.info()
        keywords = movie.keywords()["keywords"]

        return MovieContext(
            tmdb_id=movie_id,
            title=info.get("title") or "",
            overview=info.get("overview") or "",
            tagline=info.get("tagline") or "",
            genres=info.get("genres") or [],
            keywords=keywords or [],
            crew=[],
            cast=[],
            countries=info.get("production_countries") or [],
            language=info.get("original_language") or "",
            runtime=info.get("runtime") or 0,
            release_date=info.get("release_date") or "",
            similar_movies=[],
            recommendations=[],
            popularity=info.get("popularity") or 0.0,
            vote_average=info.get("vote_average") or 0.0,
            vote_count=info.get("vote_count") or 0,
            budget=info.get("budget") or 0,
            revenue=info.get("revenue") or 0,
        )
    except Exception as e:
        print(f"[fetch_ranking_context] Error for ID {movie_id}: {e}")
        return None


def format_context(ctx: MovieContext) -> str:
    crew_str = ", ".join(f"{c['name']} ({c['job']})" for c in ctx.crew)
    cast_str = ", ".join(c["name"] for c in ctx.cast[:3])
    genres_str = ", ".join(g["name"] for g in ctx.genres)
    keywords_str = ", ".join(k["name"] for k in ctx.keywords[:8])
    countries_str = ", ".join(c.get("iso_3166_1", "") for c in ctx.countries) or "?"
    return (
        f"Title: {ctx.title} ({ctx.release_date[:4]})\n"
        f"Countries: {countries_str} | Lang: {ctx.language}\n"
        f"Genres: {genres_str}\n"
        f"TMDB Keywords: {keywords_str}\n"
        f"Crew: {crew_str}\n"
        f"Cast: {cast_str}\n"
        f"Overview: {ctx.overview[:300]}\n"
    )


# --- discover ---------------------------------------------------------------
@tool
def discover_movies(
    with_keywords: str | None = None,
    with_genres: str | None = None,
    without_genres: str | None = None,
    with_original_language: str | None = None,
    with_origin_country: str | None = None,
    with_crew: str | None = None,
    with_cast: str | None = None,
    without_keywords: str | None = None,
    release_date_gte: str | None = None,
    release_date_lte: str | None = None,
    runtime_gte: int | str | None = None,
    runtime_lte: int | str | None = None,
    sort_by: str = "vote_average.desc",
    vote_count_gte: int | str = 50,
    vote_count_lte: int | str | None = None,
) -> list[dict]:
    """TMDB Discover. All ID-based fields must be pre-resolved.

    Joining IDs: "," = AND, "|" = OR. Returns up to 20 {id, title, overview, vote_average}.
    """
    def _as_int(v):
        if v is None or v == "":
            return None
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    params = {k: v for k, v in {
        "with_keywords": with_keywords,
        "with_genres": with_genres,
        "without_genres": without_genres,
        "with_original_language": with_original_language or None,
        "with_origin_country": with_origin_country or None,
        "with_crew": with_crew,
        "with_cast": with_cast,
        "without_keywords": without_keywords or None,
        "primary_release_date.gte": release_date_gte or None,
        "primary_release_date.lte": release_date_lte or None,
        "with_runtime.gte": _as_int(runtime_gte),
        "with_runtime.lte": _as_int(runtime_lte),
        "sort_by": sort_by,
        "vote_count.gte": _as_int(vote_count_gte),
        "vote_count.lte": _as_int(vote_count_lte),
    }.items() if v is not None}

    try:
        results = tmdb.Discover().movie(**params).get("results", [])
        return [
            {"id": r["id"], "title": r["title"],
             "overview": r["overview"], "vote_average": r["vote_average"]}
            for r in results
        ]
    except Exception as e:
        return [{"error": str(e)}]


# --- constraint checking ----------------------------------------------------
@lru_cache(maxsize=512)
def _cheap_movie_meta(movie_id: int):
    """(countries_tuple, original_language, release_date, runtime) for a TMDB id, cached."""
    try:
        info = tmdb.Movies(movie_id).info()
    except Exception as e:
        print(f"[cheap_meta] failed for {movie_id}: {e}")
        return None
    countries = tuple(c.get("iso_3166_1", "") for c in info.get("production_countries") or [])
    return countries, info.get("original_language") or "", info.get("release_date") or "", info.get("runtime") or 0


def _passes_constraints_by_id(movie_id: int, state) -> bool:
    allowed_countries = (state.get("with_origin_country") or "").replace(",", "|").split("|")
    allowed_countries = [c.strip() for c in allowed_countries if c.strip()]
    allowed_lang = state.get("with_original_language") or ""
    rdg = state.get("release_date_gte") or ""
    rdl = state.get("release_date_lte") or ""
    _df = state.get("discover_filters", {}) or {}
    runtime_gte = _df.get("runtime_gte")
    runtime_lte = _df.get("runtime_lte")

    if not (allowed_countries or allowed_lang or rdg or rdl or runtime_gte or runtime_lte):
        return True

    ctx = state.get("movie_contexts", {}).get(movie_id)
    if ctx is not None:
        ctx_countries = {c.get("iso_3166_1", "") for c in ctx.countries}
        lang = ctx.language
        rd = ctx.release_date
        runtime = ctx.runtime
    else:
        meta = _cheap_movie_meta(movie_id)
        if meta is None:
            return False
        ctx_countries = set(meta[0])
        lang = meta[1]
        rd = meta[2]
        runtime = meta[3]

    if allowed_countries and not (ctx_countries & set(allowed_countries)):
        return False
    if allowed_lang and lang and lang != allowed_lang:
        return False
    if rdg and rd and rd < rdg:
        return False
    if rdl and rd and rd > rdl:
        return False
    if runtime_gte and runtime and runtime < int(runtime_gte):
        return False
    if runtime_lte and runtime and runtime > int(runtime_lte):
        return False
    return True


# --- keyword population + WordNet canonicalization --------------------------
@lru_cache(maxsize=1024)
def _kw_population(kw_id: int, lang: str, country: str, rdg: str, rdl: str) -> int:
    """Solo TMDB total_results for a keyword under the HARD constraints only.

    Genre is intentionally excluded (soft, agent-derived). Measures how densely a
    keyword is tagged in the active context, so dead tags are dropped and the AND
    ladder is ordered by real productivity. Cached + key-based.
    """
    params = {k: v for k, v in {
        "with_keywords": str(kw_id),
        "with_original_language": lang or None,
        "with_origin_country": country or None,
        "primary_release_date.gte": rdg or None,
        "primary_release_date.lte": rdl or None,
        "vote_count.gte": 20,
    }.items() if v is not None}
    return _tmdb_call(lambda: tmdb.Discover().movie(**params).get("total_results", 0), 0)


try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)
from nltk.corpus import wordnet as _wn  # noqa: E402

_wn.synsets("cinema")        # force the lazy load ONCE here (single-threaded at import)
_wn_lock = threading.Lock()  # WordNet corpus reads are NOT thread-safe -> serialize them


@lru_cache(maxsize=2048)
def _wordnet_variants(word: str) -> tuple:
    """Derivational variants ONLY (lonely -> loneliness), not synonyms.

    Synonym lemmas drift across senses (strangers -> alien), so we follow only
    derivationally-related forms of the lemma that matches the word itself.
    """
    forms = {word}
    wl = word.lower()
    with _wn_lock:   # the parallel canonicalization fan-out would otherwise race in NLTK
        for syn in _wn.synsets(word)[:3]:
            for lem in syn.lemmas():
                if lem.name().replace("_", " ").lower() == wl:
                    for d in lem.derivationally_related_forms():
                        forms.add(d.name().replace("_", " "))   # lonely -> loneliness
    return tuple(forms)


def _canonicalize_keyword(word: str, lang: str, ctry: str, rdg: str, rdl: str) -> str:
    """Map a keyword to its best-populated WordNet variant that exists as a TMDB keyword.

    The model often emits a thin surface form ("lonely", pop 10) when a richer sibling
    is tagged on far more films ("loneliness", pop 164). Only morphology - cross-concept
    synonymy (lonely -> isolation) is left to embeddings.
    """
    best_word, best_pop = word, -1
    kid = _search_keyword_id(word)
    if kid is not None:
        best_pop = _kw_population(kid, lang, ctry, rdg, rdl)
    for v in _wordnet_variants(word):
        if v == word:
            continue
        vid = _search_keyword_id(v)
        if vid is None:
            continue
        p = _kw_population(vid, lang, ctry, rdg, rdl)
        if p > best_pop:
            best_word, best_pop = v, p
    if best_word != word:
        print(f"   kw canon: {word} -> {best_word} (pop {best_pop})")
    return best_word

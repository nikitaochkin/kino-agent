"""Graph state and the movie-context value object."""
from typing_extensions import TypedDict

from pydantic import BaseModel


class MovieContext(BaseModel):
    tmdb_id: int
    title: str
    overview: str
    tagline: str
    genres: list[dict]
    keywords: list[dict]
    crew: list[dict]
    cast: list[dict]
    countries: list[dict]
    language: str
    runtime: int
    release_date: str
    similar_movies: list[int]
    recommendations: list[int]
    popularity: float
    vote_average: float
    budget: int
    revenue: int
    vote_count: int


class InputState(TypedDict):
    """What the user supplies (and all Studio shows in the input form)."""
    query_text: str


class OutputState(TypedDict):
    """Clean result surfaced by the graph: readable picks, not raw TMDB ids."""
    recommendations: list  # [{title, year, tmdb_id, score, reason}]


class RecommendationState(TypedDict):
    query_text: str                          # user's original query
    seed_titles: list[str]                   # seed movie titles extracted from the query
    seed_movie_ids: list[int]                # resolved TMDB IDs for the seeds
    constraints: dict                        # extra preferences (e.g. popularity)
    release_date_gte: str                    # min release date "YYYY-MM-DD"
    release_date_lte: str                    # max release date "YYYY-MM-DD"
    with_original_language: str              # ISO 639-1 language constraint
    with_origin_country: str                 # ISO 3166-1 country constraint
    movie_contexts: dict[int, MovieContext]  # cache of fetched movie contexts
    watched_movies: list[int]                # TMDB IDs the user has already seen
    movie_ratings: dict[int, float]          # TMDB ID -> user rating
    analysis: dict                           # consolidated single-call analysis output
    relax_level: int                         # soft-constraint relaxation tier (retry loop)
    discover_filters: dict                   # filters handed to discover
    final_recommendations: list              # final ranked TMDB IDs
    recommendations: list                    # readable picks [{title, year, tmdb_id, score, reason}]
    taste_mode: bool                         # discover fell back to pure-taste recs -> rank by taste

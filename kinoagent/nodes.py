"""All graph nodes: data load, parsing, analysis, filter build, discover, ranking."""
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from pprint import pprint

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import interrupt
from pydantic import BaseModel, Field
from trustcall import create_extractor

from .config import llm_strong, tmdb
from .schemas import AnalysisOutput, ParsedQuery
from .state import RecommendationState
from .tmdb_tools import (
    _canonicalize_keyword,
    _kw_population,
    _passes_constraints_by_id,
    discover_movies,
    fetch_movie_context,
    fetch_ranking_context,
    format_context,
    resolve_genre_ids,
    resolve_keywords,
    resolve_persons,
)

# --- discover retry loop tunables -------------------------------------------
MIN_CANDIDATES = 15     # below this, relax a soft knob and retry discover
MAX_RELAX = 2           # hard stop on relaxation tiers
TASTE_SEED_COUNT = 8    # pure-taste mode: how many top-rated films to seed from

DB_PATH = Path("user_data.db")


# --- user data --------------------------------------------------------------
def load_user_data_node(state: RecommendationState) -> dict:
    """Load watched movies (and ratings) from user_data.db into state."""
    if not DB_PATH.exists():
        print(f"[load_user_data] DB not found at {DB_PATH.resolve()}, skipping")
        return {"watched_movies": [], "movie_ratings": {}}

    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()
    watched_rows = cur.execute("""
        SELECT m.tmdb_id FROM watched w
        JOIN movies m ON w.movie_id = m.id
        WHERE m.tmdb_id IS NOT NULL
    """).fetchall()
    watched = [r[0] for r in watched_rows]
    rating_rows = cur.execute("""
        SELECT m.tmdb_id, r.rating FROM ratings r
        JOIN movies m ON r.movie_id = m.id
        WHERE m.tmdb_id IS NOT NULL
    """).fetchall()
    ratings = {r[0]: r[1] for r in rating_rows}
    conn.close()
    print(f"[load_user_data] {len(watched)} watched, {len(ratings)} rated")
    return {"watched_movies": watched, "movie_ratings": ratings}


# --- parsing ----------------------------------------------------------------
def query_parser_agent(state: RecommendationState) -> dict:
    """Parse the user's query to extract seed movie titles and any constraints."""
    parser_prompt = """
You MUST call the ParsedQuery function with your extracted fields. Do not reply with plain text.

Extract structured movie recommendation query information.

Rules:
- Only extract information explicitly mentioned by the user.
- Use canonical English movie titles when obvious and reliable.
- Do not transliterate non-English titles if their canonical English equivalents are unclear.
- Do not invent constraints.
- Use short and semantically clear constraint values.
- For release_date_gte / release_date_lte: convert time period descriptions to YYYY-MM-DD.
  Example: "early 2000s" -> release_date_gte="2000-01-01", release_date_lte="2005-12-31"
  Example: "modern times" -> release_date_gte="2010-01-01"
  Example: "recent" -> release_date_gte="2015-01-01"
- For with_original_language: convert language names to ISO 639-1 codes.
  Example: "Spanish" -> "es", "Russian" -> "ru", "French" -> "fr"
- For with_origin_country: convert country names to ISO 3166-1 alpha-2 codes, separated by commas for AND or pipes for OR.
  If user says a region, expand to country codes joined by |:
  "European"       -> "FR|DE|IT|ES|PL|CZ|RO|HU|BE|NL|AT|SE|DK|NO|FI|PT|GR"
  "Scandinavian"   -> "SE|NO|DK|FI"
  "Latin American" -> "MX|BR|AR|CO|CL|PE"
  "East Asian"     -> "JP|KR|CN|TW|HK"
  "Middle Eastern" -> "IR|IL|TR|LB|EG"
  Example: "American" -> "US", "French or German" -> "FR|DE", "Japanese and South Korean" -> "JP,KR"

Examples:

User: something like Perfect Days but from early 2000s
Result:
  seed_titles=["Perfect Days"]
  constraints={"time_period": "early 2000s"}
  release_date_gte="2000-01-01"
  release_date_lte="2005-12-31"

User: I want something like The Matrix but in Russian and filmed in Russia or Soviet Union
Result:
  seed_titles=["The Matrix"]
  constraints={"language": "Russian"}
  with_original_language="ru"
  with_origin_country="RU|SU"

User: Something like lost highway and donnie darko but less popular
Result:
  seed_titles=["Lost Highway", "Donnie Darko"]
  constraints={"popularity": "lower"}
"""
    extractor = create_extractor(llm_strong, tools=[ParsedQuery], tool_choice=ParsedQuery.__name__)
    response = extractor.invoke({
        "messages": [
            SystemMessage(content=parser_prompt),
            HumanMessage(content=state["query_text"]),
        ]
    })
    parsed = response["responses"][0]
    return {
        "seed_titles": parsed.seed_titles,
        "constraints": parsed.constraints,
        "release_date_gte": parsed.release_date_gte,
        "release_date_lte": parsed.release_date_lte,
        "with_original_language": parsed.with_original_language,
        "with_origin_country": parsed.with_origin_country,
    }


def seed_resolver_node(state: RecommendationState) -> dict:
    """Resolve seed titles to TMDB IDs, asking the user when a title is ambiguous.

    If a title matches 2+ well-known films from different years (e.g. an original and
    a remake), the graph pauses via interrupt() and asks which one. The client (Studio)
    resumes with the chosen tmdb_id. NOTE: on resume the node re-executes from the top,
    so each ambiguous title is answered in order.
    """
    resolved_ids = []
    search = tmdb.Search()
    for title in state["seed_titles"]:
        try:
            results = search.movie(query=title, page=1, include_adult=True, language="en-US")["results"]
        except Exception:
            results = []
        if not results:
            print(f"[seed_resolver] Could not resolve: {title}")
            continue

        # Default to the most-voted match, NOT results[0]: TMDB search ranks an exact
        # primary-title hit (e.g. a no-name doc) above a famous film whose primary title
        # is in another language ("Until the End of the World" -> Wenders' "Bis ans Ende
        # der Welt"). vote_count is a robust "is this the known film" signal.
        best = max(results, key=lambda r: r.get("vote_count", 0))
        known = sorted([r for r in results if r.get("vote_count", 0) >= 50],
                       key=lambda r: r.get("vote_count", 0), reverse=True)[:5]
        years = {(r.get("release_date") or "")[:4] for r in known if r.get("release_date")}
        if len(known) >= 2 and len(years) >= 2:
            listing = "\n".join(
                f"{i + 1}. {r['title']} ({(r.get('release_date') or '')[:4]})"
                for i, r in enumerate(known)
            )
            choice = interrupt(
                f"'{title}' matches several films - which did you mean?\n{listing}\n\n"
                f"Reply with the number (1-{len(known)})."
            )
            try:
                resolved_ids.append(known[int(str(choice).strip()) - 1]["id"])
            except (ValueError, IndexError, TypeError):
                resolved_ids.append(best["id"])   # fallback if reply is not a valid number
        else:
            print(f"[seed_resolver] '{title}' -> {best['title']} "
                  f"({(best.get('release_date') or '')[:4]}, votes {best.get('vote_count', 0)})")
            resolved_ids.append(best["id"])
    return {"seed_movie_ids": resolved_ids}


def movie_enrichment_node(state: RecommendationState) -> dict:
    """Enrich seed movies with TMDB metadata for downstream context."""
    updated_contexts = dict(state.get("movie_contexts", {}))
    missing = [m for m in state["seed_movie_ids"] if m not in updated_contexts]
    if missing:
        with ThreadPoolExecutor(max_workers=8) as ex:
            for mid, ctx in zip(missing, ex.map(fetch_movie_context, missing)):
                if ctx:
                    updated_contexts[mid] = ctx
    return {"movie_contexts": updated_contexts}


# --- analysis ---------------------------------------------------------------
GLOBAL_RULES = """GLOBAL RULES (apply to every keyword list):
- Return AT MOST 5 keywords per axis. Pick the ones most strongly tied to the USER QUERY
  (not just the seed movies). Order them from most to least important.
- The example keywords for each axis are ILLUSTRATIVE ONLY. NEVER output them verbatim.
  Derive keywords from THESE seed films and THIS query. If fewer than 5 genuinely fit,
  return fewer (even an empty list) - never pad with generic examples.
- When a seed film's TMDB Keywords (shown in its context) fit the query, REUSE their
  exact wording - they are real, well-populated tags. Prefer "loneliness" over
  "lonely", "unlikely friendship" over "friendship". Paraphrasing into a thinner or
  broader word loses the tag.
- without_keywords: ONLY include a keyword the user EXPLICITLY excluded ("not X", "no X",
  "without X", "avoid X"). Otherwise return an empty list.

Output only discover filters. Do not recommend specific movies."""

ANALYSIS_PROMPT = """You MUST call the AnalysisOutput function. Do not reply with plain text.
Analyse the seed films across FIVE axes and fill the matching fields.

[MOOD] -> mood_keywords
  Emotional tone / atmosphere only: lonely, melancholic, warm, contemplative, tense,
  surreal, nostalgic, bleak, euphoric, claustrophobic...
[THEME] -> theme_keywords (+ with_genres)
  Plot topics / subject matter: coming-of-age, grief, found family, class struggle,
  redemption, identity, immigration, revenge...
[VISUAL] -> visual_keywords (+ with_crew)
  Aesthetics / camera: neo-noir, handheld camera, long takes, naturalistic,
  desaturated palette, symmetrical framing, grainy film...
  ALWAYS add the seed films' Director of Photography (and visually distinctive
  directors) to with_crew - this is the strongest visual signal.
[PACING] -> pacing_keywords (+ runtime_gte / runtime_lte)
  Narrative speed / structure: slow-burn, episodic, non-linear narrative,
  single-day story, fragmented narrative, mystery-box...
  Set runtime ONLY if the user explicitly named a length ("around 90 minutes",
  "no longer than 2 hours"); be generous (at least +/-20 min). NEVER infer runtime
  from the seed films - long seeds must not silently exclude similar long films.
[CHARACTER] -> character_keywords (+ with_cast)
  Character types / dynamics: anti-hero, strong female lead, found family,
  mentor-student, ensemble cast, morally ambiguous, outsider, working class...
  Add explicitly-named actors to with_cast. Do NOT lift actors from the seed films'
  casts. For people the query only *evokes* (not names), see EMBLEMATIC PEOPLE below.

EMBLEMATIC PEOPLE (use sparingly, high-confidence only):
  When the query evokes a recognisable sensibility rather than a named person, you MAY
  nominate the emblematic figure(s) behind it - even if the query names nobody.
  Put directors in with_crew (strongest signal), actors in with_cast.
  Examples: "dreamlike and surreal" -> David Lynch; "deadpan, offbeat indie" ->
  Jim Jarmusch or Wes Anderson; "gritty American independent" -> the Safdie brothers
  or Chloe Sevigny; "slow, contemplative art cinema" -> Apichatpong Weerasethakul.
  Only do this when the mapping is strong and well known; otherwise leave them empty.
  TMDB tags people far more densely than mood keywords, so a well-chosen name often
  retrieves better than abstract adjectives.

AXIS PRIORITY -> axis_priority
  Order the five axis names by how central each is to THIS query, most relevant first.
  A query about food/family leads with theme_keywords; a pure-vibe query ("dreamlike,
  surreal") leads with mood_keywords or visual_keywords; a look-driven query leads with
  visual_keywords. Use exactly these names: mood_keywords, theme_keywords,
  visual_keywords, pacing_keywords, character_keywords.

VALID GENRE NAMES - for with_genres / without_genres use EXACTLY one of these,
verbatim and case-sensitive (no abbreviations like "Sci-Fi", no lowercase):
Action, Adventure, Animation, Comedy, Crime, Documentary, Drama, Family, Fantasy,
History, Horror, Music, Mystery, Romance, Science Fiction, TV Movie, Thriller, War, Western

EXCLUSIONS (without_keywords / without_genres): leave EMPTY unless the user EXPLICITLY
excluded something. Do NOT infer exclusions from mood or tone - a slow, calm or
melancholic request does NOT mean exclude Action/Thriller/Crime (many moody, slow
films are tagged with those). NEVER exclude a genre the seed films themselves belong to.

Language, country and dates come from pre-resolved constraints - do NOT set them here.
"""


def analysis_node(state: RecommendationState) -> dict:
    movie_contexts = state["movie_contexts"]
    query = state["query_text"]
    constraints = state["constraints"]
    formatted = "\n\n".join(format_context(ctx) for ctx in movie_contexts.values())

    messages = [
        SystemMessage(content=ANALYSIS_PROMPT + "\n\n" + GLOBAL_RULES),
        HumanMessage(content=f"""
User query:
{query}

Constraints extracted from the query:
{constraints}

Pre-resolved constraints (use these directly, do NOT override or reinterpret):
- release_date_gte: {state.get('release_date_gte', '')}
- release_date_lte: {state.get('release_date_lte', '')}
- with_original_language: {state.get('with_original_language', '')}
- with_origin_country: {state.get('with_origin_country', '')}

Seed movie contexts:
{formatted}
""")
    ]
    extractor = create_extractor(llm_strong, tools=[AnalysisOutput], tool_choice=AnalysisOutput.__name__)
    response = extractor.invoke({"messages": messages})
    analysis = response["responses"][0]
    return {"analysis": analysis.model_dump()}


# --- build_filters ----------------------------------------------------------
def build_filters_node(state: RecommendationState) -> dict:
    """Build the discover filters from the single analysis output.

    Round-robin keywords across the five axes (by the query's axis priority), dedupe
    genres with a seed-genre floor, apply the cast guard, attach pre-resolved constraints.
    """
    a = state["analysis"]

    KW_AXES_DEFAULT = ["theme_keywords", "mood_keywords", "visual_keywords",
                       "pacing_keywords", "character_keywords"]

    def _norm_axis(name: str) -> str:
        name = (name or "").strip().lower().replace(" ", "_").replace("-", "_")
        return name if name.endswith("_keywords") else name + "_keywords"

    _priority = [ax for ax in (_norm_axis(x) for x in (a.get("axis_priority") or []))
                 if ax in KW_AXES_DEFAULT]
    KW_AXES = _priority + [ax for ax in KW_AXES_DEFAULT if ax not in _priority]
    print(f"[build_filters] axis order: {KW_AXES}")

    def _round_robin(lists: list[list]) -> list:
        out, seen = [], set()
        for col in range(max((len(l) for l in lists), default=0)):
            for l in lists:
                if col < len(l):
                    kw = l[col]
                    if kw and kw not in seen:
                        seen.add(kw)
                        out.append(kw)
        return out

    with_kw = _round_robin([list(a.get(f) or []) for f in KW_AXES])
    without_kw = list(a.get("without_keywords") or [])

    seen_genres: dict[str, str] = {}
    for genre in (a.get("with_genres") or []):
        if genre and genre not in seen_genres:
            seen_genres[genre] = "analysis"

    # Genre floor: fall back to the seed movies' own genres if the model proposed none.
    if not seen_genres:
        for ctx in state.get("movie_contexts", {}).values():
            for g in ctx.genres:
                name = g.get("name")
                if name and name not in seen_genres:
                    seen_genres[name] = "seed"

    without_genres = list(a.get("without_genres") or [])
    runtime_gte = a.get("runtime_gte")
    runtime_lte = a.get("runtime_lte")
    crew = list(a.get("with_crew") or [])

    # Cast guard: keep an actor if the user named them, or if it's an emblematic pick
    # not present in any seed cast (blocks the recurring "lift Gosling from seed" bug).
    _qt = (state.get("query_text") or "").lower()
    _seed_cast = {
        (c.get("name") or "").lower()
        for ctx in state.get("movie_contexts", {}).values()
        for c in ctx.cast
    }
    cast = [
        name for name in (a.get("with_cast") or [])
        if name and (
            name.lower() in _qt
            or name.split()[-1].lower() in _qt
            or name.lower() not in _seed_cast
        )
    ]

    filters = {
        "with_keywords": with_kw,
        "without_keywords": without_kw,
        "with_genres": list(seen_genres.keys()),
        "without_genres": without_genres,
        "with_crew": crew,
        "with_cast": cast,
        "with_original_language": state.get("with_original_language", ""),
        "with_origin_country": state.get("with_origin_country", ""),
        "release_date_gte": state.get("release_date_gte", ""),
        "release_date_lte": state.get("release_date_lte", ""),
        "runtime_gte": runtime_gte,
        "runtime_lte": runtime_lte,
    }

    constraints = state.get("constraints", {})
    seed_vote_counts = [ctx.vote_count for ctx in state["movie_contexts"].values() if ctx.vote_count]
    median_votes = (sorted(seed_vote_counts)[len(seed_vote_counts) // 2] if seed_vote_counts else 1000)
    if constraints.get("popularity") == "lower":
        filters["vote_count_lte"] = int(median_votes * 0.7)
    elif constraints.get("popularity") == "higher":
        filters["vote_count_gte"] = int(median_votes * 1.5)

    print("\n[build_filters] Final filters:")
    pprint(filters)
    return {"discover_filters": filters}


# --- discover ---------------------------------------------------------------
def discover_agent(state: RecommendationState) -> dict:
    """Build candidates: seed pool + crew search + deterministic keyword discovery."""
    filters = state["discover_filters"]
    sort_by = filters.get("sort_by", "vote_average.desc")
    vote_count_gte = filters.get("vote_count_gte", 20)
    vote_count_lte = filters.get("vote_count_lte", None)

    # Soft-constraint relaxation, driven by the graph retry loop. Hard constraints
    # (country / language / explicit dates) are never touched here.
    relax_level = state.get("relax_level", 0)
    if relax_level >= 1:
        vote_count_gte = 5
    if relax_level >= 2:
        vote_count_gte = 0
        filters = {**filters, "runtime_gte": None, "runtime_lte": None}
    if relax_level:
        print(f"[discover] relaxation level {relax_level}: vote_count_gte={vote_count_gte}"
              + (", runtime dropped" if relax_level >= 2 else ""))

    watched = set(state.get("watched_movies", []))
    seeds = set(state.get("seed_movie_ids", []))
    exclude = watched | seeds

    _hard = (filters.get("with_original_language") or "", filters.get("with_origin_country") or "",
             filters.get("release_date_gte") or "", filters.get("release_date_lte") or "")
    raw_kw = filters.get("with_keywords", [])
    with ThreadPoolExecutor(max_workers=8) as ex:
        canon_kw = list(ex.map(lambda w: _canonicalize_keyword(w, *_hard), raw_kw))
    with_keywords_ids = resolve_keywords(canon_kw)
    without_keywords_ids = resolve_keywords(filters.get("without_keywords", []))
    person_ids = resolve_persons(filters.get("with_crew", []) + filters.get("with_cast", []))
    genre_ids = resolve_genre_ids(filters.get("with_genres", []))
    without_genre_ids = resolve_genre_ids(filters.get("without_genres", []))

    print(f"🔧 Pre-resolved keywords: {with_keywords_ids}")
    print(f"🔧 Pre-resolved without keywords:  {without_keywords_ids}")
    print(f"🔧 Pre-resolved persons:  {person_ids}")
    print(f"🔧 Pre-resolved genres:   {genre_ids}")
    print(f"🔧 Pre-resolved without genres: {without_genre_ids}")

    discover_found: list[int] = []

    seed_contexts = state.get("movie_contexts", {})
    seed_ids_used = [sid for sid in state.get("seed_movie_ids", []) if sid in seed_contexts]
    per_seed_sets: list[set[int]] = []
    per_seed_lists: list[list[int]] = []
    seed_source: dict[int, str] = {}
    for sid in seed_ids_used:
        ctx = seed_contexts[sid]
        for mid in ctx.recommendations:
            if mid not in exclude:
                seed_source.setdefault(mid, "rec")
        for mid in ctx.similar_movies:
            if mid not in exclude:
                seed_source.setdefault(mid, "sim")
        ids = [mid for mid in ctx.recommendations + ctx.similar_movies if mid not in exclude]
        per_seed_sets.append(set(ids))
        per_seed_lists.append(ids)

    tmdb_pool: list[int] = []
    if len(per_seed_sets) >= 2:
        intersection = set.intersection(*per_seed_sets)
        for mid in per_seed_lists[0]:
            if mid in intersection and mid not in tmdb_pool:
                tmdb_pool.append(mid)
        print(f"📚 TMDB intersection across {len(seed_ids_used)} seeds: {len(tmdb_pool)} candidates")
    for ids in per_seed_lists:
        for mid in ids:
            if mid not in tmdb_pool:
                tmdb_pool.append(mid)
    print(f"📚 TMDB seed pool (intersection-first, then union): {len(tmdb_pool)} candidates")

    tmdb_seeded: list[int] = []
    TMDB_LOOKUP_CAP = 30
    for mid in tmdb_pool[:TMDB_LOOKUP_CAP]:
        if _passes_constraints_by_id(mid, state):
            tmdb_seeded.append(mid)
    print(f"📚 TMDB seed after hard-constraint filter: {len(tmdb_seeded)} / {min(len(tmdb_pool), TMDB_LOOKUP_CAP)} checked")
    _rec_n = sum(1 for mid in tmdb_seeded if seed_source.get(mid) == "rec")
    _sim_n = len(tmdb_seeded) - _rec_n
    print(f"   source split: {_rec_n} [rec] + {_sim_n} [sim]  (tmdb bucket takes the first 16, in this order)")
    print("   " + " ".join(f"{mid}[{seed_source.get(mid, chr(63))}]" for mid in tmdb_seeded))

    if person_ids:
        crew_ids_str = "|".join(str(p["id"]) for p in person_ids)
        crew_params = {
            "with_crew": crew_ids_str,
            "sort_by": sort_by,
            "vote_count_gte": vote_count_gte,
            "vote_count_lte": vote_count_lte,
            "with_original_language": filters.get("with_original_language") or None,
            "with_origin_country": filters.get("with_origin_country") or None,
            "release_date_gte": filters.get("release_date_gte") or None,
            "release_date_lte": filters.get("release_date_lte") or None,
        }
        crew_params = {k: v for k, v in crew_params.items() if v is not None}
        crew_result = discover_movies.invoke(crew_params)
        crew_found: list[int] = []
        if isinstance(crew_result, list) and crew_result and "id" in crew_result[0]:
            kept = [item for item in crew_result if item["id"] not in exclude]
            kept_filtered = [item for item in kept if _passes_constraints_by_id(item["id"], state)]
            dropped_n = len(kept) - len(kept_filtered)
            crew_found = [item["id"] for item in kept_filtered]
            print(f"🎬 Crew search: {crew_params} → {len(crew_found)} movies (post-filter dropped {dropped_n})")
            for item in kept_filtered[:10]:
                print(f"   {item['title']} ({item['id']}) ⭐{item['vote_average']}")
        else:
            print(f"🎬 Crew search returned no results: {crew_result}")
    else:
        crew_found = []

    # --- deterministic keyword discovery ---
    # Probe each keyword's solo population under hard constraints, drop dead tags, keep
    # the survivors in (query-priority) order, then shrink AND 4->1, OR, then no-genre.
    POP_FLOOR = 1
    _lang = filters.get("with_original_language") or ""
    _ctry = filters.get("with_origin_country") or ""
    _rdg = filters.get("release_date_gte") or ""
    _rdl = filters.get("release_date_lte") or ""

    with ThreadPoolExecutor(max_workers=8) as ex:
        pops = list(ex.map(lambda k: _kw_population(k["id"], _lang, _ctry, _rdg, _rdl), with_keywords_ids))
    scored = list(zip(with_keywords_ids, pops))
    live = [(k, n) for k, n in scored if n >= POP_FLOOR]
    for k, n in scored:
        tag = "" if n >= POP_FLOOR else "  (dropped)"
        print(f"   kw pop: {k['name']} ({k['id']}) -> {n}{tag}")
    kw_ids = [str(k["id"]) for k, _ in live]
    base = {
        "with_genres": "|".join(str(g["id"]) for g in genre_ids) or None,
        "without_genres": "|".join(str(g["id"]) for g in without_genre_ids) or None,
        "without_keywords": ",".join(str(k["id"]) for k in without_keywords_ids) or None,
        "with_original_language": filters.get("with_original_language") or None,
        "with_origin_country": filters.get("with_origin_country") or None,
        "release_date_gte": filters.get("release_date_gte") or None,
        "release_date_lte": filters.get("release_date_lte") or None,
        "runtime_gte": filters.get("runtime_gte"),
        "runtime_lte": filters.get("runtime_lte"),
        "sort_by": sort_by,
        "vote_count_gte": vote_count_gte,
        "vote_count_lte": vote_count_lte,
    }

    top = kw_ids[:4]

    def _kw_steps(include_or=True):
        steps = [(",".join(top[:n]), f"AND top-{n}") for n in range(len(top), 0, -1)]
        if include_or and len(top) >= 2:
            steps.append(("|".join(top), "OR top"))
        return steps

    ladder = ([(kw, lbl, True) for kw, lbl in _kw_steps()]
              + [(kw, lbl + " / no genre", False) for kw, lbl in _kw_steps(include_or=False)])

    DISCOVER_TARGET = 12
    for kw, label, use_genre in ladder:
        step = {**base, "with_keywords": kw}
        if not use_genre:
            step["with_genres"] = None
        params = {k: v for k, v in step.items() if v is not None}
        res = discover_movies.invoke(params)
        if isinstance(res, list) and res and "id" in res[0]:
            fresh = [r["id"] for r in res if r["id"] not in exclude and r["id"] not in discover_found]
            discover_found.extend(fresh)
            print(f"\U0001f50d keywords [{kw}] ({label}) → {len(fresh)} new (total {len(set(discover_found))})")
            for r in res[:8]:
                if r.get("id") not in exclude:
                    print(f"   {r['title']} ({r['id']}) ⭐{r['vote_average']}")
        else:
            print(f"\U0001f50d keywords [{kw}] ({label}) → 0")
        if len(set(discover_found)) >= DISCOVER_TARGET:
            print(f"[discover_agent] keyword discovery reached {len(set(discover_found))}, stopping")
            break

    # Deterministic safety net: genre + country + date (anchored by the seed-genre floor).
    if len(set(discover_found)) < 8 and genre_ids:
        genre_or = "|".join(str(g["id"]) for g in genre_ids)
        base = {
            "with_genres": genre_or,
            "sort_by": sort_by,
            "vote_count_gte": vote_count_gte,
            "with_original_language": filters.get("with_original_language") or None,
            "with_origin_country": filters.get("with_origin_country") or None,
            "release_date_gte": filters.get("release_date_gte") or None,
            "release_date_lte": filters.get("release_date_lte") or None,
        }
        attempts = [{"runtime_gte": filters.get("runtime_gte"), "runtime_lte": filters.get("runtime_lte")}]
        if filters.get("runtime_gte") or filters.get("runtime_lte"):
            attempts.append({})   # retry without the runtime window, but only if there was one
        for extra in attempts:
            params = {k: v for k, v in {**base, **extra}.items() if v is not None}
            res = discover_movies.invoke(params)
            if isinstance(res, list) and res and "id" in res[0]:
                added = [r["id"] for r in res if r["id"] not in exclude]
                discover_found.extend(added)
                print(f"\U0001fa82 Fallback discover {params} → {len(added)} movies")
                for r in res[:10]:
                    if "id" in r:
                        print(f"   {r['title']} ({r['id']}) ⭐{r['vote_average']}")
            if len(set(discover_found)) >= 8:
                break

    # Pure-taste fallback: the query gave nothing to retrieve from (no seeds, no usable
    # keywords, no genre, no crew) and set no constraints -> "just recommend by my taste".
    # Pull the recommendations of the user's top-rated films directly; ranking sorts by
    # taste. A thematic query (e.g. "something funny") produces keywords -> discover_found
    # is non-empty -> this never fires.
    taste_mode = False
    has_constraints = bool(
        (state.get("constraints") or {})
        or filters.get("with_origin_country") or filters.get("with_original_language")
        or filters.get("release_date_gte") or filters.get("release_date_lte")
    )
    if not tmdb_seeded and not crew_found and not discover_found and not has_constraints:
        ratings = state.get("movie_ratings") or {}
        if ratings:
            top = sorted(ratings.items(), key=lambda kv: kv[1], reverse=True)[:TASTE_SEED_COUNT]
            for fav_id, _ in top:
                try:
                    mv = tmdb.Movies(fav_id)
                    pool = ([m["id"] for m in mv.recommendations()["results"]]
                            + [m["id"] for m in mv.similar_movies()["results"]])
                except Exception:
                    pool = []
                for mid in pool:
                    if mid not in exclude and mid not in discover_found:
                        discover_found.append(mid)
            taste_mode = True
            print(f"🎲 pure-taste fallback: {len(discover_found)} candidates from top-{len(top)} rated (recs + similar)")

    TARGET = 34
    CAPS = {"tmdb": 16, "crew": 6, "discover": None}
    buckets = {"tmdb": tmdb_seeded, "crew": crew_found, "discover": discover_found}
    seen: set[int] = set()
    final: list[int] = []
    for name in ("tmdb", "crew", "discover"):
        cap = CAPS[name]
        remaining = TARGET - len(final)
        budget = remaining if cap is None else min(cap, remaining)
        taken = 0
        for mid in buckets[name]:
            if taken >= budget:
                break
            if mid not in seen:
                seen.add(mid)
                final.append(mid)
                taken += 1
        print(f"[discover_agent] bucket {name}: took {taken}/{len(buckets[name])} (budget {budget})")

    print(f"\n[discover_agent] Final {len(final)} candidates")
    return {"final_recommendations": final, "taste_mode": taste_mode}


# --- relaxation loop --------------------------------------------------------
def relax_constraints_node(state: RecommendationState) -> dict:
    level = state.get("relax_level", 0) + 1
    print(f"[relax] only {len(state.get('final_recommendations', []))} candidates "
          f"-> relaxation level {level}")
    return {"relax_level": level}


def route_after_discover(state: RecommendationState) -> str:
    n = len(state.get("final_recommendations", []))
    if n >= MIN_CANDIDATES or state.get("relax_level", 0) >= MAX_RELAX:
        return "ranking"
    return "relax_constraints"


# --- ranking + taste --------------------------------------------------------
RATING_MIDPOINT = 3.0      # Letterboxd 0.5-5: above = liked, below = disliked
TASTE_WEIGHT = 0.3         # final = (1 - w) * relevance + w * taste
TASTE_PROFILE_CAP = 200    # max rated films to profile


@lru_cache(maxsize=512)
def _taste_features(movie_id: int):
    """((genre_id, name)...), ((keyword_id, name)...) for a movie, cached."""
    try:
        mv = tmdb.Movies(movie_id)
        info = mv.info()
        kws = mv.keywords()["keywords"]
    except Exception:
        return (), ()
    return (
        tuple((g["id"], g["name"]) for g in info.get("genres") or []),
        tuple((k["id"], k["name"]) for k in kws or []),
    )


def build_taste_profile(movie_ratings: dict) -> dict:
    """Weighted genre/keyword profile from user ratings (liked -> +, disliked -> -)."""
    if not movie_ratings:
        return {}
    items = sorted(movie_ratings.items(),
                   key=lambda kv: abs(kv[1] - RATING_MIDPOINT), reverse=True)[:TASTE_PROFILE_CAP]
    genres: dict[int, float] = {}
    keywords: dict[int, float] = {}
    names: dict[int, str] = {}
    for mid, rating in items:
        w = rating - RATING_MIDPOINT
        if w == 0:
            continue
        gfeat, kfeat = _taste_features(mid)
        for gid, gname in gfeat:
            genres[gid] = genres.get(gid, 0.0) + w
            names[gid] = gname
        for kid, kname in kfeat:
            keywords[kid] = keywords.get(kid, 0.0) + w
            names[kid] = kname
    return {"genres": genres, "keywords": keywords, "names": names}


def _taste_score(ctx, profile: dict) -> float:
    """Keyword-only affinity, length-normalised so heavily-tagged films don't dominate.

    A raw sum rewards films with many keywords (more chances to overlap the profile);
    dividing by sqrt(#keywords) tames that volume bias without flipping the advantage to
    sparsely-tagged films. Genres aren't scored (too volume-driven to discriminate).
    """
    if not profile or not ctx.keywords:
        return 0.0
    raw = sum(profile["keywords"].get(kk["id"], 0.0) for kk in ctx.keywords)
    return raw / (len(ctx.keywords) ** 0.5)


def print_taste_profile(profile: dict, top: int = 12) -> None:
    if not profile:
        print("[taste] empty profile (no ratings / no DB)")
        return
    nm = profile.get("names", {})

    def fmt(items):
        return ", ".join(f"{nm.get(i, i)} ({w:+.1f})" for i, w in items)

    gen = sorted(profile["genres"].items(), key=lambda kv: kv[1], reverse=True)
    kw = sorted(profile["keywords"].items(), key=lambda kv: kv[1], reverse=True)
    liked = [x for x in kw if x[1] > 0][:top]
    disliked = [x for x in kw if x[1] < 0][-top:]
    print("[taste] genres:", fmt(gen))
    print("[taste] liked keywords:", fmt(liked))
    if disliked:
        print("[taste] disliked keywords:", fmt(disliked))


def ranking_agent(state: RecommendationState) -> dict:
    """Re-rank discovered movies by relevance to the query, blended with taste."""
    candidate_ids = state["final_recommendations"]
    if not candidate_ids:
        return {"recommendations": []}

    contexts = dict(state.get("movie_contexts", {}))
    missing = [mid for mid in candidate_ids if mid not in contexts]
    if missing:
        with ThreadPoolExecutor(max_workers=12) as ex:
            for mid, ctx in zip(missing, ex.map(fetch_ranking_context, missing)):
                if ctx:
                    contexts[mid] = ctx

    # Pure-taste mode: no query to be "relevant" to (the LLM would score everything 0,
    # "no reference"). Rank purely by the taste profile and skip the LLM call entirely.
    if state.get("taste_mode"):
        profile = build_taste_profile(state.get("movie_ratings", {}))
        scores = {mid: _taste_score(contexts[mid], profile)
                  for mid in candidate_ids if mid in contexts}
        ranked = sorted(scores, key=scores.get, reverse=True)[:12]
        top_score = max((scores[m] for m in ranked), default=0.0) or 1.0   # normalise to [0,1]
        print("\n=== RANKING (taste mode) ===")
        recommendations = []
        for mid in ranked:
            ctx = contexts[mid]
            norm = max(0.0, round(scores[mid] / top_score, 2))
            print(f"✅ {ctx.title}: {norm:.2f} (taste {scores[mid]:+.1f})")
            recommendations.append({
                "title": ctx.title,
                "year": ctx.release_date[:4] if ctx.release_date else "",
                "tmdb_id": mid,
                "score": norm,
                "reason": "matches your taste",
            })
        return {"final_recommendations": ranked, "recommendations": recommendations,
                "movie_contexts": contexts}

    listed = [(i, contexts[mid]) for i, mid in enumerate(candidate_ids) if mid in contexts]
    candidates_text = "\n".join(
        f"{i} | {ctx.title} ({ctx.release_date[:4]}) | "
        f"{', '.join(g['name'] for g in ctx.genres)} | "
        f"kw: {', '.join(k['name'] for k in ctx.keywords[:6])} | "
        f"{ctx.overview[:140]}"
        for i, ctx in listed
    )
    seed_text = "\n\n".join(
        format_context(contexts[mid])
        for mid in state["seed_movie_ids"]
        if mid in contexts
    )

    class RankedMovie(BaseModel):
        """One candidate's score, referenced by its list number."""
        index: int = Field(description="The candidate's number from the list")
        score: float = Field(description="0.0 to 1.0")
        reason: str = Field(default="", description="3-6 words max. No apostrophes.")

    class RankingOutput(BaseModel):
        """Relevance scores for every candidate movie (one entry per candidate)."""
        movies: list[RankedMovie]

    extractor = create_extractor(llm_strong, tools=[RankingOutput], tool_choice=RankingOutput.__name__)
    response = extractor.invoke({"messages": [
        SystemMessage(content=f"""
You MUST call the RankingOutput function with your scores. Do not reply with plain text.

You are a movie relevance ranking agent.
Score every candidate from 0.0 to 1.0 based on thematic similarity to the seed movies
and the user query.

Candidates have already been filtered for hard constraints (country / language / date)
upstream. Focus purely on tone, themes, style, genres, keywords, crew, cast, overview.

If the user names a SETTING - a city, region or era (e.g. "in New York", "set in
Tokyo", "1970s") - treat it as a strong signal: prefer films clearly set there and
penalise films set elsewhere, judging from the overview and your own knowledge. TMDB
cannot filter below country level, so the ranker is the only place a named city or era
is honoured.

CRITICAL OUTPUT REQUIREMENT:
- Each candidate is prefixed with a NUMBER. Reference it in the `index` field.
- Output EXACTLY {len(listed)} entries - one per candidate, each index appearing once.
- Do NOT omit any candidate. If unsure, score it 0.0, but every number must appear.

Scoring:
- Only truly similar films score above 0.6.
- Obvious mismatches (wrong tone, comedy when user wants bleak, etc.) get < 0.2.
- For every candidate also give reason: a SHORT 3-6 word phrase, no apostrophes.
"""),
        HumanMessage(content=f"""
User query: {state['query_text']}
User-extracted constraints: {state['constraints']}

Seed movies (reference points for similarity):
{seed_text}

Candidates to score (count: {len(listed)}), each prefixed with its number:
{candidates_text}
""")
    ]})["responses"][0]

    # The model references candidates by list number; map back to tmdb_id.
    picks = []   # (tmdb_id, score, reason)
    seen: set[int] = set()
    for m in response.movies:
        if not (0 <= m.index < len(candidate_ids)):
            continue
        tid = candidate_ids[m.index]
        if tid in seen:
            continue
        seen.add(tid)
        picks.append((tid, m.score, m.reason))

    profile = build_taste_profile(state.get("movie_ratings", {}))
    raws = {}
    if profile:
        raws = {tid: _taste_score(contexts[tid], profile) for tid, _, _ in picks if tid in contexts}
    if raws and any(v != 0 for v in raws.values()):
        scale = max(abs(v) for v in raws.values()) or 1.0

        def _final(tid, score):
            return max(0.0, min(1.0, score + TASTE_WEIGHT * (raws.get(tid, 0.0) / scale)))

        print(f"[ranking] taste profile active ({len(profile['keywords'])} kw / {len(profile['genres'])} genres), "
              f"+/-{int(TASTE_WEIGHT * 100)} taste on relevance")
        print_taste_profile(profile)
    else:
        def _final(tid, score):
            return score

    scored = sorted(
        ((tid, score, reason, _final(tid, score)) for tid, score, reason in picks),
        key=lambda x: x[3], reverse=True,
    )

    print("\n=== RANKING ===")
    for tid, score, reason, fs in scored:
        ctx = contexts.get(tid)
        title = ctx.title if ctx else tid
        extra = f" [rel {score:.2f}]" if raws else ""
        print(f"{'✅' if fs >= 0.5 else '❌'} {title}: {fs:.2f}{extra} — {reason}")

    final = [tid for tid, _, _, fs in scored if fs >= 0.5]
    recommendations = [
        {
            "title": contexts[tid].title if tid in contexts else str(tid),
            "year": (contexts[tid].release_date[:4]
                     if tid in contexts and contexts[tid].release_date else ""),
            "tmdb_id": tid,
            "score": round(fs, 2),
            "reason": reason,
        }
        for tid, _, reason, fs in scored if fs >= 0.5
    ]
    return {
        "final_recommendations": final,
        "recommendations": recommendations,
        "movie_contexts": contexts,
    }

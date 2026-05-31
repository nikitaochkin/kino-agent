"""Structured-output schemas for the trustcall extractors."""
from pydantic import BaseModel, Field, field_validator, model_validator


class ParsedQuery(BaseModel):
    """Structured representation of a user's movie recommendation query."""

    seed_titles: list[str] | str = Field(
        default_factory=list,
        description="Movie titles mentioned by the user as reference points",
    )
    constraints: dict = Field(
        default_factory=dict,
        description="Explicit recommendation constraints and preferences",
    )
    release_date_gte: str = Field(
        default="",
        description="Minimum release date in YYYY-MM-DD format, derived from time period constraint",
    )
    release_date_lte: str = Field(
        default="",
        description="Maximum release date in YYYY-MM-DD format, derived from time period constraint",
    )
    with_original_language: str = Field(
        default="",
        description="ISO 639-1 language code if user specified a language",
    )
    with_origin_country: str = Field(
        default="",
        description="ISO 3166-1 alpha-2 country codes separated by | (OR) or , (AND)",
    )

    @model_validator(mode="before")
    @classmethod
    def _none_to_defaults(cls, data):
        if isinstance(data, dict):
            if data.get("seed_titles") is None:
                data["seed_titles"] = []
            elif isinstance(data.get("seed_titles"), str):
                data["seed_titles"] = [data["seed_titles"]]
            if data.get("constraints") is None:
                data["constraints"] = {}
            for k in ("release_date_gte", "release_date_lte",
                      "with_original_language", "with_origin_country"):
                if data.get(k) is None:
                    data[k] = ""
        return data


class AnalysisOutput(BaseModel):
    """Single-call analysis across five axes (replaces the 5 parallel agents).

    Keyword axes are kept as separate fields so build_filters can round-robin
    across dimensions; downstream they are unioned into one keyword bag.
    """
    mood_keywords:      list[str] | str | None = Field(default_factory=list)
    theme_keywords:     list[str] | str | None = Field(default_factory=list)
    visual_keywords:    list[str] | str | None = Field(default_factory=list)
    pacing_keywords:    list[str] | str | None = Field(default_factory=list)
    character_keywords: list[str] | str | None = Field(default_factory=list)
    without_keywords:   list[str] | str | None = Field(default_factory=list)

    with_genres:    list[str] | str | None = Field(default_factory=list)
    without_genres: list[str] | str | None = Field(default_factory=list)
    with_crew: list[str] | str | None = Field(
        default_factory=list,
        description="Cinematographers or directors known for matching visual style")
    with_cast: list[str] | str | None = Field(
        default_factory=list,
        description="Actors ONLY if the user explicitly named them in the query")
    runtime_gte: int | str | None = Field(default=None)
    runtime_lte: int | str | None = Field(default=None)
    axis_priority: list[str] = Field(
        default_factory=list,
        description=("The five axis names ordered by relevance TO THIS QUERY, most "
                     "relevant first. Use exactly these names: mood_keywords, "
                     "theme_keywords, visual_keywords, pacing_keywords, character_keywords."))
    reasoning: str = Field(description="One short sentence. No apostrophes or single quotes.")

    @model_validator(mode="before")
    @classmethod
    def _normalize_list_fields(cls, data):
        # Coerce str -> [str] and None -> [] for every list-shaped field, so a model
        # that returns a bare string instead of a list does not break extraction.
        if isinstance(data, dict):
            for k, v in list(data.items()):
                if (k.endswith("_keywords") or k.startswith("with_")
                        or k.startswith("without_") or k == "axis_priority"):
                    if v is None:
                        data[k] = []
                    elif isinstance(v, str):
                        data[k] = [v] if v.strip() else []
        return data

    @field_validator("runtime_gte", "runtime_lte", mode="before")
    @classmethod
    def _coerce_int(cls, v):
        if v is None or v == "":
            return None
        return int(v)

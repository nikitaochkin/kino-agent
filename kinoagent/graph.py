"""Assemble and export the compiled LangGraph graph (`graph`).

Referenced by langgraph.json as `kinoagent.graph:graph` for `langgraph dev` / Studio.
"""
from langgraph.graph import END, START, StateGraph

from .nodes import (
    analysis_node,
    build_filters_node,
    discover_agent,
    load_user_data_node,
    movie_enrichment_node,
    query_parser_agent,
    ranking_agent,
    relax_constraints_node,
    route_after_discover,
    seed_resolver_node,
)
from .state import InputState, OutputState, RecommendationState

builder = StateGraph(RecommendationState, input_schema=InputState, output_schema=OutputState)

builder.add_node("load_user_data", load_user_data_node)
builder.add_node("query_parser", query_parser_agent)
builder.add_node("seed_resolver", seed_resolver_node)
builder.add_node("movie_enrichment", movie_enrichment_node)
builder.add_node("analysis", analysis_node)
builder.add_node("build_filters", build_filters_node)
builder.add_node("discover", discover_agent)
builder.add_node("relax_constraints", relax_constraints_node)
builder.add_node("ranking", ranking_agent)

builder.add_edge(START, "load_user_data")
builder.add_edge("load_user_data", "query_parser")
builder.add_edge("query_parser", "seed_resolver")
builder.add_edge("seed_resolver", "movie_enrichment")
builder.add_edge("movie_enrichment", "analysis")
builder.add_edge("analysis", "build_filters")
builder.add_edge("build_filters", "discover")

# Relaxation loop: if discover comes back thin, loosen a soft knob and retry it.
builder.add_conditional_edges(
    "discover",
    route_after_discover,
    {"ranking": "ranking", "relax_constraints": "relax_constraints"},
)
builder.add_edge("relax_constraints", "discover")
builder.add_edge("ranking", END)

graph = builder.compile()

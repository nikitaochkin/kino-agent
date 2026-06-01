"""Environment + shared clients (TMDB, Groq LLMs).

Importing this module loads .env and configures the global `tmdbsimple` API key,
so every other module can just `from .config import tmdb`.
"""
import os

import tmdbsimple as tmdb
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()
tmdb.API_KEY = os.getenv("TMDB_API_KEY")
tmdb.REQUESTS_TIMEOUT = (3.05, 20)  # (connect, read) seconds — fail fast instead of hanging
if os.getenv("GROQ_API_KEY"):
    os.environ.setdefault("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))

# Parser, analysis and ranking all run on llm_strong. gpt-oss spends output on its
# reasoning channel; with trustcall's forced tool_choice it can finish WITHOUT emitting
# the call (Groq 400, not retried). reasoning_effort + max_tokens leave budget for it.
llm_strong = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    max_retries=3,
    max_tokens=2048,            # free tier caps gpt-oss-120b at 8000 TPM; the reserved
                                # max_tokens counts toward "requested", so keep it modest
                                # (structured outputs here need well under 2048 anyway)
    reasoning_effort="medium",
)

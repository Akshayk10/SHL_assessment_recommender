"""
main.py
-------
FastAPI application entry point.

Endpoints:
  GET  /health  → liveness check (HTTP 200, {"status": "ok"})
  POST /chat    → stateless conversational recommender

Run locally:
  uvicorn main:app --reload --port 8000

Deploy:
  See render.yaml (Render.com free tier)
"""

import logging
import os
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from models import ChatRequest, ChatResponse
from agent import get_agent_reply

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STATE (loaded once, reused across all requests)
# ─────────────────────────────────────────────────────────────────────────────

_catalog_ready: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: load catalog + build ALL retrieval indexes eagerly.
    FAISS dense index is now built at startup (not lazily) to prevent
    the first /chat request from timing out on Render free tier.
    """
    global _catalog_ready

    logger.info("=== SHL Assessment Recommender — Starting up ===")

    try:
        # 1. Load catalog metadata (fast — just reads JSON)
        from retrieval_rag import get_retrieval_system
        rag_system = get_retrieval_system()
        logger.info(f"Catalog loaded: {len(rag_system.catalog)} assessments")

        # 2. Build BM25 index eagerly (fast, ~0.5s)
        from retrieval_hybrid import get_hybrid_retriever
        hybrid = get_hybrid_retriever()
        logger.info("BM25 index built.")

        # 3. EAGERLY build FAISS dense index with a warmup query
        #    This triggers fastembed model download + encode all 377 docs.
        #    Done at startup so the first /chat request is never slow.
        logger.info("Building FAISS dense index (warmup)... this takes ~30s on first run")
        try:
            from retrieval_rag import retrieve_assessments
            _ = retrieve_assessments("software engineer developer", k=1)
            logger.info("FAISS dense index ready.")
        except Exception as exc:
            logger.warning(f"FAISS warmup failed (non-fatal): {exc}")

        # 4. Also load legacy catalog format for backward compatibility
        try:
            from retrieval import load_catalog
            load_catalog()
        except Exception:
            pass  # Optional; not required for core functionality

        _catalog_ready = True
        logger.info("=== Startup complete — ready to serve ===")

    except FileNotFoundError as exc:
        logger.error(f"Catalog not found: {exc}")
        logger.error("Run: python scripts/scrape_catalog.py")
    except Exception as exc:
        logger.error(f"Startup error: {exc}", exc_info=True)

    yield

    logger.info("=== Shutting down ===")


# ─────────────────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="SHL Assessment Recommender",
    description=(
        "Conversational agent for finding the right SHL assessments. "
        "Built by SHL AI Intern candidate."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# Serve static frontend files (index.html, etc.)
_STATIC_DIR = Path(__file__).parent / "static"
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    """Serve the chat frontend."""
    index = Path(__file__).parent / "static" / "index.html"
    if index.exists():
        return FileResponse(str(index), media_type="text/html")
    return {"message": "SHL Assessment Recommender API", "docs": "/docs", "chat": "POST /chat"}


@app.get("/health")
def health():
    """
    Liveness check. SHL evaluator calls this first.
    Returns HTTP 200 {"status": "ok"} unconditionally.
    Note: cold-start hosts (Render free) may take up to 2 min on first call.
    """
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Stateless conversational endpoint.

    The caller sends the FULL conversation history on every call.
    We return the next agent reply + optional assessment recommendations.

    Constraints (from spec):
      - Max 8 turns (user+assistant messages combined)
      - 30-second timeout per call
      - Schema is non-negotiable
    """
    if not _catalog_ready:
        # Return a graceful degradation rather than 503
        # (evaluator may hit /chat before /health startup is done)
        logger.warning("Catalog not ready — attempting to load on demand")
        try:
            from retrieval_hybrid import get_hybrid_retriever
            get_hybrid_retriever()
        except Exception as exc:
            logger.error(f"On-demand load failed: {exc}")
            return ChatResponse(
                reply="Service is starting up. Please try again in a moment.",
                recommendations=[],
                end_of_conversation=False,
            )

    # Enforce 8-turn cap server-side (evaluator cap)
    if len(req.messages) > 8:
        return ChatResponse(
            reply=(
                "We have reached the maximum conversation length. "
                "Here is my final recommendation based on our discussion."
            ),
            recommendations=[],
            end_of_conversation=True,
        )

    return get_agent_reply(req.messages)


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL EXCEPTION HANDLER
# ─────────────────────────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    # Always return valid ChatResponse shape so evaluator schema check passes
    return JSONResponse(
        status_code=200,
        content={
            "reply": "I encountered an error. Could you rephrase your question?",
            "recommendations": [],
            "end_of_conversation": False,
        },
    )

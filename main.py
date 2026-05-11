"""
main.py
-------
FastAPI application entry point.

Endpoints:
  GET  /health  → liveness check
  POST /chat    → stateless conversational recommender

Run locally:
  uvicorn app.main:app --reload --port 8000
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from models import ChatRequest, ChatResponse
from agent import get_agent_reply
from retrieval import load_catalog, format_catalog_for_prompt, get_catalog


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


_catalog_text: str = ""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load catalog at startup, build index lazily"""
    global _catalog_text
    
    logger.info("Loading SHL catalog...")
    try:
        from retrieval_rag import get_retrieval_system
        retrieval = get_retrieval_system()
        logger.info(f"Catalog loaded: {len(retrieval.catalog)} assessments")
        logger.info("FAISS index will be built on first search (lazy loading)")
        
        from retrieval import load_catalog, format_catalog_for_prompt
        catalog = load_catalog()
        _catalog_text = format_catalog_for_prompt(catalog)
        
    except Exception as e:
        logger.error(f"Startup error: {e}")
    
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational agent for finding the right SHL assessments.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    """Liveness check. SHL evaluator calls this first."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Stateless conversational endpoint.
    The caller sends the FULL conversation history on every call.
    We return the next agent reply + optional recommendations.
    """
    if not _catalog_text:
        raise HTTPException(
            status_code=503,
            detail="Catalog not loaded. Run scripts/scrape_catalog.py first.",
        )

    if len(req.messages) > 8:
        return ChatResponse(
            reply="We've reached the maximum conversation length. Here is my final recommendation based on our discussion.",
            recommendations=[],
            end_of_conversation=True,
        )

    response = get_agent_reply(req.messages, _catalog_text)
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=200,
        content={
            "reply": "I encountered an error. Could you rephrase your question?",
            "recommendations": [],
            "end_of_conversation": False,
        },
    )

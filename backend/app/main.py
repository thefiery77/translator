"""
FastAPI application entry point.

Startup sequence:
  1. Connect to MongoDB (Motor)
  2. Connect to Qdrant
  3. Initialise repository singletons
  4. Inject repos into workflow nodes that need them
  5. Rebuild LangGraph with a persistent checkpointer (MemorySaver for MVP)

Run:
  uvicorn app.main:app --reload --port 8000
"""
from __future__ import annotations

import logging

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from qdrant_client import AsyncQdrantClient

from app.api.routes.books import router as books_router
from app.api.routes.reviews import router as reviews_router
from app.core.config import get_settings
from app.repositories.mongo.book_repo import BookRepository
from app.repositories.mongo.memory_repo import MemoryRepository
from app.repositories.mongo.review_repo import ReviewRepository
from app.repositories.vector.qdrant_repo import QdrantRepository
from app.services.memory_service import MemoryService
from app.workflow.nodes.retrieval import init_retrieval_repos
from app.workflow.nodes.human_review import init_review_repo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
settings = get_settings()


# ─────────────────────────────────────────────────────────────────
# Lifespan
# ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Connect ───────────────────────────────────────────────────
    try:
        logger.info(f"Connecting to MongoDB: {settings.mongodb_uri.split('@')[1] if '@' in settings.mongodb_uri else 'unknown'}")
        mongo_client = AsyncIOMotorClient(settings.mongodb_uri)
        db = mongo_client[settings.mongodb_db]
        logger.info(f"✓ MongoDB connected to database '{settings.mongodb_db}'")
    except Exception as e:
        logger.error(f"✗ MongoDB connection failed: {e}", exc_info=True)
        raise

    try:
        logger.info(f"Connecting to Qdrant: {settings.qdrant_url}")
        qdrant_client = AsyncQdrantClient(url=settings.qdrant_url)
        logger.info("✓ Qdrant connected")
    except Exception as e:
        logger.error(f"✗ Qdrant connection failed: {e}", exc_info=True)
        raise

    # ── Repositories ──────────────────────────────────────────────
    try:
        logger.info("Initialising repositories...")
        book_repo = BookRepository(db)
        memory_repo = MemoryRepository(db)
        review_repo = ReviewRepository(db)
        qdrant_repo = QdrantRepository(qdrant_client)
        memory_service = MemoryService(memory_repo, qdrant_repo)
        logger.info("✓ Repositories initialised")
    except Exception as e:
        logger.error(f"✗ Repository initialisation failed: {e}", exc_info=True)
        raise

    # ── Inject into workflow nodes ────────────────────────────────
    try:
        logger.info("Injecting repositories into workflow nodes...")
        init_retrieval_repos(memory_repo, qdrant_repo)
        init_review_repo(review_repo)
        logger.info("✓ Workflow nodes initialised")
    except Exception as e:
        logger.error(f"✗ Workflow node injection failed: {e}", exc_info=True)
        raise

    # ── Store on app.state for dependency injection in routes ─────
    app.state.book_repo = book_repo
    app.state.memory_repo = memory_repo
    app.state.review_repo = review_repo
    app.state.qdrant_repo = qdrant_repo
    app.state.memory_service = memory_service

    logger.info("✓ Application startup complete")
    yield

    # ── Shutdown ──────────────────────────────────────────────────
    logger.info("Shutting down...")
    mongo_client.close()
    await qdrant_client.close()
    logger.info("✓ Application shutdown complete")


# ─────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Literary Translation API — Japanese → Italian",
    version="0.1.0",
    description=(
        "Multi-agent literary translation pipeline with critic ensemble, "
        "patch synthesis, and human review queue."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request injection (pass request to route handlers via Depends) ──
# FastAPI routes that need app.state receive the `request` parameter.
# We use a simple middleware to attach the request to the scope.


@app.middleware("http")
async def attach_request(request: Request, call_next):
    logger.debug(f"{request.method} {request.url.path}")
    try:
        response = await call_next(request)
        logger.debug(f"→ {response.status_code}")
        return response
    except Exception as e:
        logger.error(f"Request failed: {e}", exc_info=True)
        raise


# ── Routers ───────────────────────────────────────────────────────
app.include_router(books_router, prefix="/api/v1")
app.include_router(reviews_router, prefix="/api/v1")


# ── Health ─────────────────────────────────────────────────────────
@app.get("/health", tags=["system"])
async def health():
    return {"status": "ok"}

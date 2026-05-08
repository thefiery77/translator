"""
API routes — Book ingestion and translation job management.

Endpoints:
  POST /books                         — upload book file → trigger ingestion + chunking
  GET  /books/{book_id}               — book status + progress
  GET  /books/{book_id}/chunks        — list chunks and their translation status
  POST /books/{book_id}/translate     — start translation workflow for all pending chunks
"""
from __future__ import annotations

import asyncio
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, UploadFile
from fastapi import File as FastFile
from fastapi import Form, status

from app.ingestion.chunker import create_semantic_chunks
from app.ingestion.parser import parse_book
from app.repositories.mongo.book_repo import BookRepository
from app.repositories.mongo.memory_repo import MemoryRepository
from app.schemas.contracts import (
    Book,
    BookMetadata,
    BookStatus,
    SourceFormat,
    TranslationJobStatus,
)
from app.workflow.graph import translation_graph

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/books", tags=["books"])


# ─────────────────────────────────────────────────────────────────
# Dependency helpers (singletons injected at startup via app.state)
# ─────────────────────────────────────────────────────────────────

def _get_book_repo(request: Request) -> BookRepository:
    return request.app.state.book_repo


def _get_memory_repo(request: Request) -> MemoryRepository:
    return request.app.state.memory_repo


# ─────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────

@router.get("")
async def list_books(
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
    book_repo: BookRepository = Depends(_get_book_repo),
):
    """List all books, optionally filtered by status."""
    books = await book_repo.list_books(status=status, limit=limit, offset=offset)
    return [
        {
            "book_id": b.id,
            "title": b.title,
            "author": b.metadata.author if b.metadata else "",
            "status": b.status,
            "source_format": b.source_format,
            "total_chunks": b.total_chunks,
            "completed_chunks": b.completed_chunks,
            "created_at": b.created_at.isoformat() if b.created_at else None,
        }
        for b in books
    ]


@router.post("", status_code=status.HTTP_202_ACCEPTED)
async def upload_book(
    background_tasks: BackgroundTasks,
    file: UploadFile = FastFile(...),
    title: str = Form(...),
    author: str = Form(""),
    book_repo: BookRepository = Depends(_get_book_repo),
):
    """
    Upload a book file (EPUB / DOCX / PDF / TXT) and trigger ingestion.

    The response is returned immediately; ingestion and chunking run in the
    background. Poll GET /books/{book_id} for status updates.
    """
    source_format = _detect_format(file.filename or "")
    if source_format is None:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Formato non supportato. Usa EPUB, DOCX, PDF o TXT.",
        )

    book = Book(
        title=title,
        source_format=source_format,
        status=BookStatus.INGESTED,
        metadata=BookMetadata(author=author),
    )
    await book_repo.create_book(book)

    # Eagerly read upload into a temp file (UploadFile is not safe across tasks)
    content = await file.read()
    suffix = Path(file.filename or "book.txt").suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(content)
    tmp.flush()
    tmp_path = Path(tmp.name)
    tmp.close()

    background_tasks.add_task(
        _run_ingestion,
        book_id=book.id,
        tmp_path=tmp_path,
        source_format=source_format,
        book_repo=book_repo,
    )

    return {"book_id": book.id, "status": book.status, "title": book.title}


@router.get("/{book_id}")
async def get_book(
    book_id: str,
    book_repo: BookRepository = Depends(_get_book_repo),
):
    book = await book_repo.get_book(book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Libro non trovato")
    return book.model_dump()


@router.get("/{book_id}/chunks")
async def list_chunks(
    book_id: str,
    book_repo: BookRepository = Depends(_get_book_repo),
):
    chunks = await book_repo.list_chunks(book_id)
    return [
        {
            "chunk_id": c.id,
            "chunk_index": c.chunk_index,
            "chapter": c.chapter,
            "token_count": c.token_count,
            "status": c.status,
        }
        for c in chunks
    ]


@router.post("/{book_id}/translate", status_code=status.HTTP_202_ACCEPTED)
async def start_translation(
    book_id: str,
    background_tasks: BackgroundTasks,
    book_repo: BookRepository = Depends(_get_book_repo),
):
    """Start the translation workflow for all pending chunks of a book."""
    book = await book_repo.get_book(book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Libro non trovato")
    if book.status not in (BookStatus.CHUNKED.value, BookStatus.CHUNKED):
        raise HTTPException(
            status_code=400,
            detail=f"Il libro deve essere in stato 'chunked'. Stato attuale: {book.status}",
        )

    chunks = await book_repo.list_chunks(book_id)
    pending = [c for c in chunks if c.status == "pending"]

    if not pending:
        return {"message": "Nessun chunk in attesa di traduzione.", "queued": 0}

    await book_repo.update_book_status(book_id, BookStatus.TRANSLATING)

    background_tasks.add_task(
        _run_translation_jobs,
        book_id=book_id,
        chunks=pending,
        book_repo=book_repo,
    )

    return {
        "message": f"Traduzione avviata per {len(pending)} chunk.",
        "queued": len(pending),
    }


# ─────────────────────────────────────────────────────────────────
# Background tasks
# ─────────────────────────────────────────────────────────────────

async def _run_ingestion(
    book_id: str,
    tmp_path: Path,
    source_format: SourceFormat,
    book_repo: BookRepository,
) -> None:
    try:
        await book_repo.update_book_status(book_id, BookStatus.CHUNKING)

        # Parse
        paragraphs = await asyncio.to_thread(parse_book, tmp_path, source_format)
        logger.info("Parsed %d paragraphs for book %s", len(paragraphs), book_id)

        # Group by chapter and chunk
        chapters: dict[int, list] = {}
        for p in paragraphs:
            chapters.setdefault(p.chapter, []).append(p)

        all_chunks = []
        for chapter_num, paras in sorted(chapters.items()):
            chunks = create_semantic_chunks(
                book_id=book_id,
                paragraphs=paras,
                chapter=chapter_num,
            )
            all_chunks.extend(chunks)

        await book_repo.create_chunks(all_chunks)
        await book_repo.update_book_status(
            book_id,
            BookStatus.CHUNKED,
            total_chunks=len(all_chunks),
        )
        logger.info("Created %d chunks for book %s", len(all_chunks), book_id)

    except Exception as exc:
        logger.error("Ingestion failed for book %s: %s", book_id, exc)
        await book_repo.update_book_status(book_id, BookStatus.FAILED)
    finally:
        tmp_path.unlink(missing_ok=True)


async def _run_translation_jobs(
    book_id: str,
    chunks: list,
    book_repo: BookRepository,
) -> None:
    """Run translation graph for each chunk sequentially to bound LLM concurrency."""
    from datetime import timezone

    for chunk in chunks:
        try:
            await book_repo.update_chunk_status(chunk.id, "translating")

            initial_state = {
                "job_id": f"job_{chunk.id}",
                "book_id": book_id,
                "chunk_id": chunk.id,
                "retry_count": 0,
                "source_text": chunk.source_text,
                "paragraphs": [p.model_dump() for p in chunk.paragraphs],
                "chunk_context": chunk.context.model_dump(),
                "started_at": datetime.now(timezone.utc).isoformat(),
                "audit_trail": [],
                "node_timings": {},
                "total_tokens": 0,
            }

            config = {"configurable": {"thread_id": chunk.id}}
            await translation_graph.ainvoke(initial_state, config=config)

            await book_repo.update_chunk_status(chunk.id, "completed")
            await book_repo.increment_completed_chunks(book_id)

        except Exception as exc:
            logger.error(
                "Translation job failed for chunk %s (book %s): %s",
                chunk.id, book_id, exc,
            )
            await book_repo.update_chunk_status(chunk.id, "failed")


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

_FORMAT_MAP = {
    ".epub": SourceFormat.EPUB,
    ".docx": SourceFormat.DOCX,
    ".pdf": SourceFormat.PDF,
    ".txt": SourceFormat.TXT,
}


def _detect_format(filename: str) -> SourceFormat | None:
    suffix = Path(filename).suffix.lower()
    return _FORMAT_MAP.get(suffix)

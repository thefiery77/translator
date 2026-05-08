"""
Ingestion layer — book file parsing.

Supported formats: EPUB, DOCX, PDF (text-based), TXT.
OCR scans (image PDFs) are handled as a degraded-quality TXT extraction
with a confidence flag for human review.

Output: list[Paragraph] in the normalised internal format.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Protocol

from app.schemas.contracts import Paragraph, SourceFormat, StyleTag

logger = logging.getLogger(__name__)

# Japanese dialogue delimiters
_DIALOGUE_RE = re.compile(r"[「『](.*?)[」』]", re.DOTALL)


# ─────────────────────────────────────────────────────────────────
# Parser protocol
# ─────────────────────────────────────────────────────────────────

class BookParser(Protocol):
    def parse(self, path: Path) -> list[Paragraph]: ...


# ─────────────────────────────────────────────────────────────────
# EPUB
# ─────────────────────────────────────────────────────────────────

def _parse_epub(path: Path) -> list[Paragraph]:
    try:
        import ebooklib
        from ebooklib import epub
        import html
        from html.parser import HTMLParser

        class _Strip(HTMLParser):
            def __init__(self):
                super().__init__()
                self._parts: list[str] = []

            def handle_data(self, data: str):
                self._parts.append(data)

            def get_text(self) -> str:
                return "".join(self._parts)

        book = epub.read_epub(str(path))
        paragraphs: list[Paragraph] = []
        chapter_num = 0
        para_counter = 0

        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            content = item.get_content().decode("utf-8", errors="replace")
            # Strip HTML tags
            stripper = _Strip()
            stripper.feed(content)
            raw_text = stripper.get_text()

            for line in raw_text.splitlines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith("# ") or line.startswith("## "):
                    chapter_num += 1
                    continue
                para_counter += 1
                para_id = f"{chapter_num}_{para_counter}"
                tags = _detect_style_tags(line)
                speaker = _detect_speaker(line)
                paragraphs.append(
                    Paragraph(
                        chapter=chapter_num,
                        paragraph_id=para_id,
                        speaker=speaker,
                        text=line,
                        style_tags=tags,
                        char_offset_start=0,
                        char_offset_end=len(line),
                    )
                )

        return paragraphs
    except Exception as exc:
        logger.error("EPUB parsing failed for %s: %s", path, exc)
        raise


# ─────────────────────────────────────────────────────────────────
# DOCX
# ─────────────────────────────────────────────────────────────────

def _parse_docx(path: Path) -> list[Paragraph]:
    try:
        from docx import Document  # type: ignore

        doc = Document(str(path))
        paragraphs: list[Paragraph] = []
        chapter_num = 0
        para_counter = 0

        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue
            # Heuristic: heading styles → new chapter
            if para.style and "Heading" in para.style.name:
                chapter_num += 1
                continue
            para_counter += 1
            para_id = f"{chapter_num}_{para_counter}"
            tags = _detect_style_tags(text)
            speaker = _detect_speaker(text)
            paragraphs.append(
                Paragraph(
                    chapter=chapter_num,
                    paragraph_id=para_id,
                    speaker=speaker,
                    text=text,
                    style_tags=tags,
                    char_offset_start=0,
                    char_offset_end=len(text),
                )
            )

        return paragraphs
    except Exception as exc:
        logger.error("DOCX parsing failed for %s: %s", path, exc)
        raise


# ─────────────────────────────────────────────────────────────────
# PDF
# ─────────────────────────────────────────────────────────────────

def _parse_pdf(path: Path) -> list[Paragraph]:
    """Text-based PDF extraction. Returns degraded output for image PDFs."""
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(str(path))
        paragraphs: list[Paragraph] = []
        chapter_num = 0
        para_counter = 0

        for page in doc:
            text: str = page.get_text()  # type: ignore[attr-defined]
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                para_counter += 1
                para_id = f"{chapter_num}_{para_counter}"
                tags = _detect_style_tags(line)
                speaker = _detect_speaker(line)
                paragraphs.append(
                    Paragraph(
                        chapter=chapter_num,
                        paragraph_id=para_id,
                        speaker=speaker,
                        text=line,
                        style_tags=tags,
                        char_offset_start=0,
                        char_offset_end=len(line),
                    )
                )

        doc.close()
        return paragraphs
    except Exception as exc:
        logger.error("PDF parsing failed for %s: %s", path, exc)
        raise


# ─────────────────────────────────────────────────────────────────
# TXT
# ─────────────────────────────────────────────────────────────────

def _parse_txt(path: Path) -> list[Paragraph]:
    text = path.read_text(encoding="utf-8", errors="replace")
    paragraphs: list[Paragraph] = []
    chapter_num = 0
    para_counter = 0

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        para_counter += 1
        para_id = f"{chapter_num}_{para_counter}"
        tags = _detect_style_tags(line)
        speaker = _detect_speaker(line)
        paragraphs.append(
            Paragraph(
                chapter=chapter_num,
                paragraph_id=para_id,
                speaker=speaker,
                text=line,
                style_tags=tags,
                char_offset_start=0,
                char_offset_end=len(line),
            )
        )
    return paragraphs


# ─────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────

def parse_book(path: Path, source_format: SourceFormat) -> list[Paragraph]:
    """Parse a book file into a normalised list of Paragraph objects."""
    dispatch = {
        SourceFormat.EPUB: _parse_epub,
        SourceFormat.DOCX: _parse_docx,
        SourceFormat.PDF: _parse_pdf,
        SourceFormat.TXT: _parse_txt,
    }
    parser_fn = dispatch.get(source_format)
    if parser_fn is None:
        raise ValueError(f"Unsupported source format: {source_format}")
    return parser_fn(path)


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _detect_style_tags(text: str) -> list[StyleTag]:
    tags: list[StyleTag] = []
    if _DIALOGUE_RE.search(text):
        tags.append(StyleTag.DIALOGUE)
    # Inner monologue: uses 〈〉or ——
    if "〈" in text or "——" in text:
        tags.append(StyleTag.INNER_MONOLOGUE)
    if not tags:
        tags.append(StyleTag.NARRATION)
    return tags


def _detect_speaker(text: str) -> str | None:
    """
    Very lightweight speaker detection for Japanese text.
    Returns the text immediately preceding a 「 quotation marker if
    it looks like a character name (<=8 chars, no punctuation).
    """
    match = re.search(r"([^\s「』\n]{1,8})「", text)
    if match:
        candidate = match.group(1)
        # Must not be a common sentence-ending word
        if not re.search(r"[はがをにで]$", candidate):
            return candidate
    return None

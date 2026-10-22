"""Read the paper PDF and prepare searchable text chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader

from config import (
    DEFAULT_CHUNK_OVERLAP_WORDS,
    DEFAULT_CHUNK_SIZE_WORDS,
    MIN_EXTRACTED_CHARACTERS,
)


REFERENCE_SECTION_PATTERN = re.compile(r"\bReferences\b\s+1\.", re.IGNORECASE)


@dataclass(frozen=True)
class PageText:
    """Clean text extracted from one page of the paper."""

    page_number: int
    text: str


@dataclass(frozen=True)
class TextChunk:
    """A searchable paper chunk with its page number preserved."""

    chunk_id: str
    page_number: int
    text: str


def validate_pdf_path(pdf_path: str | Path) -> Path:
    """Check that the supplied path is a readable PDF file."""

    # Give a direct file error before any extraction is tried.
    path = Path(pdf_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")
    if not path.is_file():
        raise ValueError(f"PDF path is not a file: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, received: {path}")
    return path


def clean_text(text: str) -> str:
    """Clean PDF text while keeping the sentence content readable."""

    if not text:
        return ""

    # PDF extraction often inserts line breaks, repeated spaces, and broken
    # hyphenation. This step makes the text suitable for chunking and TF-IDF.
    cleaned = text.replace("\x00", " ").replace("\r", "\n")
    cleaned = re.sub(r"([A-Za-z])-[\n\s]+([A-Za-z])", r"\1\2", cleaned)
    cleaned = re.sub(r"\s*\n\s*", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"([(])\s+", r"\1", cleaned)
    cleaned = re.sub(r"\s+([)])", r"\1", cleaned)
    return cleaned.strip()


def remove_reference_section_start(text: str) -> tuple[str, bool]:
    """Stop retrieval text once the paper reaches its references section."""

    # The bibliography repeats many paper titles. Removing it keeps retrieval
    # focused on the authors' own analysis rather than citation text.
    match = REFERENCE_SECTION_PATTERN.search(text)
    if not match:
        return text, False
    return text[: match.start()].strip(), True


def load_pdf_pages(pdf_path: str | Path) -> list[PageText]:
    """Extract clean text from the PDF one page at a time."""

    path = validate_pdf_path(pdf_path)

    try:
        reader = PdfReader(str(path))
    except Exception as exc:  # pypdf can raise different parser-specific errors.
        raise ValueError(f"Could not open PDF '{path}': {exc}") from exc

    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as exc:
            raise ValueError("The PDF appears to be encrypted and could not be read.") from exc

    pages: list[PageText] = []
    for page_index, page in enumerate(reader.pages, start=1):
        try:
            # Page numbers are stored with the text so answers can cite sources.
            raw_text = page.extract_text() or ""
        except Exception:
            raw_text = ""

        text = clean_text(raw_text)
        if text:
            pages.append(PageText(page_number=page_index, text=text))

    total_characters = sum(len(page.text) for page in pages)
    if total_characters < MIN_EXTRACTED_CHARACTERS:
        # This protects the demonstration from scanned PDFs or the wrong file.
        raise ValueError(
            "The PDF contains little or no extractable text. "
            "Check that the correct academic paper PDF is supplied and that it is not a scanned image."
        )

    return pages


def create_chunks(
    pages: Iterable[PageText],
    chunk_size_words: int = DEFAULT_CHUNK_SIZE_WORDS,
    overlap_words: int = DEFAULT_CHUNK_OVERLAP_WORDS,
) -> list[TextChunk]:
    """Split page text into overlapping chunks without losing page references."""

    if chunk_size_words <= 0:
        raise ValueError("chunk_size_words must be greater than zero.")
    if overlap_words < 0:
        raise ValueError("overlap_words cannot be negative.")
    if overlap_words >= chunk_size_words:
        raise ValueError("overlap_words must be smaller than chunk_size_words.")

    chunks: list[TextChunk] = []
    in_references = False
    for page in pages:
        if in_references:
            continue

        page_text, references_started = remove_reference_section_start(page.text)
        if references_started:
            in_references = True
        if not page_text:
            continue

        words = page_text.split()
        if not words:
            continue

        step = chunk_size_words - overlap_words
        page_chunk_index = 1

        # Overlap keeps nearby context together when an answer spans a chunk
        # boundary, while each chunk still points back to one source page.
        for start in range(0, len(words), step):
            end = start + chunk_size_words
            chunk_words = words[start:end]
            if not chunk_words:
                continue

            chunk_text = " ".join(chunk_words).strip()
            if len(chunk_text) < 20:
                continue

            chunk_id = f"p{page.page_number}-c{page_chunk_index}"
            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    page_number=page.page_number,
                    text=chunk_text,
                )
            )
            page_chunk_index += 1

            if end >= len(words):
                break

    if not chunks:
        raise ValueError("No usable text chunks could be created from the supplied PDF.")

    return chunks


def load_and_chunk_pdf(
    pdf_path: str | Path,
    chunk_size_words: int = DEFAULT_CHUNK_SIZE_WORDS,
    overlap_words: int = DEFAULT_CHUNK_OVERLAP_WORDS,
) -> tuple[list[PageText], list[TextChunk]]:
    """Load the PDF and prepare the chunks used by the CLI and smoke test."""

    pages = load_pdf_pages(pdf_path)
    chunks = create_chunks(
        pages,
        chunk_size_words=chunk_size_words,
        overlap_words=overlap_words,
    )
    return pages, chunks

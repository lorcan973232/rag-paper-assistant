"""Retrieve relevant paper chunks and apply grounding checks."""

from __future__ import annotations

import re
from dataclasses import dataclass

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    DEFAULT_MIN_QUERY_COVERAGE,
    DEFAULT_MIN_QUERY_TERMS_MATCHED,
    DEFAULT_THRESHOLD,
    DEFAULT_TOP_K,
)
from paper_loader import TextChunk


TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9]+")


@dataclass(frozen=True)
class RetrievalResult:
    """A retrieved paper chunk with its similarity score."""

    chunk_id: str
    page_number: int
    text: str
    score: float


class TfidfPaperRetriever:
    """Search the paper with a local TF-IDF representation."""

    def __init__(self, chunks: list[TextChunk]) -> None:
        if not chunks:
            raise ValueError("At least one text chunk is required for retrieval.")

        self.chunks = chunks
        # TF-IDF is used because it is local, transparent, and its scores are
        # easy to inspect.
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            strip_accents="unicode",
            ngram_range=(1, 2),
            max_features=30000,
        )
        self.matrix = self.vectorizer.fit_transform(chunk.text for chunk in chunks)

    def retrieve(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        min_score: float | None = None,
    ) -> list[RetrievalResult]:
        """Return the paper chunks that best match the question."""

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        cleaned_question = question.strip()
        if not cleaned_question:
            return []

        query_vector = self.vectorizer.transform([cleaned_question])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        # Ranking is kept explicit so the source pages and scores can be shown
        # to the user after each answer.
        ranked_indices = scores.argsort()[::-1][:top_k]

        results: list[RetrievalResult] = []
        for index in ranked_indices:
            score = float(scores[index])
            if min_score is not None and score < min_score:
                continue
            chunk = self.chunks[int(index)]
            results.append(
                RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    page_number=chunk.page_number,
                    text=chunk.text,
                    score=score,
                )
            )

        return results


def normalize_token(token: str) -> str:
    """Normalise terms before checking question support."""

    normalized = token.lower()
    # A small singularisation step catches simple plural matches without adding
    # another dependency or changing the retrieval model.
    if len(normalized) > 3 and normalized.endswith("s"):
        normalized = normalized[:-1]
    return normalized


def meaningful_query_terms(question: str) -> set[str]:
    """Keep the question terms that help detect weak accidental matches."""

    terms: set[str] = set()
    for token in TOKEN_PATTERN.findall(question):
        normalized = normalize_token(token)
        # Stop words are removed because they do not show whether the paper
        # actually supports the user's question.
        if normalized in ENGLISH_STOP_WORDS:
            continue
        terms.add(normalized)
    return terms


def has_sufficient_query_support(
    question: str,
    results: list[RetrievalResult],
    min_terms_matched: int = DEFAULT_MIN_QUERY_TERMS_MATCHED,
    min_coverage: float = DEFAULT_MIN_QUERY_COVERAGE,
) -> bool:
    """Check that the retrieved context covers enough of the question."""

    if min_terms_matched <= 0:
        raise ValueError("min_terms_matched must be greater than zero.")
    if not 0 <= min_coverage <= 1:
        raise ValueError("min_coverage must be between 0 and 1.")

    query_terms = meaningful_query_terms(question)
    if not query_terms:
        return False

    context_text = " ".join(result.text for result in results)
    context_terms = {normalize_token(token) for token in TOKEN_PATTERN.findall(context_text)}
    matched_terms = query_terms & context_terms

    # This second check reduces accidental matches, such as a current-affairs
    # question that happens to share one generic word with the paper.
    required_terms = min(min_terms_matched, len(query_terms))
    coverage = len(matched_terms) / len(query_terms)
    return len(matched_terms) >= required_terms and coverage >= min_coverage


def should_refuse(
    results: list[RetrievalResult],
    threshold: float = DEFAULT_THRESHOLD,
    question: str | None = None,
) -> bool:
    """Decide whether the answer should be refused before generation."""

    if threshold < 0:
        raise ValueError("threshold cannot be negative.")
    if not results:
        return True
    if max(result.score for result in results) < threshold:
        return True

    # Only chunks above the score threshold are allowed to support the question.
    eligible_results = [result for result in results if result.score >= threshold]
    if question is not None and not has_sufficient_query_support(question, eligible_results):
        return True
    return False


def source_pages(results: list[RetrievalResult]) -> list[int]:
    """Return the source pages in a stable order."""

    return sorted({result.page_number for result in results})


def format_context(results: list[RetrievalResult]) -> str:
    """Format retrieved chunks as the only context sent to the model."""

    sections: list[str] = []
    for result in results:
        # Page and score metadata are included in the prompt for transparency,
        # but the CLI remains responsible for displaying final source pages.
        sections.append(
            "\n".join(
                [
                    f"[Source: page {result.page_number}, chunk {result.chunk_id}, score {result.score:.3f}]",
                    result.text,
                ]
            )
        )
    return "\n\n---\n\n".join(sections)


def format_sources(results: list[RetrievalResult]) -> str:
    """Format source pages for display in the CLI."""

    pages = source_pages(results)
    if not pages:
        return "Sources:\n- No sufficiently relevant source pages found."
    return "Sources:\n" + "\n".join(f"- Page {page}" for page in pages)

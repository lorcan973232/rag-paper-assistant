"""Tests for the non-Ollama parts of the paper assistant.

The tests use small in-memory page samples so they remain fast and do not
depend on an installed local model.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import ollama_client
import main
from config import (
    DEFAULT_CHUNK_OVERLAP_WORDS,
    DEFAULT_CHUNK_SIZE_WORDS,
    DEFAULT_MIN_QUERY_COVERAGE,
    DEFAULT_MIN_QUERY_TERMS_MATCHED,
    DEFAULT_MODEL_NAME,
    DEFAULT_OLLAMA_TIMEOUT_SECONDS,
    DEFAULT_PDF_PATH,
    DEFAULT_THRESHOLD,
    DEFAULT_TOP_K,
    DEMO_MODEL_NAME,
    PAPER_ONLY_SYSTEM_PROMPT,
    REFUSAL_MESSAGE,
)
from ollama_client import OllamaClient, build_ollama_prompt
from paper_loader import PageText, clean_text, create_chunks, validate_pdf_path
from retriever import (
    TfidfPaperRetriever,
    format_context,
    has_sufficient_query_support,
    should_refuse,
)


def sample_pages() -> list[PageText]:
    """Create a small paper-like sample for retrieval and chunking tests."""

    return [
        PageText(
            page_number=1,
            text=(
                "Large Language Models support language understanding and generation. "
                "Agentic AI extends language model capability by planning actions, using tools, "
                "interacting with external APIs, and pursuing goals with more autonomy."
            ),
        ),
        PageText(
            page_number=2,
            text=(
                "The paper identifies challenges for LLMs and Agentic AI including privacy, "
                "security, misinformation, bias, trust, transparency, explainability, memory, "
                "human oversight, multi-agent coordination, accountability, and sustainability."
            ),
        ),
        PageText(
            page_number=3,
            text=(
                "Future research directions include better value alignment, reliable tool use, "
                "stronger evaluation, improved governance, and responsible deployment of agentic systems."
            ),
        ),
    ]


def build_retriever() -> TfidfPaperRetriever:
    """Build a retriever from the sample pages used in unit tests."""

    chunks = create_chunks(sample_pages(), chunk_size_words=35, overlap_words=5)
    return TfidfPaperRetriever(chunks)


def test_pdf_path_validation_missing_file(tmp_path: Path) -> None:
    missing_pdf = tmp_path / "missing.pdf"
    with pytest.raises(FileNotFoundError):
        validate_pdf_path(missing_pdf)


def test_text_cleaning_returns_readable_text() -> None:
    raw_text = "Agentic-\nAI   uses\n\nexternal   tools  .\x00"
    cleaned = clean_text(raw_text)

    assert "\n" not in cleaned
    assert "  " not in cleaned
    assert cleaned.endswith("tools.")
    assert "AgenticAI" in cleaned


def test_chunking_creates_non_empty_chunks() -> None:
    chunks = create_chunks(sample_pages(), chunk_size_words=30, overlap_words=5)

    assert chunks
    assert all(chunk.text for chunk in chunks)
    assert all(chunk.chunk_id for chunk in chunks)


def test_chunks_preserve_page_numbers() -> None:
    chunks = create_chunks(sample_pages(), chunk_size_words=25, overlap_words=5)
    page_numbers = {chunk.page_number for chunk in chunks}

    assert page_numbers == {1, 2, 3}


def test_reference_section_is_excluded_from_chunks() -> None:
    """The references section should not influence retrieval tests."""

    pages = [
        PageText(page_number=1, text="Agentic AI uses tools and planning in the paper body."),
        PageText(
            page_number=2,
            text="References 1. Example, A. Large Language Model paper title that should not be retrieved.",
        ),
        PageText(page_number=3, text="This page appears after references and should not become a chunk."),
    ]

    chunks = create_chunks(pages, chunk_size_words=20, overlap_words=2)
    combined = " ".join(chunk.text for chunk in chunks)

    assert "References" not in combined
    assert "should not be retrieved" not in combined
    assert {chunk.page_number for chunk in chunks} == {1}


def test_retrieval_returns_structured_results() -> None:
    retriever = build_retriever()
    results = retriever.retrieve("What challenges do LLMs introduce into Agentic AI systems?", top_k=2)

    assert results
    first = results[0]
    assert isinstance(first.chunk_id, str)
    assert isinstance(first.page_number, int)
    assert isinstance(first.text, str)
    assert isinstance(first.score, float)
    assert "Source: page" in format_context(results)


def test_relevant_question_scores_above_unrelated_question() -> None:
    retriever = build_retriever()

    relevant = retriever.retrieve("What are Agentic AI challenges and tool use issues?", top_k=1)
    unrelated = retriever.retrieve("How do I bake chocolate cake with bananas?", top_k=1)

    relevant_score = relevant[0].score if relevant else 0.0
    unrelated_score = unrelated[0].score if unrelated else 0.0

    assert relevant_score > unrelated_score
    assert relevant_score > 0


def test_low_confidence_question_triggers_refusal_logic() -> None:
    retriever = build_retriever()
    question = "How do I bake chocolate cake with bananas?"
    results = retriever.retrieve(question, top_k=3)

    assert should_refuse(results, threshold=0.2, question=question)
    assert REFUSAL_MESSAGE == "I could not find enough information in the provided paper to answer that question."


def test_accidental_keyword_match_is_refused_by_query_support() -> None:
    """A weak shared keyword should not be enough to call the model."""

    retriever = build_retriever()
    question = "What is the stock price of Apple today?"
    results = retriever.retrieve(question, top_k=3)

    assert not has_sufficient_query_support(question, results)
    assert should_refuse(results, threshold=0.01, question=question)


def test_relevant_question_has_query_support() -> None:
    retriever = build_retriever()
    question = "How does Agentic AI use tools and external APIs?"
    results = retriever.retrieve(question, top_k=3)

    assert has_sufficient_query_support(question, results)
    assert not should_refuse(results, threshold=0.01, question=question)


def test_ollama_prompt_contains_only_supplied_context() -> None:
    prompt = build_ollama_prompt(
        question="What are the challenges?",
        paper_context="Paper-only context about privacy, trust, and explainability.",
    )

    assert "Paper-only context about privacy, trust, and explainability." in prompt
    assert "Use the paper context below" in prompt
    assert "outside knowledge" not in prompt.lower()


def test_system_prompt_contains_required_grounding_rule() -> None:
    assert "You must answer only using the provided paper context." in PAPER_ONLY_SYSTEM_PROMPT
    assert "Do not use outside knowledge." in PAPER_ONLY_SYSTEM_PROMPT
    assert REFUSAL_MESSAGE in PAPER_ONLY_SYSTEM_PROMPT


def test_ollama_client_posts_local_non_streaming_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Ollama client should send a local, non-streamed generation request."""

    captured: dict[str, object] = {}

    class FakeResponse:
        status_code = 200
        text = ""

        def json(self) -> dict[str, str]:
            return {"response": "Direct answer: grounded response"}

    def fake_post(url: str, json: dict[str, object], timeout: int) -> FakeResponse:
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(ollama_client.requests, "post", fake_post)

    client = OllamaClient(model="llama3", base_url="http://localhost:11434", timeout_seconds=7)
    answer = client.generate("Prompt")

    assert answer == "Direct answer: grounded response"
    assert captured["url"] == "http://localhost:11434/api/generate"
    payload = captured["json"]
    assert isinstance(payload, dict)
    assert payload["model"] == "llama3"
    assert payload["stream"] is False
    assert "provided paper context" in str(payload["system"])


def test_help_lists_rationale_commands(capsys: pytest.CaptureFixture[str]) -> None:
    main.print_help()
    output = capsys.readouterr().out

    for command in ["stack", "why", "architecture", "files", "settings", "prompt"]:
        assert command in output


def test_technology_stack_text_can_be_produced() -> None:
    output = main.build_stack_text()

    assert "Python" in output
    assert "pypdf" in output
    assert "scikit-learn" in output
    assert "Ollama" in output
    assert "pytest" in output
    assert DEFAULT_MODEL_NAME in output
    assert DEMO_MODEL_NAME in output


def test_settings_output_matches_config_values() -> None:
    args = main.parse_args([])
    output = main.build_settings_text(args)

    assert f"Model name: {DEFAULT_MODEL_NAME}" in output
    assert f"PDF path: {DEFAULT_PDF_PATH}" in output
    assert f"Chunk size: {DEFAULT_CHUNK_SIZE_WORDS} words" in output
    assert f"Chunk overlap: {DEFAULT_CHUNK_OVERLAP_WORDS} words" in output
    assert f"Top-k retrieved chunks: {DEFAULT_TOP_K}" in output
    assert f"Similarity threshold: {DEFAULT_THRESHOLD}" in output
    assert f"Ollama timeout: {DEFAULT_OLLAMA_TIMEOUT_SECONDS} seconds" in output


def test_prompt_command_explains_grounding_rule() -> None:
    output = main.build_prompt_text()

    assert "provided paper context" in output
    assert "Do not use outside knowledge" in output
    assert REFUSAL_MESSAGE in output


def test_architecture_and_files_text_explain_project_structure() -> None:
    architecture = main.build_architecture_text()
    files = main.build_files_text()

    assert "TF-IDF matrix" in architecture
    assert "only the retrieved paper chunks" in architecture
    for filename in [
        "main.py",
        "config.py",
        "paper_loader.py",
        "retriever.py",
        "ollama_client.py",
        "smoke_test.py",
        "tests/test_core.py",
    ]:
        assert filename in files


def test_configuration_defaults_are_valid() -> None:
    assert DEFAULT_PDF_PATH.endswith(".pdf")
    assert DEFAULT_TOP_K > 0
    assert 0 <= DEFAULT_THRESHOLD <= 1
    assert DEFAULT_MIN_QUERY_TERMS_MATCHED > 0
    assert 0 <= DEFAULT_MIN_QUERY_COVERAGE <= 1
    assert DEFAULT_CHUNK_SIZE_WORDS == 260
    assert DEFAULT_CHUNK_OVERLAP_WORDS == 60
    assert DEFAULT_OLLAMA_TIMEOUT_SECONDS == 90

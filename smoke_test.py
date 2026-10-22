"""Check the paper pipeline without requiring Ollama.

Run with:
    python smoke_test.py
"""

from __future__ import annotations

from pathlib import Path

from config import DEFAULT_PDF_PATH, DEFAULT_THRESHOLD
from main import build_architecture_text, build_settings_text, build_stack_text, parse_args
from paper_loader import create_chunks, load_pdf_pages
from retriever import TfidfPaperRetriever, should_refuse


def print_result(name: str, passed: bool, detail: str = "") -> None:
    """Print one smoke-test check as a PASS or FAIL line."""

    status = "PASS" if passed else "FAIL"
    suffix = f" - {detail}" if detail else ""
    print(f"{name}: {status}{suffix}")


def main() -> int:
    """Run quick checks for the document and retrieval pipeline."""

    print("Smoke Test Results")

    results: list[bool] = []
    pdf_path = Path(DEFAULT_PDF_PATH)

    pdf_exists = pdf_path.exists()
    print_result("PDF loading", pdf_exists, "" if pdf_exists else f"Missing {pdf_path}")
    results.append(pdf_exists)
    if not pdf_exists:
        # The smoke test fails early if the paper PDF is not in place.
        print("\nOverall: FAIL")
        print(f"Place the paper PDF at: {pdf_path.resolve()}")
        return 1

    try:
        pages = load_pdf_pages(pdf_path)
        text_extracted = bool(pages and sum(len(page.text) for page in pages) > 200)
        print_result("Text extraction", text_extracted, f"{len(pages)} pages extracted")
        results.append(text_extracted)
    except Exception as exc:
        print_result("Text extraction", False, str(exc))
        print("\nOverall: FAIL")
        return 1

    try:
        chunks = create_chunks(pages)
        chunks_created = bool(chunks)
        page_numbers_preserved = all(chunk.page_number >= 1 for chunk in chunks)
        print_result("Chunking", chunks_created, f"{len(chunks)} chunks created")
        print_result("Page-number preservation", page_numbers_preserved)
        results.extend([chunks_created, page_numbers_preserved])
    except Exception as exc:
        print_result("Chunking", False, str(exc))
        print("\nOverall: FAIL")
        return 1

    try:
        retriever = TfidfPaperRetriever(chunks)
        # A known paper-specific question should retrieve body text above the
        # confidence threshold.
        relevant_question = "What is the difference between Large Language Models and Agentic AI?"
        relevant_results = retriever.retrieve(relevant_question, top_k=5)
        relevant_score = max((result.score for result in relevant_results), default=0.0)
        retrieval_passed = bool(
            relevant_results
            and relevant_score >= DEFAULT_THRESHOLD
            and not should_refuse(relevant_results, threshold=DEFAULT_THRESHOLD, question=relevant_question)
        )
        print_result("Retrieval", retrieval_passed, f"best score {relevant_score:.3f}")
        results.append(retrieval_passed)

        # This unrelated question checks that accidental keyword overlap does
        # not lead to an unsupported model answer.
        unrelated_question = "What is the stock price of Apple today?"
        unrelated_results = retriever.retrieve(unrelated_question, top_k=5)
        refusal_triggered = should_refuse(
            unrelated_results,
            threshold=DEFAULT_THRESHOLD,
            question=unrelated_question,
        )
        unrelated_score = max((result.score for result in unrelated_results), default=0.0)
        print_result(
            "Unsupported-question threshold",
            refusal_triggered,
            f"best score {unrelated_score:.3f}",
        )
        results.append(refusal_triggered)
        print_result("Ollama requirement", True, "not required for smoke test")
        results.append(True)

        stack_text = build_stack_text()
        settings_text = build_settings_text(parse_args([]))
        architecture_text = build_architecture_text()
        rationale_output = all(
            [
                "Python" in stack_text,
                "Ollama" in stack_text,
                "Chunk size: 260 words" in settings_text,
                "Similarity threshold: 0.08" in settings_text,
                "TF-IDF matrix" in architecture_text,
            ]
        )
        print_result("Rationale commands", rationale_output)
        results.append(rationale_output)
    except Exception as exc:
        print_result("Retrieval", False, str(exc))
        print("\nOverall: FAIL")
        return 1

    overall = all(results)
    print(f"\nOverall: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())

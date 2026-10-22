"""Run the paper-only command-line assistant."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config import (
    DEFAULT_CHUNK_OVERLAP_WORDS,
    DEFAULT_CHUNK_SIZE_WORDS,
    DEFAULT_MODEL_NAME,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_TIMEOUT_SECONDS,
    DEFAULT_PDF_PATH,
    DEFAULT_THRESHOLD,
    DEFAULT_TOP_K,
    DEMO_MODEL_NAME,
    EXAMPLE_QUESTIONS,
    PAPER_ONLY_SYSTEM_PROMPT,
    REFUSAL_MESSAGE,
)

try:
    # Import dependency-heavy modules after the standard library imports so a
    # missing package gives a clear installation message.
    from ollama_client import OllamaClient, OllamaError, build_ollama_prompt
    from paper_loader import load_and_chunk_pdf
    from retriever import (
        RetrievalResult,
        TfidfPaperRetriever,
        format_context,
        format_sources,
        should_refuse,
    )
except ModuleNotFoundError as exc:
    missing_dependency = exc.name or "unknown"
    print(
        f"Missing Python dependency: {missing_dependency}. "
        "Install dependencies with: pip install -r requirements.txt",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the command-line options."""

    parser = argparse.ArgumentParser(
        description="Ask questions about the Agentic AI and LLM academic paper using local Ollama.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pdf",
        default=DEFAULT_PDF_PATH,
        help="Path to the paper PDF.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help="Ollama model name.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of retrieved paper chunks to use for each question.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Minimum retrieval confidence required before calling Ollama.",
    )
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Print retrieved paper snippets before generating the answer.",
    )
    parser.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_BASE_URL,
        help="Base URL for the local Ollama server.",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    """Reject unsafe CLI values before the PDF or model is loaded."""

    if args.top_k <= 0:
        raise ValueError("--top-k must be greater than zero.")
    if args.threshold < 0:
        raise ValueError("--threshold cannot be negative.")
    if args.threshold > 1:
        raise ValueError("--threshold should be between 0 and 1 for cosine similarity.")


def print_banner(pdf_path: str, model: str, pages_count: int, chunks_count: int) -> None:
    """Show which paper and local model are being used."""

    print("\nAgentic AI Paper Assistant")
    print("Ask a question about the provided paper.")
    print("Type 'help' for commands or 'exit' to quit.\n")
    print(f"PDF: {Path(pdf_path).name}")
    print(f"Model: {model}")
    print(f"Loaded: {pages_count} pages, {chunks_count} chunks\n")


def print_help() -> None:
    """Show the interactive commands supported by the CLI."""

    print(
        """
Commands:
- help          Show this help message.
- examples      Show example questions.
- stack         Show the technology stack.
- why           Explain the main design choices.
- architecture  Explain how the system works step by step.
- files         Explain what each project file does.
- settings      Show the current model, PDF, retrieval, and Ollama settings.
- prompt        Show the grounding rule used in the system prompt.
- sources       Show source pages and retrieval scores from the last question.
- clear         Add spacing to clear the visible console area.
- exit          Quit the application.
- quit          Quit the application.

Ask any normal question about the paper at the prompt.
""".strip()
    )


def build_stack_text() -> str:
    """Explain the technology stack."""

    return f"""
Technology stack:
- Python: runs the command-line application and connects the PDF, retrieval, and Ollama steps.
- pypdf: extracts text page by page from the academic PDF.
- scikit-learn: builds the TF-IDF matrix and cosine-similarity retrieval system.
- requests: sends local HTTP requests to the Ollama API.
- Ollama: runs the local language model without a paid or cloud API.
- pytest: tests the non-Ollama parts of the assistant.

Default model: {DEFAULT_MODEL_NAME}
Lightweight model for limited hardware: {DEMO_MODEL_NAME}
""".strip()


def build_rationale_text() -> str:
    """Explain why the main technical choices were made."""

    return f"""
Design rationale:
- Python was chosen because it is well suited to NLP, PDF processing, testing, and CLI applications.
- RAG was chosen because the task is document question answering. The system must find relevant paper sections before asking the model to answer.
- TF-IDF was chosen instead of neural embeddings because the project uses one paper, and TF-IDF is local, transparent, fast, and easy to explain.
- No neural embedding model or vector database is used. The chunks are stored in memory, and TF-IDF creates a numerical matrix for searching.
- Ollama was chosen because it runs the model locally and avoids paid APIs.
- The default threshold is {DEFAULT_THRESHOLD}. If retrieval confidence is too low, the app refuses before calling Ollama.
""".strip()


def build_architecture_text() -> str:
    """Describe the pipeline from PDF loading to grounded answer generation."""

    return f"""
Architecture:
1. The user starts the CLI and supplies the PDF path and model name.
2. pypdf extracts the paper text page by page.
3. The text is cleaned and the bibliography section is excluded from retrieval chunks.
4. The paper is split into {DEFAULT_CHUNK_SIZE_WORDS}-word chunks with {DEFAULT_CHUNK_OVERLAP_WORDS} words of overlap.
5. Each chunk is stored in memory with a chunk ID, page number, and text.
6. scikit-learn converts the chunks into a TF-IDF matrix.
7. Each user question is converted into a TF-IDF vector and compared with the chunk matrix.
8. If the best retrieval evidence is below the threshold, the system refuses the question.
9. If retrieval passes, only the retrieved paper chunks are sent to Ollama.
10. Ollama generates an answer under a paper-only system prompt.
11. The CLI prints the answer and source page references.
""".strip()


def build_files_text() -> str:
    """Explain the purpose of each project file."""

    return """
Project files:
- main.py: runs the interactive CLI, handles commands, retrieves context, calls Ollama, and prints answers.
- config.py: stores default settings, example questions, and the paper-only system prompt.
- paper_loader.py: validates the PDF, extracts page text, cleans it, removes references from retrieval chunks, and creates chunks.
- retriever.py: builds the TF-IDF retriever, calculates similarity scores, formats context, and applies refusal logic.
- ollama_client.py: sends local non-streaming generation requests to the Ollama HTTP API and handles Ollama errors.
- smoke_test.py: checks PDF loading, extraction, chunking, retrieval, and refusal logic without requiring Ollama.
- tests/test_core.py: provides automated pytest checks for the non-Ollama core behaviour.
- requirements.txt: lists the Python dependencies needed to run and test the assistant.
- README.md: explains setup, use, testing, grounding, and design rationale.
- TECHNICAL_RATIONALE.md: gives a question-and-answer explanation of the technology choices.
""".strip()


def build_settings_text(args: argparse.Namespace, client: OllamaClient | None = None) -> str:
    """Show the active settings that affect retrieval and generation."""

    timeout = client.timeout_seconds if client is not None else DEFAULT_OLLAMA_TIMEOUT_SECONDS
    return f"""
Current settings:
- Model name: {args.model}
- PDF path: {args.pdf}
- Chunk size: {DEFAULT_CHUNK_SIZE_WORDS} words
- Chunk overlap: {DEFAULT_CHUNK_OVERLAP_WORDS} words
- Top-k retrieved chunks: {args.top_k}
- Similarity threshold: {args.threshold}
- Ollama URL: {args.ollama_url}
- Ollama timeout: {timeout} seconds
""".strip()


def build_prompt_text() -> str:
    """Summarise the prompt engineering used for paper-only grounding."""

    return f"""
Prompt grounding:
The system uses prompt engineering to restrict the local model to retrieved paper context.

Main grounding rule:
"You must answer only using the provided paper context. Do not use outside knowledge. If the answer is not supported by the context, say: '{REFUSAL_MESSAGE}'"

The full system prompt also tells the model to avoid unsupported claims, keep the answer concise, and not invent page numbers.
""".strip()


def print_examples() -> None:
    """Print paper-focused questions that are suitable for demonstration."""

    print("Example questions:")
    for index, question in enumerate(EXAMPLE_QUESTIONS, start=1):
        print(f"{index}. {question}")


def print_last_sources(
    last_results: list[RetrievalResult],
    threshold: float,
    used_for_answer: bool,
) -> None:
    """Show how the last answer was grounded in retrieved paper chunks."""

    if not last_results:
        print("No retrieval has been run yet.")
        return

    if not used_for_answer:
        print("No source pages were used in the last answer.")
    print("Last retrieval results:")
    for result in last_results:
        if used_for_answer and result.score >= threshold:
            status = "used"
        elif result.score >= threshold:
            status = "retrieved but not used"
        else:
            status = "below threshold"
        snippet = result.text[:220].replace("\n", " ")
        if len(result.text) > 220:
            snippet += "..."
        print(f"- Page {result.page_number}, {result.chunk_id}, score {result.score:.3f}, {status}")
        print(f"  {snippet}")


def print_debug_context(results: list[RetrievalResult]) -> None:
    """Print retrieved snippets when the user wants to inspect grounding."""

    print("\nRetrieved context:")
    for result in results:
        snippet = result.text[:700].replace("\n", " ")
        if len(result.text) > 700:
            snippet += "..."
        print(f"\n[Page {result.page_number} | {result.chunk_id} | score {result.score:.3f}]")
        print(snippet)


def retrieve_grounded_results(
    question: str,
    retriever: TfidfPaperRetriever,
    top_k: int,
    threshold: float,
) -> tuple[list[RetrievalResult], list[RetrievalResult], bool]:
    """Retrieve paper chunks and decide whether there is enough support to answer."""

    candidate_results = retriever.retrieve(question, top_k=top_k)
    # This is the first grounding gate. If retrieval is weak, the local model is
    # not called, which prevents it from answering from general knowledge.
    if should_refuse(candidate_results, threshold=threshold, question=question):
        return candidate_results, [], False

    grounded_results = [result for result in candidate_results if result.score >= threshold]
    return candidate_results, grounded_results, True


def answer_question(
    question: str,
    retriever: TfidfPaperRetriever,
    client: OllamaClient,
    top_k: int,
    threshold: float,
    show_context: bool,
) -> tuple[str, list[RetrievalResult], bool]:
    """Answer one question using retrieved paper context only."""

    candidate_results, grounded_results, can_answer = retrieve_grounded_results(
        question=question,
        retriever=retriever,
        top_k=top_k,
        threshold=threshold,
    )
    if not can_answer:
        return REFUSAL_MESSAGE, candidate_results, False

    if show_context:
        print_debug_context(grounded_results)

    # Only these retrieved chunks are sent to Ollama. The full paper and any
    # outside source are never passed to the model.
    paper_context = format_context(grounded_results)
    prompt = build_ollama_prompt(question=question, paper_context=paper_context)
    answer = client.generate(prompt=prompt, system_prompt=PAPER_ONLY_SYSTEM_PROMPT)
    return answer, grounded_results, True


def run_cli(args: argparse.Namespace) -> int:
    """Load the paper once, then run the interactive question loop."""

    validate_args(args)

    # The PDF is processed before the prompt loop so each question can reuse the
    # same in-memory retriever during a live demonstration.
    pages, chunks = load_and_chunk_pdf(args.pdf)
    retriever = TfidfPaperRetriever(chunks)
    client = OllamaClient(model=args.model, base_url=args.ollama_url)

    print_banner(args.pdf, args.model, len(pages), len(chunks))
    last_results: list[RetrievalResult] = []
    last_results_used_for_answer = False

    while True:
        try:
            user_input = input("Question: ").strip()
        except KeyboardInterrupt:
            print("\nExiting.")
            return 0
        except EOFError:
            print("\nExiting.")
            return 0

        if not user_input:
            print("Please enter a question or command. Type 'help' for commands.")
            continue

        command = user_input.lower()
        if command in {"exit", "quit"}:
            print("Goodbye.")
            return 0
        if command == "help":
            print_help()
            continue
        if command == "examples":
            print_examples()
            continue
        if command == "stack":
            print(build_stack_text())
            continue
        if command == "why":
            print(build_rationale_text())
            continue
        if command == "architecture":
            print(build_architecture_text())
            continue
        if command == "files":
            print(build_files_text())
            continue
        if command == "settings":
            print(build_settings_text(args, client))
            continue
        if command == "prompt":
            print(build_prompt_text())
            continue
        if command == "sources":
            print_last_sources(last_results, args.threshold, last_results_used_for_answer)
            continue
        if command == "clear":
            print("\n" * 40)
            continue

        print()
        # Retrieval is repeated for each question because the relevant paper
        # pages depend on the user's wording.
        candidate_results, grounded_results, can_answer = retrieve_grounded_results(
            question=user_input,
            retriever=retriever,
            top_k=args.top_k,
            threshold=args.threshold,
        )
        last_results = candidate_results
        last_results_used_for_answer = False
        if not can_answer:
            # Unsupported questions are refused before generation. The stored
            # retrieval results can still be inspected with the `sources` command.
            print("Answer:")
            print(REFUSAL_MESSAGE)
            print()
            print("Sources:")
            print("- No sufficiently relevant source pages found.")
            print()
            continue

        last_results = grounded_results
        last_results_used_for_answer = False
        if args.show_context:
            print_debug_context(grounded_results)

        try:
            # The generation prompt is built only after retrieval has passed the
            # confidence checks.
            paper_context = format_context(grounded_results)
            prompt = build_ollama_prompt(question=user_input, paper_context=paper_context)
            answer = client.generate(prompt=prompt, system_prompt=PAPER_ONLY_SYSTEM_PROMPT)
        except OllamaError as exc:
            # A failed local model should not hide the fact that retrieval worked.
            print("Ollama error:")
            print(f"{exc}")
            print(f"Install the model with: ollama pull {args.model}")
            print("The retrieval step completed, but answer generation requires local Ollama.\n")
            print(format_sources(last_results))
            print()
            continue

        print("Answer:")
        print(answer)
        print()
        last_results_used_for_answer = True
        print(format_sources(last_results))
        print()


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and convert common setup errors into clear messages."""

    args = parse_args(argv)
    try:
        return run_cli(args)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print(
            f"Place the paper PDF in this folder or pass its path with --pdf \"{DEFAULT_PDF_PATH}\".",
            file=sys.stderr,
        )
        return 1
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nExiting.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

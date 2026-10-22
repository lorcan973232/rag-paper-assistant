"""Project settings for the Agentic AI paper assistant."""

from __future__ import annotations

# These defaults keep the demonstration simple: the expected PDF can sit beside
# the scripts, and the model can be changed from the command line if needed.
DEFAULT_PDF_PATH = "Research Landscape of Agentic AI and LLM - Apps, Challenges and Future Direction.pdf"
DEFAULT_MODEL_NAME = "llama3"
DEMO_MODEL_NAME = "qwen2.5:0.5b"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_TIMEOUT_SECONDS = 90

# Retrieval settings are deliberately conservative. The threshold and query
# coverage check help the application refuse questions that are not supported
# by the paper before any local model is called.
DEFAULT_TOP_K = 5
DEFAULT_THRESHOLD = 0.08
DEFAULT_MIN_QUERY_TERMS_MATCHED = 2
DEFAULT_MIN_QUERY_COVERAGE = 0.25
DEFAULT_CHUNK_SIZE_WORDS = 260
DEFAULT_CHUNK_OVERLAP_WORDS = 60
MIN_EXTRACTED_CHARACTERS = 200

REFUSAL_MESSAGE = "I could not find enough information in the provided paper to answer that question."

# This prompt is part of the grounding strategy. It tells Ollama that the
# retrieved paper context is the only source that may be used.
PAPER_ONLY_SYSTEM_PROMPT = f"""
You are an academic question-answering assistant for one specific paper.

You must answer only using the provided paper context. Do not use outside knowledge. If the answer is not supported by the context, say: '{REFUSAL_MESSAGE}'

Rules:
- Use only the provided paper context.
- Do not add facts, examples, definitions, or assumptions from outside the context.
- If the retrieved context is insufficient, use the refusal sentence exactly.
- If only part of an answer is supported, answer only the supported part and explicitly avoid unsupported claims.
- Keep the answer concise, academic, and grounded.
- Include a brief supporting-evidence paragraph based on the context.
- Do not invent page numbers; the application displays source pages separately.
""".strip()

# The same examples are shown in the CLI, and the recorded run in
# docs/screenshots answers each of them.
EXAMPLE_QUESTIONS = [
    "What is the difference between Large Language Models and Agentic AI?",
    "What application domains of Agentic AI are discussed in the paper?",
    "What challenges do LLMs introduce into Agentic AI systems?",
    "What are the Agentic AI-specific challenges identified by the authors?",
    "How does the paper explain the role of external tools and APIs in Agentic AI?",
    "What future research directions does the paper propose?",
    "Why are transparency and explainability important for LLMs and Agentic AI?",
]

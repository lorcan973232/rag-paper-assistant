# Technical Rationale

## What technology stack was used?

The project uses Python, `pypdf`, `scikit-learn`, `requests`, Ollama, and `pytest`.

Python runs the CLI and connects the different parts of the system. `pypdf` reads the academic PDF page by page. `scikit-learn` creates the TF-IDF retrieval system. `requests` calls the local Ollama HTTP API. Ollama runs the local language model. `pytest` checks the core non-Ollama behaviour.

## Why Python?

Python is suitable for NLP and text-processing tasks. It has strong support for PDF processing, retrieval, testing, and local LLM integration, and it keeps a command-line tool simple to install and run.

## Why RAG?

The task is question answering over an academic paper. The system needs to find relevant paper sections, pass them to a language model, and show source pages. Retrieval-augmented generation is designed for this type of document question answering.

## Why TF-IDF instead of embeddings?

TF-IDF is simple, local, transparent, and easy to test. The tool works on one paper, so a neural embedding model would add complexity without much benefit. TF-IDF gives numerical similarity scores that are easy to inspect and explain.

Embeddings can capture deeper semantic meaning, but they need an additional model and are harder to inspect. For a single paper, TF-IDF is enough and easier to audit.

## Did the system use embeddings?

No neural embeddings were used.

The system uses TF-IDF numerical word-based representations. The paper chunks and user question are converted into TF-IDF vectors. This is not the same as a neural embedding model.

## How is the text stored?

The text is extracted from the PDF page by page. It is cleaned and split into overlapping chunks. Each chunk is stored in memory with a chunk ID, page number, and text. `scikit-learn` then converts the chunk text into a TF-IDF matrix for retrieval.

## Why in-memory storage instead of a database?

The project uses one PDF, and the amount of text is small. A database would add setup work without benefit at this scale. In-memory storage is simpler and faster.

## Is this a vector database?

No. The project does not use a vector database or any external database. TF-IDF creates a numerical matrix for searching, but this matrix is held in memory and is not a database.

## Why Ollama?

The project requires a locally hosted LLM. Ollama allows local model execution through a local HTTP API. It avoids paid APIs, API keys, and cloud-based runtime dependencies.

## What model was used?

The default model in the code is `llama3`, a strong general-purpose model. The CLI also supports `--model`, so another installed Ollama model can be selected.

For limited hardware, `qwen2.5:0.5b` is a useful lightweight option. It has approximately 0.5 billion parameters, which is around 500 million. If another model is selected, the parameter count depends on that model.

## Did the project use prompt engineering?

Yes. The system prompt tells the model to answer only from the retrieved paper context. It tells the model not to use outside knowledge and to refuse unsupported questions. This reduces hallucination and keeps the answer tied to the paper.

The main grounding rule is:

```text
You must answer only using the provided paper context. Do not use outside knowledge. If the answer is not supported by the context, say: 'I could not find enough information in the provided paper to answer that question.'
```

## Why pypdf?

`pypdf` extracts text from PDFs page by page. Page-by-page extraction is important because the CLI displays source page references, which let a reader check where an answer came from.

## Why chunking?

The whole paper is too large and too unfocused to send to the model for every question. Chunking allows the system to retrieve only the most relevant parts of the paper. This improves grounding and reduces prompt length.

The project uses 260-word chunks with a 60-word overlap. The chunk size gives enough context for a section, while the overlap reduces the chance of losing important information at chunk boundaries.

Large chunks can include too much unrelated text and make retrieval less focused. Very small chunks can lose context and make the answer depend on fragmented evidence.

## How does TF-IDF scoring work?

TF-IDF uses words, but it converts them into numerical weights. Important terms receive higher weights. Very common terms receive lower weight or are ignored. The system compares the question vector with chunk vectors and produces cosine-similarity scores.

## What threshold is used?

The default similarity threshold is `0.08`. If the retrieved evidence is below the threshold, or if the retrieved text does not contain enough meaningful support for the question, the application refuses before calling Ollama.

The refusal message is:

```text
I could not find enough information in the provided paper to answer that question.
```

## What timeout is used?

The Ollama timeout is 90 seconds. Local models can be slower than cloud APIs, especially on limited hardware. The timeout prevents the program from hanging forever and gives a clear error if Ollama takes too long.

## What does each file do?

`main.py` runs the interactive CLI, handles commands, retrieves context, calls Ollama, and prints answers.

`config.py` stores default settings, example questions, and the paper-only system prompt.

`paper_loader.py` validates the PDF, extracts page text, cleans it, removes references from retrieval chunks, and creates chunks.

`retriever.py` builds the TF-IDF retriever, calculates similarity scores, formats context, and applies refusal logic.

`ollama_client.py` sends local generation requests to the Ollama HTTP API and handles Ollama errors.

`smoke_test.py` checks PDF loading, extraction, chunking, retrieval, refusal logic, and the CLI's explanation commands without requiring Ollama.

`tests/test_core.py` provides automated pytest checks for the non-Ollama core behaviour.

`requirements.txt` lists the Python dependencies needed to run and test the assistant.

`README.md` explains setup, use, testing, grounding, and design rationale.

`TECHNICAL_RATIONALE.md` answers the main design questions in more detail.

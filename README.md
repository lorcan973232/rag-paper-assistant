# Paper-Grounded RAG Assistant on a Local LLM

A command-line assistant that answers questions about one academic paper using
only that paper. It retrieves passages with TF-IDF, refuses before calling the
model when the paper does not support the question, and generates answers with
a locally hosted model through Ollama, citing the source pages. No cloud API,
paid API, database or outside knowledge source is used at runtime.

This was an individual MSc assignment at Atlantic Technological University
(ATU) Donegal, 2026. The paper is Brohi et al. (2025), *A Research Landscape of
Agentic AI and Large Language Models* (see [Paper](#paper)).

## Results

**Smoke test** (`python smoke_test.py`, no model needed):

```text
PDF loading: PASS
Text extraction: PASS - 29 pages extracted
Chunking: PASS - 57 chunks created
Page-number preservation: PASS
Retrieval: PASS - best score 0.266
Unsupported-question threshold: PASS - best score 0.115
Ollama requirement: PASS - not required for smoke test
Rationale commands: PASS

Overall: PASS
```

The in-scope question ("What is the difference between Large Language Models
and Agentic AI?") retrieves body text scoring 0.266 and is answered. The
off-topic question ("What is the stock price of Apple today?") still scores
0.115, above the 0.08 similarity threshold, because it shares common words with
the paper. The query-support check refuses it, since too few of its meaningful
terms appear in the retrieved text.

**Unit tests** (`python -m pytest`): 19 tests pass without Ollama or internet
access. They cover text cleaning, chunking, page numbers, bibliography
exclusion, retrieval, both refusal checks, prompt construction and the Ollama
request payload.

**Recorded live run** with `qwen2.5:0.5b`, `--top-k 2` and `--threshold 0.08`:
the screenshots in [`docs/screenshots/`](docs/screenshots/) show the assistant
answering the seven example questions with page citations and refusing the
off-topic one.

## How it works

1. `pypdf` extracts the text page by page, and the text is cleaned.
2. Everything from the bibliography onwards is excluded, so citation titles do
   not dominate retrieval.
3. Each page is split into 260-word chunks with a 60-word overlap. Every chunk
   keeps its page number.
4. `scikit-learn` builds a TF-IDF matrix (unigrams and bigrams, English stop
   words removed) over the chunks, held in memory.
5. A question is converted to a TF-IDF vector and compared with every chunk by
   cosine similarity; the top 5 are retrieved.
6. Two checks decide whether to refuse before any model is called (below).
7. Otherwise, only the retrieved chunks are sent to Ollama's local
   `/api/generate` endpoint, under a paper-only system prompt, at temperature
   0.1, without streaming and with a 90-second timeout.
8. The CLI prints the answer followed by the source pages.

## Grounding checks

| Stage | Check | Effect |
| --- | --- | --- |
| Retrieval | No retrieved chunk reaches a cosine similarity of 0.08. | Refuse; the model is not called. |
| Retrieval | Fewer than 2 of the question's meaningful terms (stop words removed, simple plurals folded; 1 if the question has only one), or fewer than 25% of them, appear in the chunks above the threshold. | Refuse; the model is not called. |
| Generation | The system prompt restricts the model to the retrieved context and gives it a fixed refusal sentence. | Depends on the model following the prompt. |

A refusal always uses the same sentence: "I could not find enough information
in the provided paper to answer that question." The two retrieval checks are
deterministic. The generation check is not, because it relies on the model.

## Design choices

- **TF-IDF rather than neural embeddings.** For one paper, TF-IDF needs no extra
  model, and its scores are easy to inspect and explain.
- **In-memory storage rather than a database.** The text volume is small.
- **Page-level chunks.** No chunk crosses a page boundary, so every source
  citation is a single page.
- **Ollama.** Runs the model locally behind a simple HTTP API. The default
  model is `llama3`; `qwen2.5:0.5b` (about 0.5 billion parameters) is the
  lightweight option for limited hardware.

[`TECHNICAL_RATIONALE.md`](TECHNICAL_RATIONALE.md) answers the design
questions in more detail.

## Setup

Requirements: Python 3.10 or newer and [Ollama](https://ollama.com).

```bash
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
ollama pull llama3               # or: ollama pull qwen2.5:0.5b
```

The paper PDF is included in the repository root. The default `--pdf` path is
relative, so run the commands below from the repository root.

## Usage

```bash
python main.py
python main.py --model qwen2.5:0.5b
python main.py --pdf "path/to/paper.pdf" --top-k 3 --show-context
```

| Argument | Default | Purpose |
| --- | --- | --- |
| `--pdf` | the included paper | Path to the paper PDF. |
| `--model` | `llama3` | Ollama model name. |
| `--top-k` | `5` | Number of chunks retrieved per question. |
| `--threshold` | `0.08` | Minimum retrieval score before generation. |
| `--show-context` | off | Print the retrieved snippets. |
| `--ollama-url` | `http://localhost:11434` | Local Ollama server URL. |

Interactive commands: `help`, `examples`, `sources` (pages and scores for the
last question), `settings`, `prompt`, `stack`, `why`, `architecture`, `files`,
`clear`, `exit`.

## Tests

```bash
python smoke_test.py
python -m pytest
```

Neither needs Ollama. Last verified on Python 3.12 with pypdf 6.21.1,
scikit-learn 1.9.1, requests 2.34.2 and pytest 9.1.1.

## Files

| File | Purpose |
| --- | --- |
| `main.py` | Interactive CLI: commands, retrieval, refusal, generation and output. |
| `config.py` | Defaults, example questions and the paper-only system prompt. |
| `paper_loader.py` | PDF validation, page-by-page extraction, cleaning, bibliography exclusion and chunking. |
| `retriever.py` | TF-IDF retrieval, both refusal checks and context formatting. |
| `ollama_client.py` | Non-streaming requests to the local Ollama API, with clear errors. |
| `smoke_test.py` | End-to-end check of everything except generation. |
| `tests/test_core.py` | Pytest suite for the core behaviour. |
| `docs/screenshots/` | Recorded live run. |

## Limitations

- TF-IDF matches words, not meaning, so a question phrased differently from the
  paper can be refused even when the paper answers it.
- Citations are page-level, not sentence-level.
- Retrieval controls what the model sees, not what it writes. In the recorded
  run, the 0.5-billion-parameter model's "supporting evidence" for the tools and
  APIs question joins two unrelated sentences into one quotation
  ([screenshot 7](docs/screenshots/07-tools-and-apis.png)). Answers still need
  checking against the cited pages.
- The checks target unsupported questions, not adversarial ones. They have not
  been tested against prompt injection in questions or documents.
- Scanned PDFs would need OCR first.

## Paper

Brohi, S., Mastoi, Q.-u.-a., Jhanjhi, N.Z. and Pillai, T.R. (2025). A Research
Landscape of Agentic AI and Large Language Models: Applications, Challenges and
Future Directions. *Algorithms*, 18(8), 499.
https://doi.org/10.3390/a18080499

The PDF in this repository is the published article, unmodified apart from its
file name. It is © 2025 the authors and distributed under the
[Creative Commons Attribution 4.0 licence](https://creativecommons.org/licenses/by/4.0/).

## Licence

The code is released under the MIT licence; see [`LICENSE`](LICENSE). The MIT
licence does not cover the paper PDF, which keeps its CC BY 4.0 licence.

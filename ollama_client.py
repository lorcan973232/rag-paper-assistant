"""Small client for the local Ollama HTTP API."""

from __future__ import annotations

from dataclasses import dataclass

import requests

from config import DEFAULT_OLLAMA_BASE_URL, DEFAULT_OLLAMA_TIMEOUT_SECONDS, PAPER_ONLY_SYSTEM_PROMPT


class OllamaError(RuntimeError):
    """Base error for local Ollama failures."""


class OllamaConnectionError(OllamaError):
    """Raised when the local Ollama server cannot be reached."""


class OllamaModelError(OllamaError):
    """Raised when the requested local model is missing or fails."""


@dataclass(frozen=True)
class OllamaClient:
    """Client for Ollama's local `/api/generate` endpoint."""

    model: str
    base_url: str = DEFAULT_OLLAMA_BASE_URL
    timeout_seconds: int = DEFAULT_OLLAMA_TIMEOUT_SECONDS

    @property
    def generate_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/api/generate"

    def generate(self, prompt: str, system_prompt: str = PAPER_ONLY_SYSTEM_PROMPT) -> str:
        """Generate one answer from the selected local Ollama model."""

        # The request is non-streamed so the CLI can print one complete answer
        # and then show the source pages directly beneath it.
        payload = {
            "model": self.model,
            "system": system_prompt,
            "prompt": prompt,
            "stream": False,
            "options": {
                # A low temperature keeps the demonstration more stable without
                # changing the paper-only retrieval rule.
                "temperature": 0.1,
                "top_p": 0.9,
            },
        }

        try:
            response = requests.post(self.generate_url, json=payload, timeout=self.timeout_seconds)
        except requests.exceptions.ConnectionError as exc:
            # Setup problems are common during marking, so the CLI reports them
            # as clear Ollama errors rather than raw requests tracebacks.
            raise OllamaConnectionError(
                "Could not connect to Ollama at "
                f"{self.base_url}. Start Ollama with 'ollama serve' or open the Ollama app."
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise OllamaConnectionError(
                f"Ollama did not respond within {self.timeout_seconds} seconds. "
                "Try again or use a smaller/faster local model."
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise OllamaConnectionError(f"Ollama request failed: {exc}") from exc

        if response.status_code == 404:
            raise OllamaModelError(
                f"Ollama model '{self.model}' was not found. Install it with: ollama pull {self.model}"
            )

        if response.status_code >= 400:
            raise OllamaModelError(
                f"Ollama returned HTTP {response.status_code}: {response.text.strip()}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise OllamaModelError("Ollama returned a non-JSON response.") from exc

        answer = str(data.get("response", "")).strip()
        if not answer:
            raise OllamaModelError("Ollama returned an empty answer.")

        return answer


def build_ollama_prompt(question: str, paper_context: str) -> str:
    """Build the prompt using only the retrieved paper context."""

    # The prompt repeats the paper-only rule at generation time. Retrieval has
    # already decided which chunks are allowed into this context.
    return f"""
Use the paper context below to answer the question. The context is the only source you may use.

Paper context:
{paper_context}

Question:
{question.strip()}

Answer format:
Direct answer:
[answer based only on the paper context]

Supporting evidence:
[brief evidence from the provided context]
""".strip()

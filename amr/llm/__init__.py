"""Language model abstraction and OpenAI/Ollama-backed implementations."""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from types import ModuleType
from typing import (
    TYPE_CHECKING,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Sequence,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

OpenAI: ModuleType | None
OpenAIError: type[Exception]
_openai_module: ModuleType | None
_requests_module: ModuleType | None = None


def _load_openai() -> None:
    """Attempt to import the OpenAI SDK lazily."""

    global OpenAI, OpenAIError, _openai_module

    try:  # pragma: no cover - optional dependency
        _openai_module = importlib.import_module("openai")
    except Exception:  # module truly missing
        _openai_module = None  # type: ignore
        OpenAI = None  # type: ignore
        OpenAIError = Exception
        return

    if hasattr(_openai_module, "OpenAI"):
        OpenAI = getattr(_openai_module, "OpenAI")  # type: ignore[attr-defined]
        OpenAIError = getattr(_openai_module, "APIError", Exception)
    else:  # legacy SDK or missing class
        OpenAI = None  # type: ignore
        if hasattr(_openai_module, "error"):
            OpenAIError = getattr(_openai_module.error, "OpenAIError", Exception)
        else:
            OpenAIError = Exception


_load_openai()


class LanguageModel(Protocol):
    """Protocol describing a language model."""

    def complete(self, prompt: str, *, system_prompt: Optional[str] = None, **kwargs: object) -> str:
        """Return a completion string given a prompt."""

    def embed(self, texts: Sequence[str]) -> "torch.Tensor":
        """Return embeddings for a batch of strings."""


@dataclass
class OpenAIConfig:
    """Configuration for :class:`OpenAILanguageModel`."""

    model: str = "gpt-5-nano"
    embedding_model: str = "text-embedding-3-small"
    temperature: float = 0.0
    max_tokens: int = 512
    api_key: Optional[str] = None


@dataclass
class OllamaConfig:
    """Configuration for :class:`OllamaLanguageModel`."""

    model: str = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
    embedding_model: str = os.getenv("OLLAMA_EMBED_MODEL", "embeddinggemma:latest")
    base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    temperature: Optional[float] = 0.0
    num_ctx: Optional[int] = None
    num_predict: Optional[int] = 512
    headers: Optional[Mapping[str, str]] = None
    timeout: float = 300.0
    options: Optional[Mapping[str, object]] = None


class OpenAILanguageModel(LanguageModel):
    """Concrete :class:`LanguageModel` backed by OpenAI's API."""

    def __init__(self, config: OpenAIConfig | None = None) -> None:
        if OpenAI is None or _openai_module is None:
            _load_openai()

        if OpenAI is None and _openai_module is None:
            raise ImportError(
                "The openai package is required for OpenAILanguageModel. Install it with `pip install openai`."
            )

        self.config = config or OpenAIConfig()
        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be supplied via config or environment variable.")

        self._client = None
        self._legacy_module: ModuleType | None = None
        if OpenAI is not None:
            self._client = OpenAI(api_key=api_key)
        else:
            # Legacy SDK path (pre-1.0)
            self._legacy_module = _openai_module
            assert self._legacy_module is not None  # for type checkers
            self._legacy_module.api_key = api_key

    def complete(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: object,
    ) -> str:
        """Generate a completion using OpenAI's chat completion API."""

        chat_model = model or self.config.model
        chat_temperature = temperature if temperature is not None else self.config.temperature
        chat_max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens

        messages = _build_chat_messages(prompt, system_prompt=system_prompt)

        try:
            if OpenAI is not None:
                assert self._client is not None
                payload = dict(
                    model=chat_model,
                    messages=messages,
                    **kwargs,
                )
                temperature_override = chat_temperature
                if temperature_override is not None:
                    if chat_model.startswith("gpt-5") and temperature_override != 1:
                        temperature_override = None
                    elif chat_model.startswith("gpt-5"):
                        temperature_override = None
                if temperature_override is not None:
                    payload["temperature"] = temperature_override
                if chat_model.startswith("gpt-5") and "response_format" not in payload:
                    payload["response_format"] = {"type": "json_object"}
                if chat_max_tokens is not None:
                    payload["max_completion_tokens"] = chat_max_tokens
                response = self._client.chat.completions.create(**payload)
                choice = response.choices[0]
                content = getattr(choice.message, "content", None)
                if content is None:
                    raise RuntimeError("OpenAI completion returned no content.")
                return content

            assert self._legacy_module is not None
            response = self._legacy_module.ChatCompletion.create(
                model=chat_model,
                messages=messages,
                temperature=chat_temperature,
                max_tokens=chat_max_tokens,
                **kwargs,
            )
            return response["choices"][0]["message"]["content"]
        except OpenAIError as exc:  # pragma: no cover - network/API error handling
            raise RuntimeError(f"OpenAI completion failed: {exc}") from exc

    def embed(
        self,
        texts: Sequence[str],
        *,
        model: Optional[str] = None,
    ) -> "torch.Tensor":
        """Generate embeddings for ``texts`` via OpenAI's embedding endpoint."""

        if not texts:
            raise ValueError("At least one text is required to compute embeddings.")

        embedding_model = model or self.config.embedding_model
        torch = _ensure_torch()
        try:
            if OpenAI is not None:
                assert self._client is not None
                response = self._client.embeddings.create(
                    model=embedding_model,
                    input=list(texts),
                )
                embeddings_data = [torch.tensor(item.embedding, dtype=torch.float32) for item in response.data]
            else:
                assert self._legacy_module is not None
                response = self._legacy_module.Embedding.create(
                    model=embedding_model,
                    input=list(texts),
                )
                embeddings_data = [
                    torch.tensor(item["embedding"], dtype=torch.float32)
                    for item in response["data"]
                ]
        except OpenAIError as exc:  # pragma: no cover - network/API error handling
            raise RuntimeError(f"OpenAI embeddings failed: {exc}") from exc

        return torch.stack(embeddings_data)


class OllamaLanguageModel(LanguageModel):
    """Concrete :class:`LanguageModel` backed by an Ollama server."""

    def __init__(self, config: OllamaConfig | None = None) -> None:
        self.config = config or OllamaConfig()
        self._requests = _load_requests()
        self._session = self._requests.Session()

    def complete(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: object,
    ) -> str:
        messages = _build_chat_messages(prompt, system_prompt=system_prompt)
        chat_model = model or self.config.model
        payload: dict[str, object] = {
            "model": chat_model,
            "messages": messages,
            "stream": False,
        }

        options = dict(self.config.options or {})
        temp_override = temperature if temperature is not None else self.config.temperature
        if temp_override is not None:
            options.setdefault("temperature", temp_override)
        if self.config.num_ctx is not None:
            options.setdefault("num_ctx", self.config.num_ctx)
        predict_tokens = max_tokens if max_tokens is not None else self.config.num_predict
        if predict_tokens is not None:
            options["num_predict"] = predict_tokens
        if options:
            payload["options"] = options

        if kwargs:
            payload.update(kwargs)

        response = self._post("/api/chat", payload)
        message = response.get("message")
        content: Optional[str]
        if isinstance(message, dict):
            content_obj = message.get("content")
            if isinstance(content_obj, list):
                content = "".join(part.get("text", "") for part in content_obj if isinstance(part, dict))
            else:
                content = content_obj  # type: ignore[assignment]
        else:
            content = response.get("response")

        if not content:
            raise RuntimeError("Ollama completion returned no content.")
        return content

    def embed(
        self,
        texts: Sequence[str],
        *,
        model: Optional[str] = None,
    ) -> "torch.Tensor":
        if not texts:
            raise ValueError("At least one text is required to compute embeddings.")

        torch = _ensure_torch()
        embedding_model = model or self.config.embedding_model
        vectors = []
        for text in texts:
            payload = {
                "model": embedding_model,
                "prompt": text,
            }
            response = self._post("/api/embeddings", payload)
            embedding = response.get("embedding")
            if embedding is None:
                raise RuntimeError("Ollama embeddings response missing 'embedding' field.")
            vectors.append(torch.tensor(embedding, dtype=torch.float32))

        return torch.stack(vectors)

    def _post(self, path: str, payload: Mapping[str, object]) -> dict:
        url = _join_url(self.config.base_url, path)
        try:
            response = self._session.post(
                url,
                json=payload,
                headers=self.config.headers,
                timeout=self.config.timeout,
            )
            response.raise_for_status()
        except self._requests.exceptions.RequestException as exc:  # type: ignore[attr-defined]
            raise RuntimeError(f"Ollama request failed: {exc}") from exc
        data = response.json()
        if not isinstance(data, dict):
            raise RuntimeError("Ollama response payload is malformed.")
        return data


ProviderLiteral = Literal["openai", "ollama"]


def create_language_model(
    provider: ProviderLiteral = "openai",
    *,
    config: OpenAIConfig | OllamaConfig | None = None,
    **kwargs: object,
) -> LanguageModel:
    """Factory returning a language model for ``provider``. Keyword args seed a config."""

    if provider == "openai":
        cfg = config if isinstance(config, OpenAIConfig) else OpenAIConfig(**kwargs)
        return OpenAILanguageModel(cfg)
    if provider == "ollama":
        cfg = config if isinstance(config, OllamaConfig) else OllamaConfig(**kwargs)
        return OllamaLanguageModel(cfg)
    raise ValueError(f"Unknown provider '{provider}'.")


def _ensure_torch():
    """Import and return the torch module, raising a helpful error if unavailable."""

    try:  # pragma: no cover - optional dependency
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "The `torch` package is required for embedding support. Install it with `pip install torch`."
        ) from exc

    return torch


def _build_chat_messages(
    prompt: str,
    *,
    system_prompt: Optional[str] = None,
    extra_messages: Optional[Sequence[Mapping[str, object]]] = None,
) -> list[dict[str, object]]:
    messages: list[dict[str, object]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if extra_messages:
        messages.extend([dict(msg) for msg in extra_messages])
    messages.append({"role": "user", "content": prompt})
    return messages


def _load_requests() -> ModuleType:
    global _requests_module
    if _requests_module is not None:
        return _requests_module
    try:  # pragma: no cover - optional dependency
        _requests_module = importlib.import_module("requests")
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "The `requests` package is required for Ollama support. Install it with `pip install requests`."
        ) from exc
    return _requests_module


def _join_url(base: str, path: str) -> str:
    base = base.rstrip("/")
    path = path.lstrip("/")
    return f"{base}/{path}"


__all__ = [
    "LanguageModel",
    "OpenAILanguageModel",
    "OpenAIConfig",
    "OllamaLanguageModel",
    "OllamaConfig",
    "ProviderLiteral",
    "create_language_model",
    "create_llm_client",
]


def create_llm_client(provider: ProviderLiteral = "openai", **config_overrides: object) -> LanguageModel:
    """Return a ready-to-use :class:`LanguageModel` with sensible defaults.

    ``config_overrides`` are forwarded to the provider-specific config constructor,
    letting callers tweak parameters without importing the config classes.
    """

    if provider == "openai":
        config = OpenAIConfig(**config_overrides)
    elif provider == "ollama":
        config = OllamaConfig(**config_overrides)
    else:
        raise ValueError(f"Unknown provider '{provider}'.")

    return create_language_model(provider, config=config)

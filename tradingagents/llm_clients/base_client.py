import hashlib
import logging
import warnings
from abc import ABC, abstractmethod
from typing import Any

from langchain_core.messages import AIMessage

logger = logging.getLogger(__name__)


def _prompt_signature(input_: Any) -> str:
    """A stable digest of an LLM input, for the response-cache key.

    Accepts a bare string, a list of message dicts, a list of ``(role, content)``
    tuples/lists, a list of message objects, or a ``ChatPromptValue``
    (``.to_messages()``). Each shape is read by its own accessor:

    - A bare string is hashed directly (it has no ``type``/``content``).
    - A message **dict** (``{"role": ..., "content": ...}``) is read with
      ``.get()`` — a dict has no ``.type``/``.content`` attribute, so the
      ``getattr`` path would collapse every distinct dict prompt into one key
      (the Trader is the one agent that passes ``[{"role": ..., "content": ...}]``;
      two tickers then shared a cache key and the second run served the first
      ticker's cached transaction plan).
    - A ``(role, content)`` **tuple or 2-list** (the shape the reflector's
      ``reflect_on_final_decision`` sends) is read positionally. Without this
      branch a tuple falls through to ``getattr(m, "type")`` → ``"tuple"`` and
      ``getattr(m, "content")`` → ``""``, collapsing *every* reflection call
      into one cache key, so all settled decisions replay the first one's
      reflection regardless of their actual outcome (a loss was reflected as a
      "rapid appreciation" win).
    - A message object is read via ``getattr``.
    """
    if isinstance(input_, str):
        return hashlib.sha256(input_.encode("utf-8")).hexdigest()
    if isinstance(input_, list):
        messages = input_
    elif hasattr(input_, "to_messages"):
        messages = input_.to_messages()
    else:
        messages = [input_]
    parts = []
    for m in messages:
        if isinstance(m, dict):
            role = m.get("role") or m.get("type") or "dict"
            content = m.get("content", "")
        elif (
            isinstance(m, (tuple, list))
            and len(m) == 2
            and isinstance(m[0], str)
        ):
            # (role, content) pair, e.g. ("system", "...") / ("human", "...").
            role, content = m[0], m[1]
        else:
            role = getattr(m, "type", None) or type(m).__name__
            content = getattr(m, "content", "")
        if isinstance(content, list):
            content = "".join(
                c.get("text", "") if isinstance(c, dict) else str(c) for c in content
            )
        parts.append(f"{role}:{content}")
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def _serialize_llm_response(response: Any) -> dict:
    """The parts of an LLM response we cache and can faithfully rebuild.

    ``content`` (normalized to string), ``tool_calls`` and
    ``response_metadata`` are kept; reasoning blocks and provider-specific
    fields are dropped. ``tool_calls`` must round-trip: structured output
    (``function_calling``) carries the parsed result in ``tool_calls`` while
    ``content`` is empty or just a thinking fragment, so dropping the tool call
    freezes a valid decision into a "no decision" on every later cache hit.
    """
    content = getattr(response, "content", "")
    if isinstance(content, list):
        content = "\n".join(
            item.get("text", "") if isinstance(item, dict) else str(item) for item in content
        )
    tool_calls = getattr(response, "tool_calls", None)
    if tool_calls:
        # langchain-core stores tool_calls as a list of dicts; keep them as-is
        # so AIMessage reconstructs them without a ToolCall re-parse.
        tool_calls = list(tool_calls)
    else:
        tool_calls = []
    return {
        "content": content,
        "tool_calls": tool_calls,
        "response_metadata": dict(getattr(response, "response_metadata", {}) or {}),
    }


def _rebuild_llm_response(payload: dict) -> AIMessage:
    """Reconstruct an ``AIMessage`` from a cached payload."""
    return AIMessage(
        content=payload.get("content", ""),
        tool_calls=payload.get("tool_calls") or [],
        response_metadata=payload.get("response_metadata", {}),
    )


def normalize_content(response):
    """Normalize LLM response content to a plain string.

    Multiple providers (OpenAI Responses API, Google Gemini 3) return content
    as a list of typed blocks, e.g. [{'type': 'reasoning', ...}, {'type': 'text', 'text': '...'}].
    Downstream agents expect response.content to be a string. This extracts
    and joins the text blocks, discarding reasoning/metadata blocks.
    """
    content = response.content
    if isinstance(content, list):
        texts = [
            item.get("text", "") if isinstance(item, dict) and item.get("type") == "text"
            else item if isinstance(item, str) else ""
            for item in content
        ]
        response.content = "\n".join(t for t in texts if t)
    return response


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    def __init__(self, model: str, base_url: str | None = None, **kwargs):
        self.model = model
        self.base_url = base_url
        self.kwargs = kwargs

    def get_provider_name(self) -> str:
        """Return the provider name used in warning messages."""
        provider = getattr(self, "provider", None)
        if provider:
            return str(provider)
        return self.__class__.__name__.removesuffix("Client").lower()

    def warn_if_unknown_model(self) -> None:
        """Warn when the model is outside the known list for the provider."""
        if self.validate_model():
            return

        warnings.warn(
            (
                f"Model '{self.model}' is not in the known model list for "
                f"provider '{self.get_provider_name()}'. Continuing anyway."
            ),
            RuntimeWarning,
            stacklevel=2,
        )

    @abstractmethod
    def get_llm(self) -> Any:
        """Return the configured LLM instance."""
        pass

    @abstractmethod
    def validate_model(self) -> bool:
        """Validate that the model is supported by this client."""
        pass

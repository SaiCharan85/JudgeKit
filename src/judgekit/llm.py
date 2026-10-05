"""How JudgeKit talks to an LLM, without any provider code.

Consumers plug in at one of two levels:

- `StructuredLLM`: their client already returns validated Pydantic objects (and does its own
  caching, budgets, fallback and validation retries). The adapter is a thin pass-through.
- `TextLLM`: their client only returns text. Wrap it in `TextStructuredLLM`, which adds the JSON
  schema, extracts and validates the JSON, and retries once with the validation error.

Failures a judge should survive are `LLMCallError`s (adapters map their own errors to it); any
other exception is a bug and propagates.
"""

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Generic, Literal, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

T = TypeVar("T", bound=BaseModel)

ERROR_FEEDBACK_MAX = 800  # characters of validation error fed back to the model on retry


class CallInfo(BaseModel):
    """What one LLM call cost, for audit logs and budgets."""

    model_config = ConfigDict(frozen=True)

    model: str = ""
    family: str = ""  # model family, used to keep a judge away from the model it grades
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    latency_s: float = Field(default=0.0, ge=0)
    attempts: int = Field(default=1, ge=1)
    cached: bool = False


@dataclass(frozen=True)
class StructuredResult(Generic[T]):
    value: T
    calls: tuple[CallInfo, ...] = ()


class LLMCallError(Exception):
    """The model could not produce an answer (unavailable, over budget, rate-limited, ...)."""


class StructuredOutputError(LLMCallError):
    """The model answered, but not with valid structured output, even after retrying."""

    def __init__(self, message: str, raw: str = "", calls: tuple[CallInfo, ...] = ()) -> None:
        super().__init__(message)
        self.raw = raw
        self.calls = calls


class StructuredLLM(Protocol):
    def complete(
        self,
        system: str,
        user: str,
        schema: type[T],
        *,
        avoid_families: frozenset[str] = frozenset(),
    ) -> StructuredResult[T]:
        """Return a validated `schema` instance or raise `LLMCallError`.

        `avoid_families`: model families that must not answer (e.g. the family that wrote the
        text being judged, to avoid self-preference).
        """
        ...


# ---------------------------------------------------------------- text-only clients


class ChatMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class TextReply:
    text: str
    info: CallInfo = field(default_factory=CallInfo)


class TextLLM(Protocol):
    def chat(
        self, messages: Sequence[ChatMessage], *, avoid_families: frozenset[str] = frozenset()
    ) -> TextReply:
        """Return the model's text reply or raise `LLMCallError`."""
        ...


def schema_instructions(schema: type[BaseModel]) -> str:
    """Compact JSON-schema instruction appended to the system prompt (static: cache-friendly)."""
    compact = json.dumps(schema.model_json_schema(), separators=(",", ":"))
    return f"Reply with one JSON object only, no prose, matching this JSON schema:\n{compact}"


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> str:
    """The JSON object in a reply: inside a ``` fence if there is one, else first { to last }."""
    m = _FENCE.search(text)
    body = m.group(1) if m else text
    start, end = body.find("{"), body.rfind("}")
    return body[start : end + 1] if 0 <= start < end else body.strip()


def _feedback(exc: ValidationError) -> str:
    errors = "; ".join(
        f"{'.'.join(map(str, e['loc'])) or '<root>'}: {e['msg']}" for e in exc.errors()
    )
    return (
        f"Your previous reply was not valid: {errors}"[:ERROR_FEEDBACK_MAX]
        + "\nReply again with the corrected JSON object only."
    )


class TextStructuredLLM:
    """Make a text-only client a `StructuredLLM`: schema in the prompt, retry with the error."""

    def __init__(self, llm: TextLLM, max_retries: int = 1) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self.llm = llm
        self.max_retries = max_retries

    def complete(
        self,
        system: str,
        user: str,
        schema: type[T],
        *,
        avoid_families: frozenset[str] = frozenset(),
    ) -> StructuredResult[T]:
        messages = [
            ChatMessage(role="system", content=f"{system}\n\n{schema_instructions(schema)}"),
            ChatMessage(role="user", content=user),
        ]
        calls: list[CallInfo] = []
        reply = TextReply("")
        for _ in range(self.max_retries + 1):
            reply = self.llm.chat(messages, avoid_families=avoid_families)
            calls.append(reply.info)
            try:
                value = schema.model_validate_json(extract_json(reply.text))
            except ValidationError as exc:
                messages += [
                    ChatMessage(role="assistant", content=reply.text),
                    ChatMessage(role="user", content=_feedback(exc)),
                ]
                continue
            return StructuredResult(value, tuple(calls))
        raise StructuredOutputError(
            f"no valid {schema.__name__} after {len(calls)} attempt(s)", reply.text, tuple(calls)
        )

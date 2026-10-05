"""Scripted fake LLMs for tests (no network, no keys). Usable by consumer test suites too."""

from collections.abc import Callable, Sequence
from typing import TypeVar

from pydantic import BaseModel

from judgekit.llm import CallInfo, ChatMessage, LLMCallError, StructuredResult, TextReply

T = TypeVar("T", bound=BaseModel)

Script = BaseModel | Exception | Callable[[str, str], BaseModel]


class ScriptedLLM:
    """A `StructuredLLM` that replays scripted outputs in order.

    Each entry is a model instance to return, an exception to raise, or a callable
    `(system, user) -> model` for case-dependent answers. Every call is recorded.
    """

    def __init__(self, script: Sequence[Script], info: CallInfo | None = None) -> None:
        self.script = list(script)
        self.info = info or CallInfo(model="scripted", family="scripted")
        self.calls: list[tuple[str, str, frozenset[str]]] = []

    def complete(
        self,
        system: str,
        user: str,
        schema: type[T],
        *,
        avoid_families: frozenset[str] = frozenset(),
    ) -> StructuredResult[T]:
        self.calls.append((system, user, avoid_families))
        if not self.script:
            raise LLMCallError("script exhausted")
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        value = step(system, user) if callable(step) and not isinstance(step, BaseModel) else step
        return StructuredResult(schema.model_validate(value.model_dump()), (self.info,))


class ScriptedTextLLM:
    """A `TextLLM` that replays scripted text replies (or exceptions) and records every chat."""

    def __init__(self, replies: Sequence[str | Exception], info: CallInfo | None = None) -> None:
        self.replies = list(replies)
        self.info = info or CallInfo(model="scripted-text", family="scripted")
        self.chats: list[list[ChatMessage]] = []

    def chat(
        self, messages: Sequence[ChatMessage], *, avoid_families: frozenset[str] = frozenset()
    ) -> TextReply:
        self.chats.append(list(messages))
        if not self.replies:
            raise LLMCallError("script exhausted")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return TextReply(reply, self.info)

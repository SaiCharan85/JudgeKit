import json

import pytest
from pydantic import BaseModel, ConfigDict

from judgekit import CallInfo, ChatMessage, LLMCallError, StructuredOutputError, TextStructuredLLM
from judgekit.llm import ERROR_FEEDBACK_MAX, TextReply, extract_json, schema_instructions
from judgekit.testing import ScriptedTextLLM


class Point(BaseModel):
    x: int
    y: int


# ---------------------------------------------------------------- JSON extraction


@pytest.mark.parametrize(
    "text",
    [
        '{"x": 1, "y": 2}',
        'Sure! Here it is: {"x": 1, "y": 2} Hope that helps.',
        '```json\n{"x": 1, "y": 2}\n```',
        '```\n{"x": 1, "y": 2}\n```',
        '```JSON\n{"x": 1, "y": 2}\n``` trailing',
    ],
)
def test_extract_json_variants(text: str) -> None:
    assert json.loads(extract_json(text)) == {"x": 1, "y": 2}


def test_extract_json_keeps_nested_objects() -> None:
    assert json.loads(extract_json('note {"a": {"b": 1}} end')) == {"a": {"b": 1}}


def test_extract_json_without_braces_returns_text() -> None:
    assert extract_json("  no json here  ") == "no json here"


def test_schema_instructions_are_compact_and_name_fields() -> None:
    schema_line = schema_instructions(Point).split("\n", 1)[1]
    assert '"x"' in schema_line and '"y"' in schema_line
    assert ": " not in schema_line  # compact separators keep prompts short


# ---------------------------------------------------------------- TextStructuredLLM


def test_valid_first_reply() -> None:
    text = ScriptedTextLLM(['{"x": 1, "y": 2}'], CallInfo(model="m", input_tokens=10))
    res = TextStructuredLLM(text).complete("sys", "user", Point)
    assert res.value == Point(x=1, y=2)
    assert [c.model for c in res.calls] == ["m"]
    system, user = text.chats[0]
    assert system.role == "system" and "JSON schema" in system.content
    assert system.content.startswith("sys")  # static instructions first: cache-friendly
    assert (user.role, user.content) == ("user", "user")


def test_retries_once_with_the_validation_error() -> None:
    text = ScriptedTextLLM(['{"x": "one", "y": 2}', '{"x": 1, "y": 2}'])
    res = TextStructuredLLM(text).complete("sys", "user", Point)
    assert res.value.x == 1
    assert len(res.calls) == 2
    retry = text.chats[1]
    assert [m.role for m in retry] == ["system", "user", "assistant", "user"]
    assert retry[2].content == '{"x": "one", "y": 2}'
    assert "x:" in retry[3].content and "not valid" in retry[3].content


def test_gives_up_after_retries_with_raw_and_calls() -> None:
    text = ScriptedTextLLM(["nope", "still nope"])
    with pytest.raises(StructuredOutputError) as e:
        TextStructuredLLM(text).complete("sys", "user", Point)
    assert e.value.raw == "still nope"
    assert len(e.value.calls) == 2
    assert isinstance(e.value, LLMCallError)


def test_zero_retries_means_one_attempt() -> None:
    text = ScriptedTextLLM(["bad", '{"x": 1, "y": 2}'])
    with pytest.raises(StructuredOutputError):
        TextStructuredLLM(text, max_retries=0).complete("sys", "user", Point)
    assert len(text.chats) == 1


def test_negative_retries_rejected() -> None:
    with pytest.raises(ValueError):
        TextStructuredLLM(ScriptedTextLLM([]), max_retries=-1)


def test_call_errors_propagate_without_retry() -> None:
    text = ScriptedTextLLM([LLMCallError("rate limited"), '{"x": 1, "y": 2}'])
    with pytest.raises(LLMCallError, match="rate limited"):
        TextStructuredLLM(text).complete("sys", "user", Point)
    assert len(text.chats) == 1  # a failed call is not a validation retry


def test_avoid_families_passed_through() -> None:
    seen: list[frozenset[str]] = []

    class Recorder(ScriptedTextLLM):
        def chat(
            self, messages: list[ChatMessage], *, avoid_families: frozenset[str] = frozenset()
        ) -> TextReply:
            seen.append(avoid_families)
            return super().chat(messages, avoid_families=avoid_families)

    TextStructuredLLM(Recorder(['{"x": 1, "y": 2}'])).complete(
        "s", "u", Point, avoid_families=frozenset({"fam"})
    )
    assert seen == [frozenset({"fam"})]


def test_error_feedback_is_capped() -> None:
    class Strict(BaseModel):
        model_config = ConfigDict(extra="forbid")
        a: int

    huge = json.dumps({"a": 1, **{f"k{i}": i for i in range(300)}})
    text = ScriptedTextLLM([huge, '{"a": 1}'])
    TextStructuredLLM(text).complete("s", "u", Strict)
    feedback = text.chats[1][3].content
    assert len(feedback) <= ERROR_FEEDBACK_MAX + 60


def test_call_info_rejects_negative_counts() -> None:
    with pytest.raises(ValueError):
        CallInfo(input_tokens=-1)

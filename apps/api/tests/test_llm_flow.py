"""The LLM writer's control flow, without a network call.

The provider is never contacted here: the SDK client is replaced with a stub. That
keeps these tests runnable with no key, and keeps them fast and deterministic.
What is being tested is the part that matters - the retry, the tool-call parsing,
and the fallback when the model misbehaves.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.llm import writer as writer_module
from app.llm.writer import InvalidNarrative, LlmWriter, TemplateWriter, get_writer, reset_writer

FACTS = {
    "diagnosis": {"crop": "ধান", "disease": "ব্লাস্ট রোগ", "confidence": "মাঝারি"},
    "symptoms_observed": ["পাতায় ধূসর-সবুজ দিম্বাকৃতি দাগ"],
    "treatment": {
        "organic": ["আক্রান্ত পাতা সরান"],
        "chemical": [
            {
                "product": "Tricyclazole",
                "bnName": "ট্রাইসাইক্লাজল",
                "dose": "প্রতি লিটারে ০.৬ গ্রাম",
                "note": "৭ দিন পরপর",
            }
        ],
    },
    "prevention": ["অতিরিক্ত ইউরিয়া দেবেন না"],
    "weather": None,
    "uncertain": False,
}

GOOD_SPOKEN = (
    "আপনার ধান গাছে সম্ভবত ব্লাস্ট রোগ হয়েছে। "
    "পাতায় ধূসর-সবুজ দাগ দেখা যাচ্ছে। "
    "ট্রাইসাইক্লাজল প্রতি লিটারে ০.৬ গ্রাম হিসেবে ব্যবহার করতে পারেন। "
    "মনে রাখবেন, এটি একটি আনুমানিক পরামর্শ।"
)


def _response(spoken: str | None, *, finish_reason: str = "tool_calls") -> SimpleNamespace:
    if spoken is None:
        message = SimpleNamespace(tool_calls=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(finish_reason=finish_reason, message=message)]
        )
    call = SimpleNamespace(function=SimpleNamespace(arguments=json.dumps({"spoken": spoken})))
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(tool_calls=[call]),
            )
        ]
    )


class StubClient:
    """Records calls and replays queued responses."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        result = self._responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def _writer_with(responses) -> tuple[LlmWriter, StubClient]:
    writer = LlmWriter.__new__(LlmWriter)  # skip the real SDK client
    import asyncio

    writer._semaphore = asyncio.Semaphore(3)
    client = StubClient(responses)
    writer._client = client
    return writer, client


async def test_good_output_is_accepted() -> None:
    writer, client = _writer_with([_response(GOOD_SPOKEN)])
    text, generated_by = await writer.write(FACTS)

    assert generated_by == "llm"
    assert "ট্রাইসাইক্লাজল" in text
    # A forced tool call, not response_format: the gateway drops the latter.
    forced = {"type": "function", "function": {"name": "speak_advice"}}
    assert client.calls[0]["tool_choice"] == forced
    assert client.calls[0]["max_tokens"] == 600
    assert client.calls[0]["temperature"] == 0.4


async def test_bad_then_good_retries_once() -> None:
    """A rejected answer gets exactly one more attempt, with a correction."""
    writer, client = _writer_with([_response("ধান"), _response(GOOD_SPOKEN)])
    text, generated_by = await writer.write(FACTS)

    assert generated_by == "llm"
    assert "ট্রাইসাইক্লাজল" in text
    assert len(client.calls) == 2

    retry_messages = client.calls[1]["messages"]
    assert "আগের উত্তরটি নিয়ম মানেনি" in retry_messages[-1]["content"]


async def test_two_bad_answers_raise_for_fallback() -> None:
    writer, _ = _writer_with([_response("ধান"), _response("ধান")])
    with pytest.raises(InvalidNarrative, match="unusable_after_retry"):
        await writer.write(FACTS)


async def test_missing_tool_call_raises() -> None:
    writer, _ = _writer_with([_response(None), _response(None)])
    with pytest.raises(InvalidNarrative, match="unusable_after_retry"):
        await writer.write(FACTS)


async def test_truncated_output_is_rejected() -> None:
    """Truncation is an invalid answer, so it gets the single retry like any other."""
    writer, client = _writer_with(
        [_response(GOOD_SPOKEN, finish_reason="length"), _response(GOOD_SPOKEN)]
    )
    text, generated_by = await writer.write(FACTS)

    assert generated_by == "llm"
    assert len(client.calls) == 2
    assert "ট্রাইসাইক্লাজল" in text


async def test_truncation_twice_raises_for_fallback() -> None:
    writer, _ = _writer_with(
        [
            _response(GOOD_SPOKEN, finish_reason="length"),
            _response(GOOD_SPOKEN, finish_reason="length"),
        ]
    )
    with pytest.raises(InvalidNarrative, match="unusable_after_retry"):
        await writer.write(FACTS)


async def test_network_error_propagates_for_the_caller_to_handle() -> None:
    """The pipeline catches this and falls back; the writer does not retry it.

    Retrying a 429 `window_exhausted` is pointless - it resets in hours.
    """
    writer, client = _writer_with([RuntimeError("429 window_exhausted")])
    with pytest.raises(RuntimeError):
        await writer.write(FACTS)
    assert len(client.calls) == 1


async def test_template_writer_needs_no_client() -> None:
    text, generated_by = await TemplateWriter().write(FACTS)
    assert generated_by == "template"
    assert "ট্রাইসাইক্লাজল" in text
    assert "আনুমানিক" in text


async def test_template_writer_mentions_weather_only_when_present() -> None:
    facts = dict(FACTS)
    facts["weather"] = {
        "district": "ঢাকা",
        "rain_probability_next_12h": 97,
        "wind_speed_max_next_12h": 9.6,
        "humidity": 95,
        "spray_window_ok": False,
        "reason": "আগামী ১২ ঘণ্টায় বৃষ্টির সম্ভাবনা ৯৭ শতাংশ",
    }
    text, _ = await TemplateWriter().write(facts)
    assert "স্প্রে করবেন না" in text
    assert "৯৭ শতাংশ" in text

    text_without, _ = await TemplateWriter().write(FACTS)
    assert "স্প্রে" not in text_without
    assert "আবহাওয়া" not in text_without


def test_writer_selection_falls_back_without_a_key(monkeypatch) -> None:
    monkeypatch.setattr(writer_module.settings, "llm_api_key", "")
    reset_writer()
    try:
        assert isinstance(get_writer(), TemplateWriter)
    finally:
        reset_writer()


def test_writer_selection_uses_llm_when_configured(monkeypatch) -> None:
    monkeypatch.setattr(writer_module.settings, "llm_api_key", "test-key-not-real")
    reset_writer()
    try:
        assert isinstance(get_writer(), LlmWriter)
    finally:
        writer_module.settings.llm_api_key = ""
        reset_writer()

"""Bangla voice-question transcription powered by Gemini Flash audio understanding.

Failure here never blocks leaf disease analysis: an unconfigured key, timeout,
or inaudible clip logs a warning and degrades cleanly to `None`.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

import httpx

from ..config import Settings
from .writer import (
    _RETRYABLE_GEMINI_CODES,
    DEFAULT_GEMINI_MODEL,
    FALLBACK_GEMINI_MODELS,
    normalise,
)

log = logging.getLogger(__name__)

_MARKDOWN_NOISE = re.compile(r"[*_`#>\[\]{}|]")

_TRANSCRIPTION_PROMPT = (
    "এই অডিও ক্লিপে বাংলাদেশের একজন কৃষকের ফসল বা রোগ সংক্রান্ত বাংলা প্রশ্ন রয়েছে। "
    "কৃষক যা বলেছেন, কেবল সেই প্রশ্নটি সঠিক ও সংক্ষিপ্ত বাংলা অক্ষরে লিখুন। "
    "কোনো উত্তর দেবেন না, কোনো ভূমিকা, মার্কডাউন বা অতিরিক্ত মন্তব্য লিখবেন না। "
    "যদি কথা অস্পষ্ট বা নীরব হয়, তবে সম্পূর্ণ খালি উত্তর দিন।"
)


def _candidate_models(settings: Settings) -> list[str]:
    primary = settings.gemini_model.strip() or DEFAULT_GEMINI_MODEL
    candidates = [primary]
    for model in FALLBACK_GEMINI_MODELS:
        if model not in candidates:
            candidates.append(model)
    return candidates


def _build_client(api_key: str, timeout_ms: int) -> Any:
    from google import genai
    from google.genai import types

    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=timeout_ms),
    )


def _clean_transcript(raw_text: str, max_chars: int) -> str | None:
    cleaned = _MARKDOWN_NOISE.sub("", raw_text)
    cleaned = normalise(cleaned).strip("\"'“”‘’ ")
    if not cleaned:
        return None
    return cleaned[:max_chars].strip() or None


def _transcribe_blocking(
    client: Any,
    audio_bytes: bytes,
    mime_type: str,
    settings: Settings,
) -> str | None:
    from google.genai import errors as genai_errors
    from google.genai import types

    audio_part = types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)
    config = types.GenerateContentConfig(
        temperature=0.0,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    last_exc: Exception | None = None
    for model_name in _candidate_models(settings):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[_TRANSCRIPTION_PROMPT, audio_part],
                config=config,
            )
            raw_text = (getattr(response, "text", None) or "").strip()
            return _clean_transcript(raw_text, settings.max_question_chars)
        except genai_errors.APIError as exc:
            last_exc = exc
            if exc.code in _RETRYABLE_GEMINI_CODES:
                log.warning(
                    "Audio transcription model %s returned HTTP %s; trying fallback.",
                    model_name,
                    exc.code,
                )
                continue
            raise
        except (TimeoutError, httpx.TimeoutException) as exc:
            last_exc = exc
            log.warning(
                "Audio transcription model %s timed out (%s); trying fallback.",
                model_name,
                exc.__class__.__name__,
            )
            continue

    raise RuntimeError(f"All Gemini Flash models failed for audio: {last_exc}") from last_exc


async def transcribe_bangla_question(
    audio_bytes: bytes,
    mime_type: str,
    settings: Settings,
) -> str | None:
    """Transcribe a farmer's spoken Bangla question via Gemini Flash, or return None."""
    if not audio_bytes or not mime_type:
        return None

    api_key = settings.effective_gemini_api_key
    if not api_key:
        log.warning("Skipping voice-question transcription: Gemini API key is not configured.")
        return None

    try:
        client = _build_client(api_key, settings.audio_transcription_timeout_ms)
        return await asyncio.to_thread(
            _transcribe_blocking,
            client,
            audio_bytes,
            mime_type,
            settings,
        )
    except Exception as exc:  # noqa: BLE001 - voice input must never fail leaf analysis
        log.warning(
            "Voice-question transcription failed (%s); continuing without voice question.",
            exc.__class__.__name__,
        )
        return None

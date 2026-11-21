"""LLM module using Google Gemini Flash for farmer-friendly Bangla crop advisories."""

import asyncio
import json
import logging
import os
from typing import Optional

from app.schemas import AdvisoryResult, CropType, RagSource, VisionResult, WeatherResult

logger = logging.getLogger(__name__)

DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
FALLBACK_GEMINI_MODELS = (
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
)

SYSTEM_INSTRUCTION = """You are FosholLens, a cautious, empathetic agricultural advisory assistant for smallholder rice and potato farmers in Bangladesh.

Follow these strict safety and formatting rules:
1. Write `bangla_text` in clear, natural Standard Colloquial Bangla (3 to 5 short sentences) that a farmer can easily understand when read aloud.
2. If `vision.status` is "uncertain", clearly explain in Bangla that the leaf photo is not clear enough for a confident check, ask the farmer to retake a well-lit close-up photo of a single affected leaf, and advise consulting a local Sub-Assistant Agriculture Officer (উপ-সহকারী কৃষি কর্মকর্তা). Set `officer_referral` to true.
3. If `rag_sources` is empty (no verified BARI/BRRI handbook context provided yet), DO NOT invent or recommend any specific pesticide/fungicide chemical names or dosages. State clearly in Bangla that verified treatment and dosage guidance requires official BARI/BRRI handbook reference or confirmation from a local Sub-Assistant Agriculture Officer, and offer safe general field hygiene advice (e.g., inspecting nearby plants, removing badly affected leaves if appropriate, keeping field clean). Set `grounded_in_rag` to false and `officer_referral` to true.
4. Always provide a concise Bangla `ppe_reminder` reminding the farmer to wear gloves and a mask and wash hands with soap after handling any agricultural spray.
5. Output ONLY valid JSON matching the requested schema."""


def _candidate_models() -> list[str]:
    primary = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL
    candidates = [primary]
    for model in FALLBACK_GEMINI_MODELS:
        if model not in candidates:
            candidates.append(model)
    return candidates


def _generate_advisory_sync(
    crop: CropType,
    vision: VisionResult,
    weather: Optional[WeatherResult],
    rag_sources: list[RagSource],
    question_text: Optional[str],
) -> AdvisoryResult:
    from google import genai
    from google.genai import errors as genai_errors
    from google.genai import types

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured. Add it to your .env file to enable Bangla advisory generation."
        )

    client = genai.Client(api_key=api_key)

    context_payload = {
        "crop": crop,
        "crop_bangla": "ধান" if crop == "rice" else "আলু",
        "vision": vision.model_dump(),
        "question_text": question_text.strip() if question_text and question_text.strip() else None,
        "rag_sources": [src.model_dump() for src in rag_sources],
        "weather": weather.model_dump() if weather is not None else None,
    }

    user_prompt = (
        "Generate a farmer-friendly Bangla advisory JSON from the following analysis data:\n"
        f"{json.dumps(context_payload, ensure_ascii=False, indent=2)}"
    )

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        response_mime_type="application/json",
        response_schema=AdvisoryResult,
        temperature=0.3,
    )

    response = None
    last_exc: Optional[Exception] = None
    for model_name in _candidate_models():
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=user_prompt,
                config=config,
            )
            break
        except genai_errors.APIError as exc:
            last_exc = exc
            if exc.code in (404, 429, 503):
                logger.warning(
                    "Gemini model %s returned HTTP %s; trying fallback model.",
                    model_name,
                    exc.code,
                )
                continue
            raise

    if response is None:
        raise RuntimeError(f"All Gemini Flash models failed: {last_exc}") from last_exc

    raw_text = (response.text or "").strip()
    if not raw_text:
        raise RuntimeError("Gemini returned an empty advisory response.")

    advisory = AdvisoryResult.model_validate_json(raw_text)

    # Enforce strict safety invariants in code
    if not rag_sources:
        advisory.grounded_in_rag = False
        advisory.officer_referral = True
    if vision.status == "uncertain":
        advisory.officer_referral = True
    if not advisory.ppe_reminder.strip():
        advisory.ppe_reminder = (
            "যেকোনো বালাইনাশক ব্যবহারের সময় অবশ্যই গ্লাভস ও মাস্ক পরুন এবং কাজ শেষে সাবান দিয়ে ভালোভাবে হাত ধুয়ে নিন।"
        )

    return advisory


async def generate_advisory(
    crop: CropType,
    vision: VisionResult,
    weather: Optional[WeatherResult] = None,
    rag_sources: Optional[list[RagSource]] = None,
    audio_bytes: Optional[bytes] = None,
    audio_content_type: Optional[str] = None,
    question_text: Optional[str] = None,
) -> AdvisoryResult:
    """Generate a grounded, farmer-friendly Bangla advisory using Gemini Flash."""
    sources = rag_sources or []
    return await asyncio.to_thread(
        _generate_advisory_sync,
        crop,
        vision,
        weather,
        sources,
        question_text,
    )

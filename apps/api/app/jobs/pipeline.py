"""The analysis job.

Runs stages in order, persisting progress so `GET /analyses/{id}` can report it.
Each stage degrades independently: a weather timeout, a vision miss, or a TTS
failure never withholds the diagnosis or the curated treatment.

Vision reports; it does not decide. The classifier can name a leaf, decline to
name it, or be unavailable, and each of those three has an honest representation
downstream. What it must never do is supply a confidence nobody computed, which
is why the `uncertain` flag and the confidence the LLM sees are derived from the
classifier's actual output rather than assumed.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

from .. import store
from ..config import settings
from ..grounding import knowledge_base
from ..grounding.districts import get_district
from ..grounding.weather import fetch_weather
from ..llm.writer import InvalidNarrative, get_writer
from ..tts.engine import TtsSafetyError, get_engine
from ..vision import VisionFacts, get_classifier

log = logging.getLogger(__name__)

#: Which KB entry is used when the classifier cannot name the disease. Every
#: crop still gets real, curated treatment; the result says it is unconfirmed.
#: Rice and potato only: those are the two crops the shipped model can name.
FALLBACK_DISEASE = {
    "rice": "rice_blast",
    "potato": "potato_late_blight",
}

#: Confidence band boundaries, applied to the classifier's probability.
HIGH_CONFIDENCE = 0.75
LOW_CONFIDENCE = 0.45

CONCURRENCY = 2

#: Created per event loop. A module-level `Semaphore` binds to the loop that
#: first awaits it, so a second loop (uvicorn reload, pytest) would raise
#: "bound to a different event loop" instead of queueing.
_semaphores: dict[int, asyncio.Semaphore] = {}


def _semaphore_for_current_loop() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    semaphore = _semaphores.get(id(loop))
    if semaphore is None:
        semaphore = asyncio.Semaphore(CONCURRENCY)
        _semaphores[id(loop)] = semaphore
    return semaphore


#: Strong references to in-flight tasks. `asyncio` only holds weak references, so
#: a fire-and-forget task can be garbage-collected mid-run without this.
_tasks: set[asyncio.Task[None]] = set()


def spawn(analysis_id: str, crop_id: str, district_id: str | None) -> None:
    """Start a pipeline run in the background."""
    task = asyncio.create_task(run(analysis_id, crop_id, district_id))
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)


async def run(analysis_id: str, crop_id: str, district_id: str | None) -> None:
    """Run the whole pipeline. Never raises: failures are recorded on the row."""
    async with _semaphore_for_current_loop():
        try:
            await _pipeline(analysis_id, crop_id, district_id)
        except Exception as exc:
            log.exception("analysis %s failed", analysis_id)
            await store.mark_failed(analysis_id, exc.__class__.__name__)


async def _pipeline(analysis_id: str, crop_id: str, district_id: str | None) -> None:
    await store.update_stage(analysis_id, "reading", "processing")

    vision = await _classify(analysis_id, crop_id)
    disease = _resolve_disease(crop_id, vision)
    if disease is None:
        await store.mark_failed(analysis_id, "knowledge_base_miss")
        return

    await store.update_stage(analysis_id, "grounding")

    district = get_district(district_id) if district_id else None
    weather = None
    if district is not None:
        weather = await fetch_weather(
            latitude=float(district["latitude"]),  # type: ignore[arg-type]
            longitude=float(district["longitude"]),  # type: ignore[arg-type]
            district_id=str(district["id"]),
            district_name=str(district["name"]),
        )

    facts = _build_facts(disease, weather, vision)

    await store.update_stage(analysis_id, "writing")
    narrative, generated_by = await _write(facts)

    result = _build_result(
        analysis_id,
        disease,
        weather,
        narrative,
        generated_by,
        vision=vision,
        district_name=str(district["name"]) if district else None,
    )
    await store.save_result(analysis_id, result, disease_id=disease["id"])

    # Text is complete and served now. Speech continues in the background, so
    # nobody waits for a voice while reading advice that is already there -
    # `/audio` answers 202 until the file exists.
    if settings.tts_enabled:
        await store.update_stage(analysis_id, "speaking", "completed")
        task = asyncio.create_task(_speak(analysis_id, narrative, disease))
        _tasks.add(task)
        task.add_done_callback(_tasks.discard)
    else:
        await store.update_stage(analysis_id, "done", "completed")


async def _classify(analysis_id: str, crop_id: str) -> VisionFacts:
    """Run the classifier, guarding only against a missing image.

    The classifier itself never raises; this exists so that a deleted file after
    upload reads as "no model output" rather than a failed job.
    """
    image_path = settings.images_dir / f"{analysis_id}.jpg"
    if not image_path.exists():
        log.warning("analysis %s has no stored image; skipping vision", analysis_id)
        return VisionFacts(
            status="unavailable",
            label=None,
            confidence=None,
            crop_id=None,
            disease_id=None,
            is_healthy=False,
            model=None,
        )
    return await get_classifier().classify(image_path, crop_id)


def _resolve_disease(crop_id: str, vision: VisionFacts) -> knowledge_base.DiseaseEntry | None:
    """Pick the KB entry the advice will be built on.

    A classifier match wins — including a `healthy` match, which resolves to the
    crop's healthy entry and carries no chemical. Everything else falls back to
    one curated entry per crop and is marked unconfirmed downstream, so the
    farmer still gets correct treatment steps for their crop.
    """
    if vision["status"] == "identified" and vision["disease_id"]:
        matched = knowledge_base.get_disease(vision["disease_id"])
        if matched is not None:
            return matched
    fallback_id = FALLBACK_DISEASE.get(crop_id)
    return knowledge_base.get_disease(fallback_id) if fallback_id else None


def _confidence_band(confidence: float | None) -> str:
    if confidence is None:
        return "low"
    if confidence >= HIGH_CONFIDENCE:
        return "high"
    if confidence < LOW_CONFIDENCE:
        return "low"
    return "medium"


async def _write(facts: dict[str, Any]) -> tuple[str, str]:
    writer = get_writer()
    try:
        return await writer.write(facts)
    except (InvalidNarrative, Exception) as exc:  # noqa: BLE001
        log.warning("LLM write failed (%s); using template", exc.__class__.__name__)
        from ..llm.writer import TemplateWriter

        return await TemplateWriter().write(facts)


async def _speak(analysis_id: str, narrative: str, disease: knowledge_base.DiseaseEntry) -> None:
    """Synthesise in the background, then close out the stage.

    Every exit path moves the stage to `done`. Leaving a row at `speaking` after
    a TTS failure would mean a result that never looks finished.
    """
    try:
        await _speak_inner(analysis_id, narrative, disease)
    finally:
        await store.update_stage(analysis_id, "done", "completed")


async def _speak_inner(
    analysis_id: str, narrative: str, disease: knowledge_base.DiseaseEntry
) -> None:
    engine = get_engine()
    if not engine.available:
        await store.update_audio(analysis_id, {"status": "failed", "url": None})
        return

    await store.update_audio(analysis_id, {"status": "pending", "url": None})
    terms = [chemical["bnName"] for chemical in disease["chemical"]]
    try:
        clip = await engine.synthesise(narrative, settings.audio_dir, required_terms=terms)
    except TtsSafetyError as exc:
        log.error("TTS refused to synthesise for %s: %s", analysis_id, exc)
        await store.update_audio(analysis_id, {"status": "failed", "url": None})
        return
    except Exception as exc:  # noqa: BLE001
        log.warning("TTS failed for %s: %s", analysis_id, exc.__class__.__name__)
        await store.update_audio(analysis_id, {"status": "failed", "url": None})
        return

    await store.update_audio(
        analysis_id,
        {
            "status": "ready",
            "url": f"/api/v1/analyses/{analysis_id}/audio",
            "durationMs": clip.duration_ms,
            "byteSize": clip.byte_size,
            "file": clip.path.name,
        },
    )


def _build_facts(
    disease: knowledge_base.DiseaseEntry, weather: Any, vision: VisionFacts
) -> dict[str, Any]:
    """The exact shape handed to the LLM. Facts only, no instructions.

    `vision` is what the classifier actually returned. When it did not identify
    anything, the facts say so and carry no confidence, so the model cannot
    write "symptoms were observed at medium confidence" about a photo it never
    saw a result for.
    """
    identified = vision["status"] == "identified"
    healthy = bool(vision.get("is_healthy"))
    return {
        "diagnosis": {
            "crop": disease["cropName"],
            # The short name, so the narrative reads "ধানে ব্লাস্ট" rather than
            # "ধানে ধানের ব্লাস্ট". The full label is used in the result card.
            "disease": f"{disease['bnName']} রোগ" if not healthy else disease["bnName"],
            "confidence": _confidence_label(vision["confidence"]) if identified else None,
            "confirmed_by": "ছবি বিশ্লেষণ" if identified else None,
            "healthy": healthy,
        },
        # A healthy plant has no symptoms to describe; the entry's symptom list
        # is empty anyway, but being explicit stops a future edit from making
        # "no disease" read like one.
        "symptoms_observed": disease["symptoms"] if identified and not healthy else [],
        "treatment": {
            "organic": disease["organic"],
            "chemical": (
                [
                    {
                        "product": chemical["product"],
                        "bnName": chemical["bnName"],
                        "dose": chemical["dose"],
                        "note": chemical["note"],
                    }
                    for chemical in disease["chemical"]
                ]
                if identified and not healthy
                # No identified disease means no chemical reaches the model, so
                # it cannot phrase a dose the evidence does not support. A
                # healthy verdict is the strongest case of that: spraying a
                # healthy crop wastes money and exposes the farmer to risk.
                else []
            ),
        },
        "prevention": disease["prevention"],
        "weather": weather.as_prompt_facts() if weather else None,
        "uncertain": not identified,
    }


def _confidence_label(confidence: float | None) -> str:
    """Bangla band name, matching what the UI shows."""
    return {"high": "বেশি", "medium": "মাঝারি", "low": "কম"}[_confidence_band(confidence)]


def _build_result(
    analysis_id: str,
    disease: knowledge_base.DiseaseEntry,
    weather: Any,
    narrative: str,
    generated_by: str,
    *,
    vision: VisionFacts,
    district_name: str | None = None,
) -> dict[str, Any]:
    identified = vision["status"] == "identified"
    healthy = bool(vision.get("is_healthy"))
    return {
        "id": analysis_id,
        "createdAt": datetime.now(UTC).isoformat(),
        "imageUrl": f"/api/v1/analyses/{analysis_id}/image",
        "cropName": disease["cropName"],
        "status": "completed",
        "stage": "speaking" if settings.tts_enabled else "done",
        "diagnosis": {
            "diseaseLabel": disease["diseaseLabel"],
            "cropName": disease["cropName"],
            # Still null on a miss: the label shown is the fallback entry, and
            # attaching a number to it would be inventing evidence.
            "confidence": (round(vision["confidence"], 3) if identified else None),
            "confidenceBand": _confidence_band(vision["confidence"]) if identified else "low",
            "isAiEstimate": True,
        },
        "symptoms": disease["symptoms"] if identified and not healthy else [],
        "treatment": {
            "summary": (
                f"{disease['cropName']} ফসলের জন্য সাধারণ পরিচর্যা"
                if healthy
                else (
                    "নিচের ধাপগুলো অনুসরণ করুন"
                    if identified
                    else f"{disease['cropName']} ফসলের জন্য সাধারণ পরিচর্যা"
                )
            ),
            "steps": (
                [
                    f"{chemical['bnName']}: {chemical['dose']} ({chemical['note']})"
                    for chemical in disease["chemical"]
                ]
                + disease["organic"]
                if identified and not healthy
                # A dose is disease-specific. Listing one for a disease nobody
                # confirmed would be a precise instruction with no evidence
                # behind it, which is worse than no instruction at all.
                else disease["organic"]
            ),
            "source": disease["source"],
        },
        "prevention": disease["prevention"],
        "needsProfessionalVerification": True,
        "uncertainty": {
            "isUncertain": not identified,
            "message": (
                "এটি একটি আনুমানিক ফলাফল। নিশ্চিত হওয়ার জন্য কৃষি কর্মকর্তার পরামর্শ নিন।"
                if identified and not healthy
                else (
                    "আপনার গাছটি সুস্থ দেখাচ্ছে। নিয়মিত পরিচর্যা চালিয়ে যান এবং "
                    "রোগের লক্ষণ দেখা দিলে আবার ছবি দিন।"
                    if healthy
                    else "ছবি থেকে রোগ নিশ্চিতভাবে শনাক্ত করা যায়নি। নিচের পরামর্শটি "
                    f"{disease['cropName']} ফসলের সাধারণ নির্দেশনা। সঠিক রোগ নিশ্চিত করতে "
                    "আরও পরিষ্কার ছবি দিন, অথবা উপজেলা কৃষি কর্মকর্তার পরামর্শ নিন।"
                )
            ),
        },
        "narrative": narrative,
        "generatedBy": generated_by,
        "weather": (
            {
                "district": weather.district_name,
                "rainProbabilityNext12h": weather.rain_probability,
                "windSpeedMax12h": round(weather.wind_speed_max, 1),
                "humidity": weather.humidity,
                "sprayWindowOk": weather.spray_window_ok,
                "reason": weather.reason,
            }
            if weather
            else None
        ),
        "audio": {"status": "pending" if settings.tts_enabled else "none", "url": None},
        "district": district_name,
    }

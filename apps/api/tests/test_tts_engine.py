"""TTS engine concurrency and caching.

The model is not loaded here: `_synthesise_blocking` is replaced with a stub that
records how many calls overlap. What is being tested is the scheduling rule, not
the voice.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from app.tts import engine as engine_module
from app.tts.engine import AudioClip, TtsEngine


@pytest.fixture()
def engine(tmp_path, monkeypatch) -> TtsEngine:
    monkeypatch.setattr(engine_module.settings, "tts_enabled", True)
    monkeypatch.setattr(engine_module.settings, "ffmpeg_path", "ffmpeg")
    instance = TtsEngine()
    # Pretend the model is loaded so ensure_loaded() is a no-op.
    instance._tts = object()
    return instance


async def test_synthesis_is_serialised(engine: TtsEngine, tmp_path: Path, monkeypatch) -> None:
    active = 0
    peak = 0

    def fake_blocking(prepared: str, out_dir: Path, target: Path) -> AudioClip:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            # Stand in for the CPU-bound model call.
            import time

            time.sleep(0.12)
            target.write_bytes(b"fake")
            return AudioClip(target, 1000, 4, "audio/ogg")
        finally:
            active -= 1

    monkeypatch.setattr(engine, "_synthesise_blocking", fake_blocking)

    texts = [f"এটি পরামর্শ নম্বর {i} যা কিছুক্ষণ সময় নেবে" for i in range(4)]
    await asyncio.gather(*(engine.synthesise(text, tmp_path) for text in texts))

    assert peak == 1, f"expected serialised synthesis, saw {peak} concurrent calls"


async def test_identical_text_is_synthesised_once(
    engine: TtsEngine, tmp_path: Path, monkeypatch
) -> None:
    """Audio is content-addressed, so the same sentence never costs twice."""
    calls = 0

    def fake_blocking(prepared: str, out_dir: Path, target: Path) -> AudioClip:
        nonlocal calls
        calls += 1
        target.write_bytes(b"fake")
        return AudioClip(target, 1000, 4, "audio/ogg")

    monkeypatch.setattr(engine, "_synthesise_blocking", fake_blocking)

    text = "একই পরামর্শ দুইবার"
    await engine.synthesise(text, tmp_path)
    await engine.synthesise(text, tmp_path)

    assert calls == 1


async def test_concurrent_identical_requests_hit_cache(
    engine: TtsEngine, tmp_path: Path, monkeypatch
) -> None:
    """The re-check inside the lock means a waiter does not re-synthesise."""
    calls = 0

    def fake_blocking(prepared: str, out_dir: Path, target: Path) -> AudioClip:
        nonlocal calls
        calls += 1
        import time

        time.sleep(0.1)
        target.write_bytes(b"fake")
        return AudioClip(target, 1000, 4, "audio/ogg")

    monkeypatch.setattr(engine, "_synthesise_blocking", fake_blocking)

    text = "একসাথে ডাকা একই বাক্য"
    await asyncio.gather(*(engine.synthesise(text, tmp_path) for _ in range(4)))

    assert calls == 1, f"synthesised {calls} times for identical text"


async def test_prepare_runs_before_synthesis(
    engine: TtsEngine, tmp_path: Path, monkeypatch
) -> None:
    """The model must never see a raw slash or Latin text."""
    seen: list[str] = []

    def fake_blocking(prepared: str, out_dir: Path, target: Path) -> AudioClip:
        seen.append(prepared)
        target.write_bytes(b"fake")
        return AudioClip(target, 1000, 4, "audio/ogg")

    monkeypatch.setattr(engine, "_synthesise_blocking", fake_blocking)

    await engine.synthesise("মাত্রা ২ মিলি/লিটার", tmp_path)
    assert seen == ["মাত্রা ২ মিলি প্রতি লিটার"]

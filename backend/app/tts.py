"""TTS module for Bangla voice synthesis using edge-tts."""

import base64
import logging
import re
from typing import Optional

import edge_tts

logger = logging.getLogger(__name__)

BANGLA_VOICE = "bn-BD-NabanitaNeural"


def _sanitize_for_tts(text: str) -> str:
    """Strip markdown formatting and collapse whitespace for cleaner speech synthesis."""
    cleaned = re.sub(r"[*_`#>]+", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


async def synthesize_bangla_audio(bangla_text: str) -> Optional[str]:
    """Synthesize Bangla advisory text to an MP3 base64 data URI in memory.

    Returns `None` on failure so `/api/analyze` degrades gracefully.
    """
    text = _sanitize_for_tts(bangla_text or "")
    if not text:
        return None

    for attempt in range(2):
        try:
            communicate = edge_tts.Communicate(text=text, voice=BANGLA_VOICE)
            audio_buffer = bytearray()

            async for chunk in communicate.stream():
                if chunk.get("type") == "audio":
                    audio_buffer.extend(chunk["data"])

            if audio_buffer:
                encoded = base64.b64encode(bytes(audio_buffer)).decode("ascii")
                return f"data:audio/mpeg;base64,{encoded}"

            logger.warning(
                "edge-tts returned empty audio stream (attempt %d, voice=%s)",
                attempt + 1,
                BANGLA_VOICE,
            )
        except Exception:
            if attempt == 0:
                logger.warning(
                    "edge-tts attempt 1 failed (voice=%s); retrying once.", BANGLA_VOICE
                )
            else:
                logger.exception(
                    "Bangla TTS synthesis failed (edge-tts voice=%s).", BANGLA_VOICE
                )

    return None


# Alias kept for backward compatibility with initial scaffold interface
synthesize_bangla_speech = synthesize_bangla_audio

"""TTS module stub for Bangla voice synthesis (Gemini TTS / edge-tts)."""

from typing import Optional


async def synthesize_bangla_speech(bangla_text: str) -> Optional[str]:
    """Convert Bangla advisory text into playable audio (URL or base64 data URI).

    TODO (RAG/Weather/LLM Teammate):
      - Synthesize natural Bangla speech via Gemini TTS or `edge-tts`
        (e.g., `bn-BD-NabanitaNeural` / `bn-BD-PradeepNeural`).
      - Return a base64 data URI (`data:audio/mpeg;base64,...`) or served audio URL.
    """
    raise NotImplementedError(
        "Bangla TTS (Gemini TTS / edge-tts) is not yet implemented."
    )

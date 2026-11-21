"""LLM module stub for Google Gemini Flash (Bangla audio understanding + grounded advisory)."""

from typing import Optional

from app.schemas import AdvisoryResult, CropType, RagSource, VisionResult, WeatherResult


async def generate_advisory(
    crop: CropType,
    vision: VisionResult,
    weather: WeatherResult,
    rag_sources: list[RagSource],
    audio_bytes: Optional[bytes] = None,
    audio_content_type: Optional[str] = None,
    question_text: Optional[str] = None,
) -> AdvisoryResult:
    """Generate a grounded, farmer-friendly Bangla advisory using Gemini Flash.

    TODO (RAG/Weather/LLM Teammate):
      - Use `GEMINI_API_KEY` from environment variables.
      - Pass optional farmer Bangla voice `audio_bytes` and/or `question_text`,
        combined with `vision`, `weather`, and `rag_sources`.
      - Enforce grounding in `rag_sources`, include a PPE safety reminder, and set
        `officer_referral=True` when vision is uncertain or severe.
    """
    raise NotImplementedError(
        "LLM advisory generator (Gemini Flash Bangla audio + advisory) is not yet implemented."
    )

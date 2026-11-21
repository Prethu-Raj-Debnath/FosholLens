"""RAG module stub for BARI/BRRI JSON vector retrieval."""

from typing import Optional

from app.schemas import CropType, RagSource


async def retrieve_guidance(
    crop: CropType,
    disease_label: str,
    question_text: Optional[str] = None,
    top_k: int = 3,
) -> list[RagSource]:
    """Retrieve grounded agricultural guidance from BARI/BRRI JSON knowledge base.

    TODO (RAG/Weather/LLM Teammate):
      - Load curated BARI Krishi Projukti Hatboi & BRRI JSON corpus from `backend/data/`.
      - Perform vector/keyword retrieval filtered by `crop` and `disease_label`
        (augmented with `question_text` if provided).
      - Return up to `top_k` `RagSource` items.
    """
    raise NotImplementedError(
        "RAG retrieval (BARI/BRRI JSON knowledge base) is not yet implemented."
    )

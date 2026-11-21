"""Vision module stub for Rice & Potato leaf disease classification (ONNX / Hugging Face)."""

from typing import Optional

from app.schemas import CropType, VisionResult


async def classify_leaf(
    image_bytes: bytes,
    crop: CropType,
    content_type: Optional[str] = None,
) -> VisionResult:
    """Classify a rice or potato leaf image and return top predictions with confidence.

    TODO (Vision Teammate):
      - Load ONNX / Hugging Face plant-disease classifier weights for `rice` and `potato`.
      - Preprocess `image_bytes` (resize, normalize, RGB conversion).
      - Return top-k predictions and set `status="confident"` or `"uncertain"`
        based on the confidence threshold.
    """
    raise NotImplementedError(
        "Vision classifier (Rice/Potato ONNX/HF model) is not yet implemented."
    )

"""Vision module for Rice & Potato leaf disease classification using Hugging Face ViT."""

import asyncio
import io
import logging
import re
import threading
from typing import Optional

from PIL import Image, UnidentifiedImageError

from app.schemas import CropType, PredictionItem, VisionResult

logger = logging.getLogger(__name__)

MODEL_ID = "wambugu71/crop_leaf_diseases_vit"
CONFIDENCE_THRESHOLD = 0.55

# Canonical labels per crop matching wambugu71/crop_leaf_diseases_vit classes
CROP_CANONICAL_LABELS: dict[CropType, dict[str, str]] = {
    "potato": {
        "potato early blight": "Potato Early Blight",
        "potato late blight": "Potato Late Blight",
        "potato healthy": "Potato Healthy",
    },
    "rice": {
        "rice leaf blast": "Rice Leaf Blast",
        "rice brown spot": "Rice Brown Spot",
        "rice healthy": "Rice Healthy",
        "rice hispa": "Rice Hispa",
    },
}

_model_lock = threading.Lock()
_processor = None
_model = None


def _normalize_label(raw_label: str) -> str:
    """Normalize Hugging Face id2label strings for reliable crop matching."""
    cleaned = re.sub(r"[_\-\s]+", " ", raw_label).strip().lower()
    return cleaned


def _get_model_and_processor():
    """Lazily load and cache the ViT processor and model on CPU."""
    global _processor, _model
    if _model is not None and _processor is not None:
        return _processor, _model

    with _model_lock:
        if _model is None or _processor is None:
            from transformers import AutoImageProcessor, AutoModelForImageClassification

            logger.info("Loading Hugging Face vision model: %s", MODEL_ID)
            _processor = AutoImageProcessor.from_pretrained(MODEL_ID)
            _model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
            _model.eval()
            logger.info("Vision model %s loaded successfully.", MODEL_ID)

    return _processor, _model


def _classify_leaf_sync(image_bytes: bytes, crop: CropType) -> VisionResult:
    """Run synchronous CPU inference and filter/renormalize to the selected crop."""
    import torch

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Uploaded file is not a valid image.") from exc

    processor, model = _get_model_and_processor()
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0].cpu().tolist()

    id2label = model.config.id2label
    crop_label_map = CROP_CANONICAL_LABELS[crop]

    matched_scores: list[tuple[str, float]] = []
    for idx, prob in enumerate(probs):
        raw_label = id2label.get(idx) or id2label.get(str(idx)) or ""
        norm_label = _normalize_label(raw_label)

        canonical = crop_label_map.get(norm_label)
        if canonical is None and norm_label.startswith(crop):
            # Fallback if a model variant uses slightly different spacing/punctuation
            for key, target in crop_label_map.items():
                key_tokens = set(key.split())
                if key_tokens.issubset(set(norm_label.split())):
                    canonical = target
                    break

        if canonical is not None:
            matched_scores.append((canonical, float(prob)))

    if not matched_scores:
        raise RuntimeError(
            f"No model classes matched crop '{crop}' in {MODEL_ID} labels: {list(id2label.values())}"
        )

    # Renormalize probabilities across the selected crop's classes
    total_prob = sum(score for _, score in matched_scores)
    if total_prob > 0.0:
        renormalized = [(label, score / total_prob) for label, score in matched_scores]
    else:
        uniform = 1.0 / len(matched_scores)
        renormalized = [(label, uniform) for label, _ in matched_scores]

    renormalized.sort(key=lambda item: item[1], reverse=True)
    top_3 = [
        PredictionItem(label=label, confidence=round(min(max(conf, 0.0), 1.0), 4))
        for label, conf in renormalized[:3]
    ]

    top_confidence = top_3[0].confidence if top_3 else 0.0
    status = "uncertain" if top_confidence < CONFIDENCE_THRESHOLD else "confident"

    return VisionResult(top_predictions=top_3, status=status)


async def classify_leaf(
    image_bytes: bytes,
    crop: CropType,
    content_type: Optional[str] = None,
) -> VisionResult:
    """Classify a rice or potato leaf image and return top-3 predictions."""
    return await asyncio.to_thread(_classify_leaf_sync, image_bytes, crop)

"""Leaf disease classification.

A ViT fine-tuned on corn/potato/rice/wheat leaves, run through `onnxruntime`.

Why ONNX and not `transformers`: the model ships both, but the PyTorch path
costs a ~900 MB torch wheel for a 22 MB model. `onnxruntime` is already a
dependency here (it powers the TTS engine), so this adds no install weight and
the weights download lazily on first use, like the TTS model does.
"""

from __future__ import annotations

import asyncio
import io
import logging
from pathlib import Path
from typing import Any, Literal, TypedDict

from ..config import settings

log = logging.getLogger(__name__)

#: Emission order of the model's 13 classes. Fixed by the exported graph: the
#: output tensor is `[batch, 13]` with no label metadata attached, so this order
#: is part of the model's contract, not a guess. It matches `id2label` in the
#: upstream `config.json`.
LABELS: tuple[str, ...] = (
    "Corn___Common_Rust",
    "Corn___Gray_Leaf_Spot",
    "Corn___Healthy",
    "Invalid",
    "Potato___Early_Blight",
    "Potato___Healthy",
    "Potato___Late_Blight",
    "Rice___Brown_Spot",
    "Rice___Healthy",
    "Rice___Leaf_Blast",
    "Wheat___Brown_Rust",
    "Wheat___Healthy",
    "Wheat___Yellow_Rust",
)

#: Model label -> the crop that label belongs to.
LABEL_CROP: dict[str, str] = {
    "Corn___Common_Rust": "corn",
    "Corn___Gray_Leaf_Spot": "corn",
    "Corn___Healthy": "corn",
    "Potato___Early_Blight": "potato",
    "Potato___Healthy": "potato",
    "Potato___Late_Blight": "potato",
    "Rice___Brown_Spot": "rice",
    "Rice___Healthy": "rice",
    "Rice___Leaf_Blast": "rice",
    "Wheat___Brown_Rust": "wheat",
    "Wheat___Healthy": "wheat",
    "Wheat___Yellow_Rust": "wheat",
}

#: Model label -> curated knowledge-base entry.
#:
#: Deliberately incomplete. The model can name `Rice___Brown_Spot`, but the
#: knowledge base has no curated treatment for it, and inventing a dose for a
#: disease we have no source for is exactly the failure this project exists to
#: avoid. Unmapped labels degrade to "not confidently identified" instead.
LABEL_DISEASE: dict[str, str] = {
    "Rice___Leaf_Blast": "rice_blast",
    "Potato___Early_Blight": "potato_early_blight",
    "Potato___Late_Blight": "potato_late_blight",
}

HEALTHY_LABELS = frozenset(
    {"Corn___Healthy", "Potato___Healthy", "Rice___Healthy", "Wheat___Healthy"}
)

#: A leaf that is genuinely healthy would land here; so would a photo the model
#: cannot place. Neither is a diagnosis.
NON_DIAGNOSTIC_LABELS = frozenset({"Invalid"}) | HEALTHY_LABELS

#: Crops this model can say anything about at all.
SUPPORTED_CROPS = frozenset({"rice", "potato"})

INPUT_SIZE = 224
IMAGE_MEAN = 0.5
IMAGE_STD = 0.5

VisionStatus = Literal["identified", "undetermined", "unavailable"]


class VisionFacts(TypedDict):
    """What the classifier saw. Never asserts a disease on its own."""

    status: VisionStatus
    label: str | None
    #: Softmax probability of the winning class, over all 13 classes.
    confidence: float | None
    crop_id: str | None
    disease_id: str | None


class LeafClassifier:
    """Lazy, serialised ONNX inference.

    `onnxruntime` sessions are thread-safe for `run`, but `_session` construction
    is not, so loading takes the same lock the TTS engine uses for its weights.
    """

    def __init__(self) -> None:
        self._session: Any = None
        self._load_lock = asyncio.Lock()

    @property
    def available(self) -> bool:
        return settings.vision_enabled

    def _load(self) -> Any:
        """Download (once) and open the ONNX graph. Blocking."""
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            settings.vision_model_id,
            filename=settings.vision_model_file,
        )
        session = ort.InferenceSession(
            str(path),
            providers=["CPUExecutionProvider"],
            sess_options=_session_options(ort),
        )
        log.info("vision model loaded: %s", settings.vision_model_id)
        return session

    async def ensure_loaded(self) -> None:
        if self._session is not None:
            return
        async with self._load_lock:
            if self._session is None:
                self._session = await asyncio.to_thread(self._load)

    async def classify(self, image_path: Path, crop_id: str) -> VisionFacts:
        """Classify one stored image for one crop.

        Never raises: a missing model, unreadable image, or inference error all
        return `unavailable`, because vision is the one stage that may degrade
        without withholding the curated advice behind it.
        """
        if not self.available:
            return _unavailable()

        try:
            await self.ensure_loaded()
            return await asyncio.to_thread(self._classify_sync, image_path, crop_id)
        except Exception as exc:  # noqa: BLE001 - vision must never fail the job
            log.warning(
                "vision failed for %s: %s: %s",
                image_path.name,
                exc.__class__.__name__,
                str(exc) or "no detail",
            )
            return _unavailable()

    def _classify_sync(self, image_path: Path, crop_id: str) -> VisionFacts:
        import numpy as np
        from PIL import Image

        try:
            with Image.open(image_path) as opened:
                image = opened.convert("RGB").resize(
                    (INPUT_SIZE, INPUT_SIZE), Image.Resampling.BICUBIC
                )
        except Exception as exc:  # noqa: BLE001
            log.warning("vision image unreadable: %s", exc.__class__.__name__)
            return _unavailable()

        batch = _to_tensor(np, image)
        logits = self._session.run(None, {"pixel_values": batch})[0][0]

        probabilities = _softmax(np, logits)
        return _decide(probabilities, crop_id)


def _session_options(ort: Any) -> Any:
    options = ort.SessionOptions()
    # One image at a time, two pipeline slots. Extra threads would contend with
    # the TTS synth that shares this process.
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    return options


def _to_tensor(np: Any, image: Any) -> Any:
    """HWC uint8 -> NCHW float32, rescaled 1/255 then normalised by 0.5/0.5.

    Matches the model's `preprocessor_config.json`; getting this wrong does not
    error, it just quietly degrades accuracy.
    """
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - IMAGE_MEAN) / IMAGE_STD
    return np.ascontiguousarray(array.transpose(2, 0, 1)[None])


def _softmax(np: Any, logits: Any) -> Any:
    shifted = logits - logits.max()
    exponents = np.exp(shifted)
    return exponents / exponents.sum()


def _decide(probabilities: Any, crop_id: str) -> VisionFacts:
    """Turn probabilities into facts, applying the threshold to raw mass.

    The threshold is deliberately compared against the *unfiltered* probability
    of the winning class. Renormalising onto the requested crop's classes first
    and then thresholding — the tempting shortcut — lets a corn leaf clear 0.55
    as "rice blast" whenever the rice classes merely outscore the corn ones,
    because the renormalisation throws away the evidence that the whole photo
    was a different crop.
    """
    index = int(probabilities.argmax())
    label = LABELS[index]
    confidence = float(probabilities[index])

    if confidence < settings.vision_confidence_threshold:
        return _undetermined(label, confidence)

    if label in NON_DIAGNOSTIC_LABELS:
        return _undetermined(label, confidence)

    label_crop = LABEL_CROP.get(label)
    if label_crop != crop_id:
        # Wrong crop in frame, or a crop this model cannot name. Either way the
        # answer says nothing about the leaf the farmer actually photographed.
        return _undetermined(label, confidence)

    disease_id = LABEL_DISEASE.get(label)
    if disease_id is None:
        return _undetermined(label, confidence)

    return VisionFacts(
        status="identified",
        label=label,
        confidence=confidence,
        crop_id=label_crop,
        disease_id=disease_id,
    )


def _undetermined(label: str | None, confidence: float | None) -> VisionFacts:
    return VisionFacts(
        status="undetermined",
        label=label,
        confidence=confidence,
        crop_id=None,
        disease_id=None,
    )


def _unavailable() -> VisionFacts:
    return VisionFacts(
        status="unavailable",
        label=None,
        confidence=None,
        crop_id=None,
        disease_id=None,
    )


_engine: LeafClassifier | None = None


def get_classifier() -> LeafClassifier:
    global _engine
    if _engine is None:
        _engine = LeafClassifier()
    return _engine


def reset_classifier() -> None:
    """Drop the cached model. Used by tests after changing settings."""
    global _engine
    _engine = None


def image_bytes_to_path_source(raw: bytes) -> io.BytesIO:  # pragma: no cover
    """Kept for symmetry with callers that hold bytes rather than a path."""
    return io.BytesIO(raw)

"""Model specs for the leaf classifier.

A spec is everything that differs between checkpoints: the label order, how to
map a label onto a crop and a knowledge-base entry, and how to preprocess an
image. The decision rule in `classifier.py` is shared, so the two models cannot
drift apart in how they decide to *refuse* — only in what they can name.

Two checkpoints are supported, chosen with `VISION_MODEL`:

``vit``
    ViT-tiny over corn/potato/rice/wheat, 13 classes, downloaded from Hugging
    Face on first use. Broadest coverage, but most of it is for crops the
    knowledge base has nothing to say about.

``efficientnet``
    EfficientNet-B0 trained on rice and potato only, 9 classes, loaded from a
    local directory. Narrower, but it is the only one that can name rice
    bacterial leaf blight, and it is roughly three times faster.

Neither is strictly better. `vit` can say "that looks like corn"; `efficientnet`
can name one more disease you actually have curated treatment for.
"""

from __future__ import annotations

from typing import Literal, TypedDict

ModelId = Literal["vit", "efficientnet"]


class ModelSpec(TypedDict):
    """Everything about a checkpoint except its bytes."""

    id: str
    labels: tuple[str, ...]
    #: Model label -> crop. A label naming a different crop than the one the
    #: farmer selected is treated as "no answer", not as a diagnosis.
    label_crop: dict[str, str]
    #: Model label -> curated knowledge-base entry. Labels absent here are
    #: recognised by the model but have no sourced treatment, so they must not
    #: become a diagnosis.
    label_disease: dict[str, str]
    #: Healthy classes. Correct, but not a disease.
    healthy: frozenset[str]
    #: Classes that mean "this is not a disease I can name".
    non_diagnostic: frozenset[str]
    input_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    #: ONNX input node name. Part of the graph, not a convention.
    input_name: str
    #: Filename to look for, both locally and on the Hub.
    file: str
    #: Hugging Face repo to fetch from. Empty means "must already exist in
    #: `settings.vision_model_dir`" — a team-trained checkpoint is not published
    #: anywhere, so guessing a download location for it would fail confusingly.
    hf_repo: str


# --------------------------------------------------------------------------
# ViT-tiny, 13 classes: corn / potato / rice / wheat + an `Invalid` catch-all.
# --------------------------------------------------------------------------
VIT_LABELS: tuple[str, ...] = (
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

_VIT_LABEL_CROP: dict[str, str] = {
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

_VIT_LABEL_DISEASE: dict[str, str] = {
    "Rice___Leaf_Blast": "rice_blast",
    "Potato___Early_Blight": "potato_early_blight",
    "Potato___Late_Blight": "potato_late_blight",
}

VIT: ModelSpec = {
    "id": "vit",
    "labels": VIT_LABELS,
    "label_crop": _VIT_LABEL_CROP,
    "label_disease": _VIT_LABEL_DISEASE,
    "healthy": frozenset(
        {"Corn___Healthy", "Potato___Healthy", "Rice___Healthy", "Wheat___Healthy"}
    ),
    "non_diagnostic": frozenset({"Invalid"}),
    "input_size": 224,
    "mean": (0.5, 0.5, 0.5),
    "std": (0.5, 0.5, 0.5),
    "input_name": "pixel_values",
    "file": "crop_leaf_diseases_vit.onnx",
    "hf_repo": "wambugu71/crop_leaf_diseases_vit_onnx",
}


# --------------------------------------------------------------------------
# EfficientNet-B0, 9 classes: rice and potato only.
# --------------------------------------------------------------------------
EFFICIENTNET_LABELS: tuple[str, ...] = (
    "potato_early_blight",
    "potato_healthy",
    "potato_late_blight",
    "rice_bacterial_leaf_blight",
    "rice_blast",
    "rice_brown_spot",
    "rice_healthy",
    "rice_other",
    "rice_tungro",
)

_EFFICIENTNET_LABEL_CROP: dict[str, str] = {
    "potato_early_blight": "potato",
    "potato_healthy": "potato",
    "potato_late_blight": "potato",
    "rice_bacterial_leaf_blight": "rice",
    "rice_blast": "rice",
    "rice_brown_spot": "rice",
    "rice_healthy": "rice",
    "rice_other": "rice",
    "rice_tungro": "rice",
}

_EFFICIENTNET_LABEL_DISEASE: dict[str, str] = {
    "potato_early_blight": "potato_early_blight",
    "potato_late_blight": "potato_late_blight",
    "rice_blast": "rice_blast",
    # The checkpoint spells this `_leaf_blight`; the knowledge base calls the
    # entry `rice_bacterial_blight`. Mapped explicitly rather than by string
    # surgery, so a rename on either side fails a test instead of going quiet.
    "rice_bacterial_leaf_blight": "rice_bacterial_blight",
}

EFFICIENTNET: ModelSpec = {
    "id": "efficientnet",
    "labels": EFFICIENTNET_LABELS,
    "label_crop": _EFFICIENTNET_LABEL_CROP,
    "label_disease": _EFFICIENTNET_LABEL_DISEASE,
    "healthy": frozenset({"potato_healthy", "rice_healthy"}),
    # `rice_other` is the checkpoint's own "something else" class. `rice_tungro`
    # is a real disease but the knowledge base has no entry for it, so it falls
    # through the missing-`label_disease` check rather than being listed twice.
    "non_diagnostic": frozenset({"rice_other"}),
    "input_size": 320,
    "mean": (0.485, 0.456, 0.406),
    "std": (0.229, 0.224, 0.225),
    "input_name": "input",
    "file": "foshollens_leaf.onnx",
    "hf_repo": "",
}


SPECS: dict[ModelId, ModelSpec] = {
    "vit": VIT,
    "efficientnet": EFFICIENTNET,
}


def get_spec(model_id: str) -> ModelSpec:
    try:
        return SPECS[model_id]  # type: ignore[index]
    except KeyError:
        raise ValueError(
            f"unknown vision model {model_id!r}; expected one of {sorted(SPECS)}"
        ) from None

"""The classifier's decision logic.

These tests exercise `_decide` with injected probabilities instead of running
the model: the weights are a 16-22 MB download, and the interesting decisions
are all about what to do with a probability vector, not about computing one.

The regression this suite exists for is the renormalisation bug: filtering onto
the requested crop's classes *before* applying the confidence threshold lets a
corn leaf clear 0.55 as "rice blast", because the evidence that the photo was
the wrong crop is thrown away by the renormalisation itself.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.config import settings
from app.vision.classifier import LeafClassifier, _decide, _softmax, missing_weights
from app.vision.models import EFFICIENTNET, SPECS, VIT, ModelSpec, get_spec


def _probs(spec: ModelSpec, **weights: float) -> np.ndarray:
    """Build a probability vector by label name, zero elsewhere."""
    vector = np.zeros(len(spec["labels"]), dtype=np.float32)
    for label, weight in weights.items():
        vector[spec["labels"].index(label)] = weight
    return vector


# ---------------------------------------------------------------- vit ------
def test_vit_labels_match_the_exported_graph() -> None:
    assert len(VIT["labels"]) == 13
    # Order is part of the model contract, so a silent reorder is a bug.
    assert VIT["labels"][9] == "Rice___Leaf_Blast"
    assert VIT["labels"][3] == "Invalid"
    assert VIT["input_name"] == "pixel_values"
    assert VIT["input_size"] == 224


def test_confident_rice_blast_resolves_to_the_kb_entry() -> None:
    facts = _decide(_probs(VIT, Rice___Leaf_Blast=0.9, Invalid=0.05), "rice", VIT)
    assert facts["status"] == "identified"
    assert facts["disease_id"] == "rice_blast"
    assert facts["confidence"] == pytest.approx(0.9)
    assert facts["model"] == "vit"


def test_below_threshold_is_undetermined() -> None:
    facts = _decide(_probs(VIT, Rice___Leaf_Blast=0.4), "rice", VIT)
    assert facts["status"] == "undetermined"
    assert facts["disease_id"] is None
    # The raw number is kept for the UI, but never promoted to a diagnosis.
    assert facts["confidence"] == pytest.approx(0.4)


def test_threshold_is_applied_to_raw_mass_not_renormalised() -> None:
    """A corn leaf must not become a confident rice diagnosis.

    `Rice___Leaf_Blast` is the highest *rice* class here, but the crop mass is
    tiny. Renormalising onto rice first — then thresholding — would report
    ~0.75 confidence; the raw probability is 0.015 and must be judged as such.
    """
    facts = _decide(
        _probs(VIT, Corn___Common_Rust=0.5, Corn___Gray_Leaf_Spot=0.48, Rice___Leaf_Blast=0.015),
        "rice",
        VIT,
    )
    assert facts["status"] == "undetermined"
    assert facts["disease_id"] is None


def test_winning_label_from_another_crop_is_undetermined() -> None:
    facts = _decide(_probs(VIT, Corn___Common_Rust=0.9), "rice", VIT)
    assert facts["status"] == "undetermined"
    assert facts["confidence"] == pytest.approx(0.9)


@pytest.mark.parametrize("label", ["Invalid", "Rice___Healthy", "Potato___Healthy"])
def test_vit_non_diagnostic_labels_are_never_a_disease(label: str) -> None:
    facts = _decide(_probs(VIT, **{label: 0.99}), "rice", VIT)
    assert facts["status"] == "undetermined"
    assert facts["disease_id"] is None


def test_vit_known_label_without_a_kb_entry_is_not_invented() -> None:
    """The model can name brown spot; the knowledge base has no entry for it.

    Emitting a treatment anyway would mean writing a dose we have no source for.
    """
    assert "Rice___Brown_Spot" not in VIT["label_disease"]
    facts = _decide(_probs(VIT, Rice___Brown_Spot=0.9), "rice", VIT)
    assert facts["status"] == "undetermined"


# -------------------------------------------------------- efficientnet -----
def test_efficientnet_labels_match_the_shipped_checkpoint() -> None:
    assert len(EFFICIENTNET["labels"]) == 9
    assert EFFICIENTNET["input_name"] == "input"
    assert EFFICIENTNET["input_size"] == 320
    # ImageNet statistics, not the 0.5/0.5 the ViT uses.
    assert EFFICIENTNET["mean"] == (0.485, 0.456, 0.406)


def test_efficientnet_covers_bacterial_blight() -> None:
    """The one disease it can name that the ViT cannot."""
    facts = _decide(_probs(EFFICIENTNET, rice_bacterial_leaf_blight=0.88), "rice", EFFICIENTNET)
    assert facts["status"] == "identified"
    # Checkpoint spelling differs from the KB id; the mapping is explicit.
    assert facts["disease_id"] == "rice_bacterial_blight"


def test_efficientnet_rice_other_is_not_a_disease() -> None:
    facts = _decide(_probs(EFFICIENTNET, rice_other=0.9), "rice", EFFICIENTNET)
    assert facts["status"] == "undetermined"


def test_efficientnet_tungro_has_no_kb_entry() -> None:
    facts = _decide(_probs(EFFICIENTNET, rice_tungro=0.9), "rice", EFFICIENTNET)
    assert facts["status"] == "undetermined"


def test_efficientnet_respects_the_requested_crop() -> None:
    """A potato label must not answer a rice question.

    `predict.py` in the delivered zip ignored the crop entirely, so this is the
    behaviour the app adds on top of it.
    """
    facts = _decide(_probs(EFFICIENTNET, potato_late_blight=0.95), "rice", EFFICIENTNET)
    assert facts["status"] == "undetermined"


@pytest.mark.parametrize("label", ["rice_healthy", "potato_healthy"])
def test_efficientnet_healthy_is_not_dressed_up_as_disease(label: str) -> None:
    crop = "potato" if label.startswith("potato") else "rice"
    facts = _decide(_probs(EFFICIENTNET, **{label: 0.95}), crop, EFFICIENTNET)
    assert facts["status"] == "undetermined"


def test_crops_without_classes_in_either_model_are_undetermined() -> None:
    """Tomato and chili are out of scope for both checkpoints."""
    for crop in ("tomato", "chili"):
        for spec in (VIT, EFFICIENTNET):
            label = spec["labels"][0]
            facts = _decide(_probs(spec, **{label: 0.95}), crop, spec)
            assert facts["status"] == "undetermined", (spec["id"], crop)


# ------------------------------------------------------------- shared ------
def test_threshold_comes_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "vision_confidence_threshold", 0.95)
    assert _decide(_probs(VIT, Rice___Leaf_Blast=0.9), "rice", VIT)["status"] == "undetermined"

    monkeypatch.setattr(settings, "vision_confidence_threshold", 0.2)
    assert _decide(_probs(VIT, Rice___Leaf_Blast=0.3), "rice", VIT)["status"] == "identified"


def test_softmax_is_numerically_stable() -> None:
    """Large logits must not overflow to nan."""
    logits = np.array([1000.0, 1001.0, 999.0], dtype=np.float32)
    probabilities = _softmax(logits)
    assert not np.isnan(probabilities).any()
    assert probabilities.sum() == pytest.approx(1.0, abs=1e-6)


def test_both_specs_only_map_labels_onto_real_kb_entries() -> None:
    """A typo in a label_to_disease table would otherwise fail silently."""
    from app.grounding import knowledge_base

    for spec in SPECS.values():
        for label, disease_id in spec["label_disease"].items():
            assert label in spec["labels"], f"{spec['id']}: {label} not in labels"
            assert knowledge_base.get_disease(disease_id) is not None, (
                f"{spec['id']}: {label} -> {disease_id} has no KB entry"
            )


def test_unknown_model_id_is_rejected_clearly() -> None:
    with pytest.raises(ValueError, match="unknown vision model"):
        get_spec("resnet")


# ------------------------------------------------------- health truthfulness
def test_status_is_ready_only_when_the_local_checkpoint_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bug this guards: `VISION_ENABLED=true` used to mean `vision: ready`.

    A fresh clone has the flag set and no weights, so `/health` claimed the
    stage was fine while every upload silently returned `undetermined`.
    """
    monkeypatch.setattr(settings, "vision_enabled", True)

    monkeypatch.setattr(settings, "vision_model_dir", tmp_path)
    assert LeafClassifier(EFFICIENTNET).status == "unavailable"
    assert missing_weights(EFFICIENTNET) == tmp_path / EFFICIENTNET["file"]

    (tmp_path / EFFICIENTNET["file"]).write_bytes(b"not really a model")
    assert LeafClassifier(EFFICIENTNET).status == "ready"
    assert missing_weights(EFFICIENTNET) is None


def test_status_is_unavailable_when_vision_is_disabled(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "vision_enabled", False)
    monkeypatch.setattr(settings, "vision_model_dir", tmp_path)
    (tmp_path / EFFICIENTNET["file"]).write_bytes(b"not really a model")
    assert LeafClassifier(EFFICIENTNET).status == "unavailable"


def test_hub_checkpoint_is_not_reported_missing_offline(monkeypatch) -> None:
    """`/health` must not hit the network; a Hub spec is not probed."""
    monkeypatch.setattr(settings, "vision_enabled", True)
    assert missing_weights(VIT) is None
    assert LeafClassifier(VIT).status == "ready"


def test_relative_model_dir_is_anchored_to_the_api_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`.env.example` documents `./models`; that must not be CWD-relative.

    A literal reading resolves against wherever uvicorn was launched from, so
    the same `.env` worked under `npm run dev:api` (which `cd`s into apps/api)
    and silently found nothing from the repo root.
    """
    from app.config import API_ROOT, Settings

    env = tmp_path / ".env"
    env.write_text("VISION_MODEL_DIR=./models\n", encoding="utf-8")
    parsed = Settings(_env_file=env)

    assert parsed.vision_model_dir.is_absolute()
    assert parsed.vision_model_dir == (API_ROOT / "models").resolve()


def test_absolute_model_dir_is_left_alone(tmp_path: Path) -> None:
    from app.config import Settings

    env = tmp_path / ".env"
    env.write_text(f"VISION_MODEL_DIR={tmp_path / 'weights'}\n", encoding="utf-8")
    assert Settings(_env_file=env).vision_model_dir == (tmp_path / "weights").resolve()

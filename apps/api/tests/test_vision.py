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


def test_vit_invalid_label_is_never_a_disease() -> None:
    facts = _decide(_probs(VIT, Invalid=0.99), "rice", VIT)
    assert facts["status"] == "undetermined"
    assert facts["disease_id"] is None


def test_vit_healthy_labels_resolve_to_a_healthy_answer() -> None:
    """The deprecated ViT still carries healthy classes; they are a real answer."""
    facts = _decide(_probs(VIT, Rice___Healthy=0.9), "rice", VIT)
    assert facts["status"] == "identified"
    assert facts["is_healthy"] is True
    assert facts["disease_id"] == "rice_healthy"


def test_vit_corn_and_wheat_have_no_kb_entry() -> None:
    """The deprecated ViT can name crops this app does not support.

    Emitting a treatment anyway would mean writing a dose we have no source for.
    """
    for label in ("Corn___Common_Rust", "Wheat___Brown_Rust"):
        assert label not in VIT["label_disease"]
        facts = _decide(_probs(VIT, **{label: 0.9}), "rice", VIT)
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


def test_efficientnet_tungro_resolves_to_its_kb_entry() -> None:
    """Tungro used to be unmapped; it now has curated advice."""
    facts = _decide(_probs(EFFICIENTNET, rice_tungro=0.9), "rice", EFFICIENTNET)
    assert facts["status"] == "identified"
    assert facts["disease_id"] == "rice_tungro"
    assert facts["is_healthy"] is False


def test_efficientnet_brown_spot_resolves_to_its_kb_entry() -> None:
    """The regression that started this: brown spot was recognised and dropped."""
    facts = _decide(_probs(EFFICIENTNET, rice_brown_spot=0.96), "rice", EFFICIENTNET)
    assert facts["status"] == "identified"
    assert facts["disease_id"] == "rice_brown_spot"
    assert facts["confidence"] == pytest.approx(0.96)


@pytest.mark.parametrize(
    ("label", "crop", "disease_id"),
    [
        ("rice_blast", "rice", "rice_blast"),
        ("rice_bacterial_leaf_blight", "rice", "rice_bacterial_blight"),
        ("rice_brown_spot", "rice", "rice_brown_spot"),
        ("rice_tungro", "rice", "rice_tungro"),
        ("potato_early_blight", "potato", "potato_early_blight"),
        ("potato_late_blight", "potato", "potato_late_blight"),
    ],
)
def test_every_efficientnet_disease_label_is_surfaced(
    label: str, crop: str, disease_id: str
) -> None:
    """Every disease class the shipped model can emit must reach the farmer.

    This is the point of the change: the model naming a disease the app has
    advice for must never be swallowed into a generic "could not identify".
    """
    facts = _decide(_probs(EFFICIENTNET, **{label: 0.9}), crop, EFFICIENTNET)
    assert facts["status"] == "identified"
    assert facts["disease_id"] == disease_id


def test_efficientnet_respects_the_requested_crop() -> None:
    """A potato label must not answer a rice question.

    `predict.py` in the delivered zip ignored the crop entirely, so this is the
    behaviour the app adds on top of it.
    """
    facts = _decide(_probs(EFFICIENTNET, potato_late_blight=0.95), "rice", EFFICIENTNET)
    assert facts["status"] == "undetermined"


@pytest.mark.parametrize(
    ("label", "crop", "disease_id"),
    [("rice_healthy", "rice", "rice_healthy"), ("potato_healthy", "potato", "potato_healthy")],
)
def test_efficientnet_healthy_is_surfaced_as_a_healthy_answer(
    label: str, crop: str, disease_id: str
) -> None:
    """Healthy is a real result, not a miss.

    It resolves to the crop's healthy entry, which carries no chemical, so the
    farmer is told the plant looks fine instead of being left with "could not
    identify" or — worse — a spray.
    """
    facts = _decide(_probs(EFFICIENTNET, **{label: 0.95}), crop, EFFICIENTNET)
    assert facts["status"] == "identified"
    assert facts["is_healthy"] is True
    assert facts["disease_id"] == disease_id

    from app.grounding import knowledge_base

    entry = knowledge_base.get_disease(disease_id)
    assert entry is not None
    assert entry["chemical"] == []


def test_crops_outside_rice_and_potato_are_undetermined() -> None:
    """Tomato and chili are out of scope: no checkpoint can name them."""
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


# ------------------------------------------------------------ coverage ------
def test_every_efficientnet_label_is_either_mapped_or_explicitly_non_diagnostic() -> None:
    """No label may silently fall through unmapped.

    This is the guard for the brown-spot bug: a class the model is confident
    about but the app has no treatment for is a dead end for the farmer. Each
    label must either reach a KB entry or be listed as deliberately
    non-diagnostic, so adding a class to the checkpoint forces a decision here.
    """
    for label in EFFICIENTNET["labels"]:
        mapped = label in EFFICIENTNET["label_disease"]
        ignored = label in EFFICIENTNET["non_diagnostic"]
        assert mapped != ignored, f"{label} must be exactly one of mapped/non-diagnostic"


def test_efficientnet_has_no_missing_kb_entries() -> None:
    from app.grounding import knowledge_base

    for label, disease_id in EFFICIENTNET["label_disease"].items():
        assert knowledge_base.get_disease(disease_id) is not None, (label, disease_id)


def test_supported_crops_match_the_efficientnet_classes() -> None:
    """The crops the app offers must be exactly the crops the model can name."""
    from app.grounding import knowledge_base

    model_crops = set(EFFICIENTNET["label_crop"].values())
    assert model_crops == set(knowledge_base.SUPPORTED_CROPS) == {"rice", "potato"}


def test_knowledge_base_has_no_entries_outside_supported_crops() -> None:
    from app.grounding import knowledge_base

    for entry in knowledge_base.KB:
        assert entry["cropId"] in knowledge_base.SUPPORTED_CROPS, entry["id"]


def test_logs_top3_probabilities_and_applies_crop_scoped_confidence(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    caplog.set_level(logging.INFO, logger="app.vision.classifier")

    # Split-mass field photo: raw max is 0.354 (< 0.55), but potato classes hold 0.527
    # so potato_early_blight has 0.354 / 0.527 = 0.672 within potato.
    split_probs = _probs(
        EFFICIENTNET,
        potato_early_blight=0.354,
        potato_late_blight=0.123,
        potato_healthy=0.050,
        rice_blast=0.166,
        rice_other=0.139,
        rice_brown_spot=0.168,
    )
    potato_facts = _decide(split_probs, "potato", EFFICIENTNET)
    assert potato_facts["status"] == "identified"
    assert potato_facts["disease_id"] == "potato_early_blight"
    assert potato_facts["confidence"] == pytest.approx(0.354 / 0.527, rel=1e-3)
    assert any("vision top3 (efficientnet, crop=potato)" in r.message for r in caplog.records)
    assert any("potato_early_blight=0.354" in r.message for r in caplog.records)

    # Same split distribution queried as rice has crop_top=rice_brown_spot (0.355 < 0.55) -> miss.
    rice_facts = _decide(split_probs, "rice", EFFICIENTNET)
    assert rice_facts["status"] == "undetermined"

    # Truly uncertain / blurry uniform image (all 9 classes ~0.111, raw max < 0.35) -> miss.
    uniform_probs = np.full(len(EFFICIENTNET["labels"]), 1.0 / len(EFFICIENTNET["labels"]))
    assert _decide(uniform_probs, "potato", EFFICIENTNET)["status"] == "undetermined"

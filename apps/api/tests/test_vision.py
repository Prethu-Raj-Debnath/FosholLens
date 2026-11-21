"""The classifier's decision logic.

These tests exercise `_decide` with injected probabilities instead of running
the model: the weights download on first use, and the interesting decisions are
all about what to do with a probability vector, not about computing one.

The regression this suite exists for is the renormalisation bug: filtering onto
the requested crop's classes *before* applying the confidence threshold lets a
corn leaf clear 0.55 as "rice blast", because the evidence that the photo was
the wrong crop is thrown away by the renormalisation itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.config import settings
from app.vision.classifier import (
    LABEL_DISEASE,
    LABELS,
    NON_DIAGNOSTIC_LABELS,
    _decide,
    _softmax,
)


def _probs(**weights: float) -> np.ndarray:
    """Build a probability vector by label name, zero elsewhere."""
    vector = np.zeros(len(LABELS), dtype=np.float32)
    for label, weight in weights.items():
        name = label.replace("__", "___").replace("_", "__", 0)
        # Accept either the exact label or a python-friendly alias.
        index = _index_of(label) if label in LABELS else _index_of(name)
        vector[index] = weight
    return vector


def _index_of(label: str) -> int:
    return LABELS.index(label)


def test_labels_match_the_exported_graph() -> None:
    assert len(LABELS) == 13
    # Order is part of the model contract, so a silent reorder is a bug.
    assert LABELS[9] == "Rice___Leaf_Blast"
    assert LABELS[3] == "Invalid"


def test_confident_rice_blast_resolves_to_the_kb_entry() -> None:
    facts = _decide(_probs(Rice___Leaf_Blast=0.9, Invalid=0.05), "rice")
    assert facts["status"] == "identified"
    assert facts["disease_id"] == "rice_blast"
    assert facts["confidence"] == pytest.approx(0.9)


def test_below_threshold_is_undetermined() -> None:
    facts = _decide(_probs(Rice___Leaf_Blast=0.4), "rice")
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
        _probs(Corn___Common_Rust=0.5, Corn___Gray_Leaf_Spot=0.48, Rice___Leaf_Blast=0.015),
        "rice",
    )
    assert facts["status"] == "undetermined"
    assert facts["disease_id"] is None


def test_winning_label_from_another_crop_is_undetermined() -> None:
    facts = _decide(_probs(Corn___Common_Rust=0.9), "rice")
    assert facts["status"] == "undetermined"
    assert facts["confidence"] == pytest.approx(0.9)


@pytest.mark.parametrize("label", sorted(NON_DIAGNOSTIC_LABELS))
def test_non_diagnostic_labels_are_never_a_disease(label: str) -> None:
    facts = _decide(_probs(**{label: 0.99}), "rice")
    assert facts["status"] == "undetermined"
    assert facts["disease_id"] is None


def test_healthy_leaf_is_not_dressed_up_as_disease() -> None:
    facts = _decide(_probs(Rice___Healthy=0.95), "rice")
    assert facts["status"] == "undetermined"


def test_known_label_without_a_kb_entry_is_not_invented() -> None:
    """The model can name brown spot; the knowledge base has no entry for it.

    Emitting a treatment anyway would mean writing a dose we have no source for.
    """
    assert "Rice___Brown_Spot" not in LABEL_DISEASE
    facts = _decide(_probs(Rice___Brown_Spot=0.9), "rice")
    assert facts["status"] == "undetermined"


def test_potato_late_blight_resolves() -> None:
    facts = _decide(_probs(Potato___Late_Blight=0.8), "potato")
    assert facts["status"] == "identified"
    assert facts["disease_id"] == "potato_late_blight"


def test_crops_without_model_classes_are_undetermined() -> None:
    """Tomato and chili are out of scope for this model, by inspection.

    Any confident class they produce belongs to some other crop, so the crop
    check catches them regardless of which label won.
    """
    for crop in ("tomato", "chili"):
        facts = _decide(_probs(Potato___Early_Blight=0.95), crop)
        assert facts["status"] == "undetermined", crop


def test_threshold_comes_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "vision_confidence_threshold", 0.95)
    assert _decide(_probs(Rice___Leaf_Blast=0.9), "rice")["status"] == "undetermined"

    monkeypatch.setattr(settings, "vision_confidence_threshold", 0.2)
    assert _decide(_probs(Rice___Leaf_Blast=0.3), "rice")["status"] == "identified"


def test_softmax_is_numerically_stable() -> None:
    """Large logits must not overflow to nan."""
    logits = np.array([1000.0, 1001.0, 999.0], dtype=np.float32)
    probabilities = _softmax(np, logits)
    assert not np.isnan(probabilities).any()
    assert probabilities.sum() == pytest.approx(1.0, abs=1e-6)

"""Vision: leaf disease classification.

Optional. Without a downloaded model, or when the classifier cannot place a
photo, the pipeline falls back to an unconfirmed knowledge-base entry and says
so — it never invents a confidence.
"""

from .classifier import (
    LABEL_CROP,
    LABEL_DISEASE,
    LABELS,
    LeafClassifier,
    VisionFacts,
    get_classifier,
    reset_classifier,
)

__all__ = [
    "LABELS",
    "LABEL_CROP",
    "LABEL_DISEASE",
    "LeafClassifier",
    "VisionFacts",
    "get_classifier",
    "reset_classifier",
]

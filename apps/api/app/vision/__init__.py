"""Vision: leaf disease classification.

Optional. Without a downloaded model, or when the classifier cannot place a
photo, the pipeline falls back to an unconfirmed knowledge-base entry and says
so — it never invents a confidence.

`efficientnet` is the model this app ships and is built around: rice and potato,
9 classes, and a curated entry for every one of them except its own
`rice_other` catch-all.

`vit` is deprecated and opt-in (`VISION_MODEL=vit`). It is kept only for
comparison; its corn and wheat classes have no sourced treatment here.
"""

from .classifier import (
    LeafClassifier,
    VisionFacts,
    get_classifier,
    reset_classifier,
)
from .models import (
    EFFICIENTNET,
    SPECS,
    VIT,
    ModelSpec,
    get_spec,
)

__all__ = [
    "EFFICIENTNET",
    "SPECS",
    "VIT",
    "LeafClassifier",
    "ModelSpec",
    "VisionFacts",
    "get_classifier",
    "get_spec",
    "reset_classifier",
]

"""Vision: leaf disease classification.

Optional. Without a downloaded model, or when the classifier cannot place a
photo, the pipeline falls back to an unconfirmed knowledge-base entry and says
so — it never invents a confidence.

Two checkpoints are available, selected by `VISION_MODEL`:

* ``vit``          - broad but mostly corn/wheat, downloaded on first use.
* ``efficientnet`` - rice/potato only, local file, faster, one more disease.
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

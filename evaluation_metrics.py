"""
evaluation_metrics.py (root-level)
-----------------------------------
Re-exports from tests/evaluation_metrics.py for compatibility with test_chat.py.
The test file imports `from evaluation_metrics import ...` which requires this
module to exist at the project root.
"""

from tests.evaluation_metrics import (
    RetrievalMetrics,
    GroundednessMetrics,
    EffectivenessMetrics,
    RecommendationRelevance,
)

__all__ = [
    "RetrievalMetrics",
    "GroundednessMetrics",
    "EffectivenessMetrics",
    "RecommendationRelevance",
]

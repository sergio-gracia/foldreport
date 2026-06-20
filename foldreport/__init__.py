"""FoldReport — unify structure-prediction outputs into a single HTML report."""

from foldreport.models import (
    Chain,
    Prediction,
    PredictionMetrics,
)

__version__ = "0.1.0"

__all__ = [
    "Chain",
    "Prediction",
    "PredictionMetrics",
    "__version__",
]

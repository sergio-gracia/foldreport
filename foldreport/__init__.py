"""FoldReport — unify structure-prediction outputs into a single HTML report."""

from importlib.metadata import PackageNotFoundError, version

from foldreport.models import (
    Chain,
    Prediction,
    PredictionMetrics,
)

try:
    __version__ = version("foldreport")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0.0.0+unknown"

__all__ = [
    "Chain",
    "Prediction",
    "PredictionMetrics",
    "__version__",
]

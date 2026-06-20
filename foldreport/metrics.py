"""Normalize a list of predictions into a sortable/filterable metrics table.

The table has exactly one row per prediction. Missing metrics stay as ``None`` (which
pandas renders as ``NaN``); nothing here invents values. The default ranking key is a
confidence score that gracefully falls back when a tool omits a metric.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from foldreport.models import Prediction

# Columns in display order. Keep human-readable; the report renders these as headers.
COLUMNS = [
    "name",
    "source_tool",
    "rank",
    "mean_plddt",
    "ptm",
    "iptm",
    "mpdockq",
    "ranking_score",
    "n_chains",
    "n_residues",
]


def metrics_dataframe(predictions: Sequence[Prediction]) -> pd.DataFrame:
    """Build a one-row-per-prediction DataFrame of normalized metrics."""
    rows = []
    for pred in predictions:
        m = pred.metrics
        rows.append(
            {
                "name": pred.name,
                "source_tool": pred.source_tool,
                "rank": pred.rank,
                "mean_plddt": m.mean_plddt,
                "ptm": m.ptm,
                "iptm": m.iptm,
                "mpdockq": m.mpdockq,
                "ranking_score": m.ranking_score,
                "n_chains": m.n_chains,
                "n_residues": m.n_residues,
            }
        )
    df = pd.DataFrame(rows, columns=COLUMNS)
    return df


def confidence_score(pred: Prediction) -> float:
    """A single comparable confidence value used to rank predictions.

    Preference order, using whatever the tool provided:
        1. ipTM (complexes) blended with pTM: 0.8*ipTM + 0.2*pTM
        2. pTM alone
        3. mean pLDDT scaled to 0-1
        4. the tool's own ranking_score
        5. 0.0 as a last resort (keeps it sortable, sorts last)
    """
    m = pred.metrics
    if m.iptm is not None and m.ptm is not None:
        return 0.8 * m.iptm + 0.2 * m.ptm
    if m.iptm is not None:
        return m.iptm
    if m.ptm is not None:
        return m.ptm
    if m.mean_plddt is not None:
        return m.mean_plddt / 100.0
    if m.ranking_score is not None:
        return m.ranking_score
    return 0.0


def rank_predictions(predictions: Sequence[Prediction]) -> list[Prediction]:
    """Return predictions sorted by descending confidence (best first)."""
    return sorted(predictions, key=confidence_score, reverse=True)


def ranked_dataframe(predictions: Sequence[Prediction]) -> pd.DataFrame:
    """Metrics table sorted best-first, with a 1-based ``overall_rank`` column."""
    ordered = rank_predictions(predictions)
    df = metrics_dataframe(ordered)
    df.insert(0, "overall_rank", range(1, len(df) + 1))
    df["confidence"] = [round(confidence_score(p), 4) for p in ordered]
    return df

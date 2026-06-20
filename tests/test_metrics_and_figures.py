"""Phase 2 acceptance: metrics table and figures, tolerant of missing values."""

from __future__ import annotations

import pandas as pd

from foldreport import figures
from foldreport.metrics import confidence_score, metrics_dataframe, ranked_dataframe
from foldreport.models import Prediction, PredictionMetrics
from foldreport.parsers import parse_folder


def test_metrics_dataframe_one_row_per_prediction(colabfold_dir):
    preds = parse_folder(colabfold_dir)
    df = metrics_dataframe(preds)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(preds)
    assert "mean_plddt" in df.columns


def test_ranked_dataframe_is_sorted_best_first(colabfold_dir):
    preds = parse_folder(colabfold_dir)
    df = ranked_dataframe(preds)
    assert list(df["overall_rank"]) == list(range(1, len(df) + 1))
    confidences = list(df["confidence"])
    assert confidences == sorted(confidences, reverse=True)


def test_missing_metrics_become_nan_not_invented():
    pred = Prediction(
        name="bare",
        source_tool="test",
        structure_path=__file__,  # not read here
        plddt=[],
        pae=None,
        metrics=PredictionMetrics(n_chains=1, n_residues=3),
    )
    df = metrics_dataframe([pred])
    # Absent metrics surface as missing (None/NaN), never a fabricated number.
    assert pd.isna(df.loc[0, "mean_plddt"])
    assert pd.isna(df.loc[0, "ptm"])
    # confidence_score falls back gracefully to 0.0 when nothing is available.
    assert confidence_score(pred) == 0.0


def test_figures_return_data_uris(colabfold_dir):
    pred = parse_folder(colabfold_dir)[0]
    figs = figures.make_figures(pred)
    assert figs["plddt"].startswith("data:image/png;base64,")
    assert figs["pae"].startswith("data:image/png;base64,")


def test_figures_handle_absent_data():
    pred = Prediction(
        name="empty",
        source_tool="test",
        structure_path=__file__,
        plddt=[],
        pae=None,
        metrics=PredictionMetrics(),
    )
    figs = figures.make_figures(pred)
    assert figs["plddt"] is None
    assert figs["pae"] is None

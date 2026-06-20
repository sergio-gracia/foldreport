"""Publication-quality static figures: PAE and per-residue pLDDT.

Figures are rendered with a non-interactive matplotlib backend and returned as base64
PNG data URIs so the report can embed them with zero external files. Predictions that
lack a given metric simply produce no figure (the caller renders "N/A").
"""

from __future__ import annotations

import base64
import io

import matplotlib

matplotlib.use("Agg")  # Headless: never requires a display server.

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from foldreport.models import Prediction

# pLDDT confidence bands used by the AlphaFold family (0-100 scale).
_PLDDT_BANDS = [
    (90, 100, "#0053D6", "Very high (90-100)"),
    (70, 90, "#65CBF3", "Confident (70-90)"),
    (50, 70, "#FFDB13", "Low (50-70)"),
    (0, 50, "#FF7D45", "Very low (0-50)"),
]


def _fig_to_data_uri(fig: Figure, dpi: int = 150) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def plddt_figure(pred: Prediction) -> str | None:
    """Per-residue pLDDT line plot with confidence bands. Returns a PNG data URI."""
    if not pred.plddt:
        return None
    plddt = np.asarray(pred.plddt, dtype=float)
    x = np.arange(1, len(plddt) + 1)

    fig, ax = plt.subplots(figsize=(7, 2.6))
    for low, high, color, _label in _PLDDT_BANDS:
        ax.axhspan(low, high, color=color, alpha=0.12, linewidth=0)
    ax.plot(x, plddt, color="#1a1a1a", linewidth=1.2)

    _draw_chain_boundaries(ax, pred)

    ax.set_xlim(1, len(plddt))
    ax.set_ylim(0, 100)
    ax.set_xlabel("Residue")
    ax.set_ylabel("pLDDT")
    ax.set_title("Per-residue pLDDT")
    ax.grid(True, axis="y", alpha=0.2)
    return _fig_to_data_uri(fig)


def pae_figure(pred: Prediction) -> str | None:
    """PAE heatmap (Predicted Aligned Error). Returns a PNG data URI or None."""
    if pred.pae is None:
        return None
    pae = np.asarray(pred.pae, dtype=float)
    if pae.ndim != 2:
        return None

    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(pae, cmap="Greens_r", vmin=0, vmax=max(float(pae.max()), 1.0), origin="upper")
    ax.set_xlabel("Scored residue")
    ax.set_ylabel("Aligned residue")
    ax.set_title("Predicted Aligned Error")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Expected position error (Å)")

    _draw_pae_chain_lines(ax, pred)
    return _fig_to_data_uri(fig)


def _chain_offsets(pred: Prediction) -> list[int]:
    """Cumulative residue offsets at chain boundaries (excluding the final end)."""
    offsets: list[int] = []
    cumulative = 0
    for chain in pred.chains[:-1]:
        cumulative += chain.n_residues
        offsets.append(cumulative)
    return offsets


def _draw_chain_boundaries(ax, pred: Prediction) -> None:
    for boundary in _chain_offsets(pred):
        ax.axvline(boundary + 0.5, color="#888888", linestyle="--", linewidth=0.8)


def _draw_pae_chain_lines(ax, pred: Prediction) -> None:
    for boundary in _chain_offsets(pred):
        ax.axhline(boundary - 0.5, color="#444444", linewidth=0.6)
        ax.axvline(boundary - 0.5, color="#444444", linewidth=0.6)


def pae_data_for_js(pred: Prediction) -> dict | None:
    """Return PAE matrix data for the interactive JS heatmap, or None."""
    if pred.pae is None:
        return None
    pae = np.asarray(pred.pae, dtype=float)
    if pae.ndim != 2:
        return None
    return {
        "matrix": np.round(pae, 2).tolist(),
        "size": int(pae.shape[0]),
        "max_val": round(float(pae.max()), 2),
        "chain_boundaries": _chain_offsets(pred),
    }


def make_figures(pred: Prediction) -> dict[str, str | None]:
    """Return both figures for a prediction as a dict of data URIs (or None)."""
    return {
        "plddt": plddt_figure(pred),
        "pae": pae_figure(pred),
        "pae_interactive": pae_data_for_js(pred),
    }

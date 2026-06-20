"""Publication-quality static figures: PAE and per-residue pLDDT.

Two consumers share the figure-building code here:

* the HTML report, which embeds figures as base64 PNG data URIs (zero external files);
* :func:`save_publication_figures`, which writes submission-ready files to disk
  (vector PDF/SVG with embedded fonts, or PNG at a chosen DPI).

Honest comparison is a design goal: PAE heatmaps are drawn on a **single shared colour
scale** (the AlphaFold 0-31.75 Å convention) so a darker panel always means lower error,
never a different per-figure ``vmax``. Predictions that lack a metric simply produce no
figure (the caller renders "N/A").
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Headless: never requires a display server.

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from foldreport.models import Prediction

# AlphaFold caps PAE at 31.75 Å; using it as the shared ceiling makes every PAE plot
# directly comparable (and comparable across reports), which a per-figure max would not.
PAE_MAX_ANGSTROM = 31.75

# Embed fonts in vector output so figures render identically on any machine: TrueType
# in PDF/PS (fonttype 42), and text drawn as paths in SVG (no font dependency at all).
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "path"

# pLDDT confidence bands used by the AlphaFold family (0-100 scale).
_PLDDT_BANDS = [
    (90, 100, "#0053D6", "Very high (90-100)"),
    (70, 90, "#65CBF3", "Confident (70-90)"),
    (50, 70, "#FFDB13", "Low (50-70)"),
    (0, 50, "#FF7D45", "Very low (0-50)"),
]

# Colorblind-safe alternative (Okabe-Ito) for the categorical pLDDT bands, which are the
# part of the default palette that deuteranopes/protanopes struggle to tell apart.
_PLDDT_BANDS_CB = [
    (90, 100, "#0072B2", "Very high (90-100)"),
    (70, 90, "#56B4E9", "Confident (70-90)"),
    (50, 70, "#E69F00", "Low (50-70)"),
    (0, 50, "#D55E00", "Very low (0-50)"),
]


def plddt_bands(colorblind: bool = False) -> list[tuple[int, int, str, str]]:
    """Return the pLDDT confidence bands (high to low) for the chosen palette."""
    return _PLDDT_BANDS_CB if colorblind else _PLDDT_BANDS


def _fig_to_data_uri(fig: Figure, dpi: int = 150) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


# --- Figure construction (format-agnostic) --------------------------------------------


def build_plddt_figure(pred: Prediction, colorblind: bool = False) -> Figure | None:
    """Per-residue pLDDT line plot with confidence bands, or None if no pLDDT."""
    if not pred.plddt:
        return None
    bands = _PLDDT_BANDS_CB if colorblind else _PLDDT_BANDS
    plddt = np.asarray(pred.plddt, dtype=float)
    x = np.arange(1, len(plddt) + 1)

    fig, ax = plt.subplots(figsize=(7, 2.6))
    for low, high, color, _label in bands:
        ax.axhspan(low, high, color=color, alpha=0.12, linewidth=0)
    ax.plot(x, plddt, color="#1a1a1a", linewidth=1.2)

    _draw_chain_boundaries(ax, pred)

    ax.set_xlim(1, len(plddt))
    ax.set_ylim(0, 100)
    ax.set_xlabel("Residue")
    ax.set_ylabel("pLDDT")
    ax.set_title("Per-residue pLDDT")
    ax.grid(True, axis="y", alpha=0.2)
    return fig


def build_pae_figure(pred: Prediction, vmax: float | None = None) -> Figure | None:
    """PAE heatmap on the shared 0-``vmax`` Å scale, or None if no PAE."""
    if pred.pae is None:
        return None
    pae = np.asarray(pred.pae, dtype=float)
    if pae.ndim != 2:
        return None
    scale_max = PAE_MAX_ANGSTROM if vmax is None else float(vmax)

    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(pae, cmap="Greens_r", vmin=0, vmax=scale_max, origin="upper")
    ax.set_xlabel("Scored residue")
    ax.set_ylabel("Aligned residue")
    ax.set_title("Predicted Aligned Error")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Expected position error (Å)")

    _draw_pae_chain_lines(ax, pred)
    return fig


# --- Data-URI wrappers (for the embedded HTML report) ---------------------------------


def plddt_figure(pred: Prediction, colorblind: bool = False) -> str | None:
    """Per-residue pLDDT plot as a PNG data URI, or None."""
    fig = build_plddt_figure(pred, colorblind=colorblind)
    return _fig_to_data_uri(fig) if fig is not None else None


def pae_figure(pred: Prediction, vmax: float | None = None) -> str | None:
    """PAE heatmap as a PNG data URI on the shared scale, or None."""
    fig = build_pae_figure(pred, vmax=vmax)
    return _fig_to_data_uri(fig) if fig is not None else None


def pae_data_for_js(pred: Prediction, vmax: float | None = None) -> dict | None:
    """Return PAE matrix data for the interactive JS heatmap, or None.

    ``max_val`` is the **shared** colour-scale ceiling (not this matrix's own max) so
    every interactive heatmap in the report colours on one honest scale.
    """
    if pred.pae is None:
        return None
    pae = np.asarray(pred.pae, dtype=float)
    if pae.ndim != 2:
        return None
    scale_max = PAE_MAX_ANGSTROM if vmax is None else float(vmax)
    return {
        "matrix": np.round(pae, 2).tolist(),
        "size": int(pae.shape[0]),
        "max_val": round(scale_max, 2),
        "data_max": round(float(pae.max()), 2),
        "chain_boundaries": _chain_offsets(pred),
    }


# --- Chain-boundary helpers -----------------------------------------------------------


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


def make_figures(
    pred: Prediction, vmax: float | None = None, colorblind: bool = False
) -> dict[str, str | None]:
    """Return both figures for a prediction as a dict of data URIs (or None)."""
    return {
        "plddt": plddt_figure(pred, colorblind=colorblind),
        "pae": pae_figure(pred, vmax=vmax),
        "pae_interactive": pae_data_for_js(pred, vmax=vmax),
    }


# --- Submission-ready export to disk --------------------------------------------------


def save_publication_figures(
    pred: Prediction,
    out_dir: Path,
    *,
    dpi: int = 300,
    formats: tuple[str, ...] = ("pdf", "png"),
    vmax: float | None = None,
    colorblind: bool = False,
) -> list[Path]:
    """Write submission-ready pLDDT/PAE figures for ``pred`` into ``out_dir``.

    Vector formats (pdf, svg) embed fonts so the figure is reproducible anywhere; raster
    formats (png) honour ``dpi``. PAE uses the shared colour scale, so figures exported
    for different predictions stay visually comparable. Returns the paths written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_stem(pred.name)

    written: list[Path] = []
    builders = (
        ("plddt", build_plddt_figure(pred, colorblind=colorblind)),
        ("pae", build_pae_figure(pred, vmax=vmax)),
    )
    for label, fig in builders:
        if fig is None:
            continue
        for fmt in formats:
            path = out_dir / f"{stem}_{label}.{fmt}"
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            written.append(path)
        plt.close(fig)
    return written


def _safe_stem(name: str) -> str:
    """A filesystem-safe stem derived from a prediction name."""
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(name))

"""Assemble the single self-contained HTML report.

The output is one ``.html`` file with everything inlined: CSS, the 3Dmol.js library,
structure coordinates, and base64 PNG figures. No internet connection or adjacent
files are needed to open it.
"""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from typing import Sequence

import numpy as np

from foldreport import __version__, figures
from foldreport.metrics import confidence_score, rank_predictions
from foldreport.models import Prediction
from foldreport.parsers import base

# Columns shown in the ranking table: (key, header, numeric?, formatter).
_TABLE_COLUMNS = [
    ("overall_rank", "#", True, lambda v: f"{int(v)}"),
    ("name", "Prediction", False, str),
    ("source_tool", "Tool", False, str),
    ("confidence", "Confidence", True, lambda v: f"{v:.3f}"),
    ("mean_plddt", "Mean pLDDT", True, lambda v: f"{v:.1f}"),
    ("ptm", "pTM", True, lambda v: f"{v:.3f}"),
    ("iptm", "ipTM", True, lambda v: f"{v:.3f}"),
    ("mpdockq", "mpDockQ", True, lambda v: f"{v:.3f}"),
    ("n_chains", "Chains", True, lambda v: f"{int(v)}"),
    ("n_residues", "Residues", True, lambda v: f"{int(v)}"),
]


def build_report(
    predictions: Sequence[Prediction],
    output_path: Path,
    title: str = "FoldReport",
    colorblind: bool = False,
) -> Path:
    """Render ``predictions`` into a single self-contained HTML file at ``output_path``.

    When ``colorblind`` is set, the per-residue pLDDT bands use a colorblind-safe
    palette. PAE already uses a single-hue sequential scale, which is colorblind-safe.
    """
    output_path = Path(output_path)
    ordered = rank_predictions(predictions)

    template = _load_template()
    tdmol_js = _load_3dmol()

    rows = [_row_dict(p, i + 1) for i, p in enumerate(ordered)]
    ids = [f"viewer-{i}" for i in range(len(ordered))]

    models_json = {}
    pae_json = {}
    provenance_json = {}
    detail_cards = []
    for idx, (pred, row) in enumerate(zip(ordered, rows)):
        viewer_id = ids[idx]
        data, fmt, ca_only = _structure_for_viewer(pred)
        models_json[viewer_id] = {"data": data, "format": fmt, "ca_only": ca_only}
        pae_data = figures.pae_data_for_js(pred)
        if pae_data is not None:
            pae_json[viewer_id] = pae_data
        provenance_json[viewer_id] = _provenance_dict(pred)
        detail_cards.append(_detail_card(pred, row, viewer_id, colorblind))

    html_out = (
        template.replace("__TITLE__", html.escape(title))
        .replace("__VERSION__", html.escape(__version__))
        .replace("__GENERATED__", _now())
        .replace("__SUMMARY__", _summary(ordered))
        .replace("__RANKING_TABLE__", _ranking_table(rows, colorblind))
        .replace("__DETAIL_CARDS__", "\n".join(detail_cards))
        .replace("__MODELS_JSON__", json.dumps(models_json))
        .replace("__PAE_JSON__", json.dumps(pae_json))
        .replace("__PROVENANCE_JSON__", json.dumps(provenance_json))
        .replace("__PLDDT_COLORS_JS__", _plddt_colors_js(colorblind))
        # 3Dmol.js goes in last; it must not be touched by earlier replacements.
        .replace("__TDMOL_JS__", tdmol_js)
    )

    output_path.write_text(html_out, encoding="utf-8")
    return output_path


# --- Template / asset loading ---------------------------------------------------------


def _load_template() -> str:
    return resources.files("foldreport.report").joinpath("template.html").read_text(
        encoding="utf-8"
    )


def _load_3dmol() -> str:
    return resources.files("foldreport.report").joinpath("3Dmol-min.js").read_text(
        encoding="utf-8"
    )


# --- Row / summary helpers ------------------------------------------------------------


def _row_dict(pred: Prediction, overall_rank: int) -> dict:
    m = pred.metrics
    return {
        "overall_rank": overall_rank,
        "name": pred.name,
        "source_tool": pred.source_tool,
        "confidence": round(confidence_score(pred), 4),
        "mean_plddt": m.mean_plddt,
        "ptm": m.ptm,
        "iptm": m.iptm,
        "mpdockq": m.mpdockq,
        "n_chains": m.n_chains,
        "n_residues": m.n_residues,
    }


def _summary(predictions: Sequence[Prediction]) -> str:
    tools = sorted({p.source_tool for p in predictions})
    best = predictions[0] if predictions else None
    stats = [
        ("Predictions", str(len(predictions))),
        ("Tools", ", ".join(tools) if tools else "&mdash;"),
        ("Top prediction", html.escape(best.name) if best else "&mdash;"),
        (
            "Best confidence",
            f"{confidence_score(best):.3f}" if best else "&mdash;",
        ),
    ]
    return "".join(
        f'<div class="stat">{label}<b>{value}</b></div>' for label, value in stats
    )


def _ranking_table(rows: list[dict], colorblind: bool = False) -> str:
    # Tooltips spell out each metric so the table is self-explanatory to a reader who
    # opens the report cold.
    headers_help = {
        "#": "Overall rank by combined confidence (best first)",
        "Prediction": "Prediction name; click to jump to its detail card",
        "Tool": "Tool that produced the prediction",
        "Confidence": "Combined confidence used for ranking (0-1, higher is better)",
        "Mean pLDDT": "Mean per-residue confidence (0-100, higher is better)",
        "pTM": "Predicted TM-score for the whole structure (0-1)",
        "ipTM": "Interface predicted TM-score for complexes (0-1)",
        "mpDockQ": "Multi-chain DockQ for complex interfaces (0-1)",
        "Chains": "Number of chains",
        "Residues": "Total number of residues",
    }
    head_cells = []
    for _key, header, numeric, _fmt in _TABLE_COLUMNS:
        cls = "" if numeric else ' class="txt"'
        tip = headers_help.get(header, "")
        title = f' title="{html.escape(tip)}"' if tip else ""
        head_cells.append(f"<th{cls}{title}>{html.escape(header)}</th>")
    thead = "<thead><tr>" + "".join(head_cells) + "</tr></thead>"

    body_rows = []
    for row in rows:
        rank = row.get("overall_rank")
        row_cls = f' class="top{rank}"' if isinstance(rank, int) and rank <= 3 else ""
        cells = []
        for key, _header, numeric, fmt in _TABLE_COLUMNS:
            value = row.get(key)
            cells.append(_table_cell(key, value, numeric, fmt, colorblind))
        body_rows.append(f"<tr{row_cls}>" + "".join(cells) + "</tr>")
    tbody = "<tbody>" + "".join(body_rows) + "</tbody>"
    return thead + tbody


def _table_cell(key: str, value, numeric: bool, fmt, colorblind: bool = False) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        # Sort missing numbers to the bottom regardless of direction intent.
        return '<td class="na" data-v="">N/A</td>'
    if key == "name":
        anchor = _anchor_id(value)
        text = html.escape(str(value))
        return f'<td class="txt" data-v="{text}"><a class="jump" href="#{anchor}">{text}</a></td>'
    cls = "" if numeric else ' class="txt"'
    data_v = value if numeric else html.escape(str(value))
    body = html.escape(fmt(value))
    # Color-coded dots let a reader triage hundreds of rows at a glance: the two primary
    # confidence metrics get a band swatch (matching the figures/legend) next to the value.
    dot = _metric_dot(key, value, colorblind)
    return f'<td{cls} data-v="{data_v}">{dot}{body}</td>'


# --- Confidence color-coding (triage at a glance) -------------------------------------

# Bands for the combined confidence score (0-1). Green/amber/red reads as good/ok/poor
# regardless of the pLDDT palette in use.
_CONFIDENCE_COLORS = [(0.8, "#15803d"), (0.6, "#b45309"), (0.0, "#b91c1c")]


def _confidence_color(value: float) -> str:
    for threshold, color in _CONFIDENCE_COLORS:
        if value >= threshold:
            return color
    return _CONFIDENCE_COLORS[-1][1]


def _plddt_dot_color(value: float, colorblind: bool) -> str:
    """Swatch color for a mean-pLDDT value, matching the figure/legend bands."""
    for low, _high, color, _label in figures.plddt_bands(colorblind):
        if value >= low:
            return color
    return figures.plddt_bands(colorblind)[-1][2]


def _metric_dot(key: str, value, colorblind: bool) -> str:
    """A small color swatch for the confidence/pLDDT columns, else nothing."""
    if key == "confidence":
        color = _confidence_color(float(value))
    elif key == "mean_plddt":
        color = _plddt_dot_color(float(value), colorblind)
    else:
        return ""
    return f'<span class="mdot" style="background:{color}"></span>'


# --- Detail cards ---------------------------------------------------------------------


def _detail_card(pred: Prediction, row: dict, viewer_id: str, colorblind: bool = False) -> str:
    anchor = _anchor_id(pred.name)
    figs = figures.make_figures(pred, colorblind=colorblind)
    downloads = figures.publication_downloads(pred, colorblind=colorblind)
    chips = _chips(pred, row)
    stem = figures._safe_stem(pred.name)

    plddt_html = (
        f'<img src="{figs["plddt"]}" alt="Per-residue pLDDT">'
        if figs["plddt"]
        else '<div class="na-fig">pLDDT not available</div>'
    )
    plddt_html += _download_links(downloads.get("plddt"), stem, "plddt")

    pae_data = figures.pae_data_for_js(pred)
    if pae_data is not None:
        # Interactive PAE canvas. The canvas has no baked-in title or scale (unlike the
        # static figure), so we add an HTML label and a color-scale legend here.
        pae_html = (
            '<div class="panel-label">Predicted Aligned Error '
            '<span class="panel-hint">hover for residue-pair error</span></div>'
            f'<div class="pae-container"><canvas class="pae-canvas" id="pae-{viewer_id}" width="400" height="400"></canvas>'
            f'<div class="pae-tooltip" id="pae-tip-{viewer_id}"></div></div>'
            + _pae_scale_legend(pae_data.get("max_val"))
        )
    elif figs["pae"]:
        # Fallback static image
        pae_html = f'<img src="{figs["pae"]}" alt="Predicted Aligned Error">'
    else:
        pae_html = '<div class="na-fig">PAE not available</div>'
    pae_html += _download_links(downloads.get("pae"), stem, "pae")

    return f"""
<div class="card" id="{anchor}">
  <div class="card-head">
    <h3>{html.escape(pred.name)}</h3>
    <a class="to-top" href="#ranking" title="Back to ranking">&uarr; ranking</a>
  </div>
  <div class="meta">{html.escape(pred.source_tool)} &middot; {pred.metrics.n_chains} chains &middot; {pred.metrics.n_residues} residues</div>
  <div class="chips">{chips}</div>
  <div class="grid">
    <div>
      <div class="panel-label">3D structure <span class="panel-hint">colored by pLDDT</span></div>
      <div class="viewer" id="{viewer_id}"></div>
      <div class="legend">{_plddt_legend(colorblind)}</div>
    </div>
    <div class="figs">
      {plddt_html}
      {pae_html}
    </div>
  </div>
  {_provenance_html(pred)}
</div>
"""


def _pae_scale_legend(max_val: float | None) -> str:
    """A horizontal color-scale legend for the interactive PAE heatmap.

    Mirrors the JS ``paeColor`` gradient (dark green = low error, white = high) so the
    interactive canvas is as interpretable as the static figure's colorbar.
    """
    ceiling = f"{float(max_val):.1f}" if max_val is not None else "max"
    return (
        '<div class="pae-scale">'
        '<span class="pae-scale-cap">Expected position error</span>'
        '<div class="pae-scale-row">'
        '<span class="pae-scale-end">0 &#8491;</span>'
        '<span class="pae-scale-bar"></span>'
        f'<span class="pae-scale-end">{ceiling} &#8491;</span>'
        "</div>"
        "</div>"
    )


# Human-readable labels for the embedded download formats.
_DOWNLOAD_LABELS = {"png": "PNG (300 dpi)", "pdf": "PDF (vector)", "svg": "SVG (vector)"}


def _download_links(uris: dict | None, stem: str, label: str) -> str:
    """Submission-ready download links for one figure, or empty if it has none.

    Each link is a self-contained data URI, so the publication-quality figure travels
    inside the single HTML file with no adjacent assets.
    """
    if not uris:
        return ""
    links = "".join(
        f'<a class="dl" download="{html.escape(stem)}_{label}.{fmt}" '
        f'href="{uri}">{_DOWNLOAD_LABELS.get(fmt, fmt.upper())}</a>'
        for fmt, uri in uris.items()
    )
    return f'<div class="dl-row"><span class="dl-label">Download</span>{links}</div>'


# --- Provenance -----------------------------------------------------------------------

# (label, attribute, formatter) for the human-readable provenance rows.
_PROVENANCE_FIELDS = [
    ("Model", "model_name", str),
    ("Tool version", "tool_version", str),
    ("Seeds", "seeds", lambda v: ", ".join(str(s) for s in v)),
    ("MSA mode", "msa_mode", str),
    ("MSA depth", "msa_depth", lambda v: f"{int(v):,} sequences"),
    ("Recycles", "num_recycles", lambda v: str(int(v))),
    ("Templates", "use_templates", lambda v: "Yes" if v else "No"),
    ("Database snapshot", "database_snapshot", str),
    ("Created", "created_date", str),
]


def _provenance_rows(pred: Prediction) -> list[tuple[str, str]]:
    """(label, value) pairs for everything actually captured, in display order."""
    prov = pred.provenance
    rows: list[tuple[str, str]] = []
    for label, attr, fmt in _PROVENANCE_FIELDS:
        value = getattr(prov, attr)
        if value is None or (isinstance(value, list) and not value):
            continue
        rows.append((label, fmt(value)))
    for key, value in prov.extra.items():
        rows.append((key.replace("_", " ").capitalize(), str(value)))
    return rows


def _provenance_html(pred: Prediction) -> str:
    """A collapsible provenance panel: captured metadata plus the source files.

    Reproducibility is the point — the reviewer (or the future you) should see exactly
    what produced this figure without leaving the HTML. Fields the tool did not record
    are omitted rather than invented.
    """
    rows = _provenance_rows(pred)
    items = [
        f'<div class="prov-item"><span>{html.escape(label)}</span><b>{html.escape(value)}</b></div>'
        for label, value in rows
    ]
    if not items:
        items.append('<div class="prov-item prov-na">No provenance recorded by the tool</div>')

    files = [
        f'<div class="prov-item"><span>{html.escape(role)}</span><b>{html.escape(path.name)}</b></div>'
        for role, path in pred.raw_files.items()
    ]
    files_block = (
        '<div class="prov-sub">Source files</div>' + "".join(files) if files else ""
    )

    return (
        '<details class="provenance">'
        "<summary>Provenance &amp; reproducibility</summary>"
        f'<div class="prov-grid">{"".join(items)}{files_block}</div>'
        "</details>"
    )


def _provenance_dict(pred: Prediction) -> dict:
    """Machine-readable provenance embedded in the HTML for downstream extraction."""
    prov = pred.provenance
    return {
        "name": pred.name,
        "source_tool": pred.source_tool,
        "model_name": prov.model_name,
        "tool_version": prov.tool_version,
        "seeds": prov.seeds,
        "msa_mode": prov.msa_mode,
        "msa_depth": prov.msa_depth,
        "num_recycles": prov.num_recycles,
        "use_templates": prov.use_templates,
        "database_snapshot": prov.database_snapshot,
        "created_date": prov.created_date,
        "extra": prov.extra,
        "source_files": {role: path.name for role, path in pred.raw_files.items()},
    }


_PLDDT_LEGEND_LABELS = ["Very high", "Confident", "Low", "Very low"]


def _plddt_legend(colorblind: bool) -> str:
    """Legend swatches matching the pLDDT palette actually used in the figures/viewer."""
    bands = figures.plddt_bands(colorblind)
    return "".join(
        f'<span><i style="background:{color}"></i>{label}</span>'
        for (_low, _high, color, _full), label in zip(bands, _PLDDT_LEGEND_LABELS)
    )


def _plddt_colors_js(colorblind: bool) -> str:
    """JSON array of band colors (high to low) for the 3D viewer's color function."""
    return json.dumps([color for (_low, _high, color, _full) in figures.plddt_bands(colorblind)])


def _chips(pred: Prediction, row: dict) -> str:
    conf = row["confidence"]
    conf_dot = f'<span class="mdot" style="background:{_confidence_color(float(conf))}"></span>'
    items = [
        (f"{conf_dot}Confidence", f"{conf:.3f}"),
        ("Mean pLDDT", _fmt_opt(pred.metrics.mean_plddt, ".1f")),
        ("pTM", _fmt_opt(pred.metrics.ptm, ".3f")),
        ("ipTM", _fmt_opt(pred.metrics.iptm, ".3f")),
        ("mpDockQ", _fmt_opt(pred.metrics.mpdockq, ".3f")),
    ]
    if pred.rank is not None:
        items.append(("Tool rank", str(pred.rank)))
    return "".join(
        f'<span class="chip">{label} <b>{value}</b></span>' for label, value in items
    )


def _fmt_opt(value, spec: str) -> str:
    return "N/A" if value is None else format(value, spec)


# --- Structure embedding --------------------------------------------------------------


def _structure_for_viewer(pred: Prediction) -> tuple[str, str, bool]:
    """Return (coordinate_text, format, ca_only) with B-factors set to pLDDT (0-100).

    Rewriting the B-factor column guarantees the viewer colors by a consistent 0-100
    pLDDT scale across tools (e.g. Boltz natively stores 0-1).
    """
    structure = base.read_structure(pred.structure_path)
    model = structure[0] if len(structure) else None

    n_atoms = 0
    n_ca = 0
    if model is not None:
        plddt = pred.plddt
        ri = 0
        for chain in model:
            for residue in chain:
                value = plddt[ri] if ri < len(plddt) else None
                ri += 1
                for atom in residue:
                    n_atoms += 1
                    if atom.name == "CA":
                        n_ca += 1
                    if value is not None:
                        atom.b_iso = float(value)

    ca_only = n_atoms > 0 and n_atoms == n_ca
    try:
        text = structure.make_pdb_string()
        fmt = "pdb"
    except Exception:
        text = structure.make_mmcif_document().as_string()
        fmt = "cif"
    return text, fmt, ca_only


# --- Misc -----------------------------------------------------------------------------


def _anchor_id(name: str) -> str:
    return "pred-" + "".join(c if c.isalnum() else "-" for c in str(name))


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

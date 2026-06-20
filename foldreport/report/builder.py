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
) -> Path:
    """Render ``predictions`` into a single self-contained HTML file at ``output_path``."""
    output_path = Path(output_path)
    ordered = rank_predictions(predictions)

    template = _load_template()
    tdmol_js = _load_3dmol()

    rows = [_row_dict(p, i + 1) for i, p in enumerate(ordered)]
    ids = [f"viewer-{i}" for i in range(len(ordered))]

    models_json = {}
    pae_json = {}
    detail_cards = []
    for idx, (pred, row) in enumerate(zip(ordered, rows)):
        viewer_id = ids[idx]
        data, fmt, ca_only = _structure_for_viewer(pred)
        models_json[viewer_id] = {"data": data, "format": fmt, "ca_only": ca_only}
        pae_data = figures.pae_data_for_js(pred)
        if pae_data is not None:
            pae_json[viewer_id] = pae_data
        detail_cards.append(_detail_card(pred, row, viewer_id))

    html_out = (
        template.replace("__TITLE__", html.escape(title))
        .replace("__VERSION__", html.escape(__version__))
        .replace("__GENERATED__", _now())
        .replace("__SUMMARY__", _summary(ordered))
        .replace("__RANKING_TABLE__", _ranking_table(rows))
        .replace("__DETAIL_CARDS__", "\n".join(detail_cards))
        .replace("__MODELS_JSON__", json.dumps(models_json))
        .replace("__PAE_JSON__", json.dumps(pae_json))
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


def _ranking_table(rows: list[dict]) -> str:
    head_cells = []
    for _key, header, numeric, _fmt in _TABLE_COLUMNS:
        cls = "" if numeric else ' class="txt"'
        head_cells.append(f"<th{cls}>{html.escape(header)}</th>")
    thead = "<thead><tr>" + "".join(head_cells) + "</tr></thead>"

    body_rows = []
    for row in rows:
        cells = []
        for key, _header, numeric, fmt in _TABLE_COLUMNS:
            value = row.get(key)
            cells.append(_table_cell(key, value, numeric, fmt))
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    tbody = "<tbody>" + "".join(body_rows) + "</tbody>"
    return thead + tbody


def _table_cell(key: str, value, numeric: bool, fmt) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        # Sort missing numbers to the bottom regardless of direction intent.
        return '<td class="na" data-v="">N/A</td>'
    if key == "name":
        anchor = _anchor_id(value)
        text = html.escape(str(value))
        return f'<td class="txt" data-v="{text}"><a class="jump" href="#{anchor}">{text}</a></td>'
    cls = "" if numeric else ' class="txt"'
    data_v = value if numeric else html.escape(str(value))
    return f'<td{cls} data-v="{data_v}">{html.escape(fmt(value))}</td>'


# --- Detail cards ---------------------------------------------------------------------


def _detail_card(pred: Prediction, row: dict, viewer_id: str) -> str:
    anchor = _anchor_id(pred.name)
    figs = figures.make_figures(pred)
    chips = _chips(pred, row)

    plddt_html = (
        f'<img src="{figs["plddt"]}" alt="Per-residue pLDDT">'
        if figs["plddt"]
        else '<div class="na-fig">pLDDT not available</div>'
    )
    pae_data = figures.pae_data_for_js(pred)
    if pae_data is not None:
        # Interactive PAE canvas
        pae_html = f'<div class="pae-container"><canvas class="pae-canvas" id="pae-{viewer_id}" width="400" height="400"></canvas><div class="pae-tooltip" id="pae-tip-{viewer_id}"></div></div>'
    elif figs["pae"]:
        # Fallback static image
        pae_html = f'<img src="{figs["pae"]}" alt="Predicted Aligned Error">'
    else:
        pae_html = '<div class="na-fig">PAE not available</div>'

    return f"""
<div class="card" id="{anchor}">
  <h3>{html.escape(pred.name)}</h3>
  <div class="meta">{html.escape(pred.source_tool)} &middot; {pred.metrics.n_chains} chains &middot; {pred.metrics.n_residues} residues</div>
  <div class="chips">{chips}</div>
  <div class="grid">
    <div>
      <div class="viewer" id="{viewer_id}"></div>
      <div class="legend">
        <span><i style="background:#0053D6"></i>Very high</span>
        <span><i style="background:#65CBF3"></i>Confident</span>
        <span><i style="background:#FFDB13"></i>Low</span>
        <span><i style="background:#FF7D45"></i>Very low</span>
      </div>
    </div>
    <div class="figs">
      {plddt_html}
      {pae_html}
    </div>
  </div>
</div>
"""


def _chips(pred: Prediction, row: dict) -> str:
    items = [
        ("Confidence", f"{row['confidence']:.3f}"),
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

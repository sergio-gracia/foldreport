"""Parser for AlphaFold Protein Structure Database downloads.

The AlphaFold DB (https://alphafold.ebi.ac.uk) distributes one predicted model per
UniProt accession as a small set of files::

    AF-<ACC>-F<frag>-model_v<ver>.cif                    # structure, pLDDT in B-factors
    AF-<ACC>-F<frag>-predicted_aligned_error_v<ver>.json # PAE matrix (DB JSON format)
    AF-<ACC>-F<frag>-metadata.json                       # optional: our saved API record

Unlike ColabFold/AF3/Boltz, the DB ships a single model per entry (no ranks), pLDDT
lives only in the structure's B-factor column, and the PAE JSON uses AlphaFold-DB's own
shape: ``[{"predicted_aligned_error": [[...]], "max_predicted_aligned_error": N}]``.
pTM / ipTM are not published, so they stay ``None`` (rendered as N/A).

The optional ``metadata.json`` (the JSON returned by the DB's ``/api/prediction/<ACC>``
endpoint, saved alongside by :mod:`examples.fetch_afdb`) is used, when present, for a
human-readable name (gene / organism) and the published mean pLDDT.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from foldreport.models import Prediction, PredictionMetrics
from foldreport.parsers import base

_MODEL_RE = re.compile(
    r"^AF-(?P<acc>[A-Za-z0-9]+)-F(?P<frag>\d+)-model_v(?P<ver>\d+)\.(?:cif|pdb)$"
)
_PAE_RE = re.compile(
    r"^AF-(?P<acc>[A-Za-z0-9]+)-F(?P<frag>\d+)-predicted_aligned_error_v(?P<ver>\d+)\.json$"
)
_META_RE = re.compile(r"^AF-(?P<acc>[A-Za-z0-9]+)-F(?P<frag>\d+)-metadata\.json$")


class AlphaFoldDBParser:
    """Parse a folder of AlphaFold Protein Structure Database downloads."""

    name = "alphafold_db"

    def can_handle(self, path: Path) -> bool:
        path = Path(path)
        if not path.is_dir():
            return False
        return any(_MODEL_RE.match(entry.name) for entry in path.iterdir() if entry.is_file())

    def parse(self, path: Path) -> list[Prediction]:
        path = Path(path)
        models = _index(path, _MODEL_RE)
        paes = _index(path, _PAE_RE)
        metas = _index(path, _META_RE)

        predictions: list[Prediction] = []
        for acc, struct_path in sorted(models.items()):
            plddt = base.plddt_from_bfactors(struct_path)
            pae = _load_pae(paes.get(acc))

            structure = base.read_structure(struct_path)
            chains = base.chains_from_structure(structure)
            n_residues = sum(c.n_residues for c in chains)

            meta = _load_json(metas.get(acc))
            mean_plddt = _as_opt_float(meta.get("globalMetricValue"))
            if mean_plddt is None and plddt:
                mean_plddt = float(np.mean(plddt))

            metrics = PredictionMetrics(
                mean_plddt=mean_plddt,
                ptm=None,  # not published by AlphaFold DB
                iptm=None,
                mpdockq=None,
                n_chains=len(chains),
                n_residues=n_residues,
            )

            raw_files: dict[str, Path] = {"structure": struct_path}
            if paes.get(acc):
                raw_files["pae"] = paes[acc]
            if metas.get(acc):
                raw_files["metadata"] = metas[acc]

            predictions.append(
                Prediction(
                    name=_display_name(acc, meta),
                    source_tool=self.name,
                    structure_path=struct_path,
                    chains=chains,
                    plddt=plddt,
                    pae=pae,
                    metrics=metrics,
                    rank=1,  # one published model per entry
                    raw_files=raw_files,
                )
            )
        return predictions


def _display_name(acc: str, meta: dict) -> str:
    gene = meta.get("gene") or meta.get("uniprotId")
    return f"{gene} ({acc})" if gene else acc


def _index(path: Path, pattern: re.Pattern) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for entry in path.iterdir():
        if not entry.is_file():
            continue
        m = pattern.match(entry.name)
        if m:
            out[m["acc"]] = entry
    return out


def _load_json(path: Path | None) -> dict:
    if path is None:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return {}


def _load_pae(path: Path | None) -> np.ndarray | None:
    """Read AlphaFold-DB's PAE JSON list into an N x N float matrix, or None."""
    if path is None:
        return None
    data = _load_json(path)
    record = data[0] if isinstance(data, list) and data else data
    if not isinstance(record, dict):
        return None
    matrix = record.get("predicted_aligned_error")
    if matrix is None:
        return None
    arr = np.asarray(matrix, dtype=float)
    return arr if arr.ndim == 2 else None


def _as_opt_float(value) -> float | None:
    return None if value is None else float(value)

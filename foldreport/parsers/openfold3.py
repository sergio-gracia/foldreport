"""Parser for OpenFold3 output folders.

OpenFold3 (aqlaboratory/openfold-3, released March 2026) writes one directory per
job, with one subdirectory per (seed, sample) draw::

    output_dir/
    └── <job>/
        ├── seed-<seed>_sample-<sample>/
        │   ├── <job>_model.cif                  -> structure (pLDDT in B-factors)
        │   ├── <job>_confidences.json           -> atom_plddts, pae, token_chain_ids, pde
        │   └── <job>_summary_confidences.json   -> ptm, iptm, ranking_score, plddt, ...
        └── experiment_config.json

OpenFold3 reports pLDDT per *atom* (``atom_plddts``), unlike the other tools which
expose it per residue. We collapse it to per-residue with
:func:`~foldreport.parsers.base.aggregate_atom_plddt_to_residue`, falling back to the
structure B-factors when the per-atom arrays are absent. The PAE matrix is per token.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from foldreport.models import Prediction, PredictionMetrics
from foldreport.parsers import base

_SAMPLE_DIR_RE = re.compile(r"^seed-(?P<seed>\d+)_sample-(?P<sample>\d+)$")
_SUMMARY_RE = re.compile(r"^(?P<job>.+)_summary_confidences\.json$")
_CONF_RE = re.compile(r"^(?P<job>.+)_confidences\.json$")
_MODEL_RE = re.compile(r"^(?P<job>.+)_model\.(?:cif|pdb)$")


class OpenFold3Parser:
    """Parse a folder produced by OpenFold3."""

    name = "openfold3"

    def can_handle(self, path: Path) -> bool:
        return any(True for _ in self._iter_sample_dirs(Path(path)))

    def parse(self, path: Path) -> list[Prediction]:
        path = Path(path)
        predictions: list[Prediction] = []
        for sample_dir in self._iter_sample_dirs(path):
            pred = self._parse_sample(sample_dir)
            if pred is not None:
                predictions.append(pred)
        # Stable ordering: best ranking_score first, then by name.
        predictions.sort(
            key=lambda p: (
                -(p.metrics.ranking_score if p.metrics.ranking_score is not None else -1.0),
                p.name,
            )
        )
        for rank, pred in enumerate(predictions, start=1):
            pred.rank = rank
        return predictions

    def _parse_sample(self, sample_dir: Path) -> Prediction | None:
        model_path = _find(sample_dir, _MODEL_RE)
        if model_path is None:
            return None
        summary = _load_json(_find(sample_dir, _SUMMARY_RE))
        conf = _load_json(_find(sample_dir, _CONF_RE))

        plddt = _per_residue_plddt(conf)
        if not plddt:
            plddt = base.plddt_from_bfactors(model_path)

        pae = _as_matrix(conf.get("pae"))

        structure = base.read_structure(model_path)
        chains = base.chains_from_structure(structure)
        n_residues = sum(c.n_residues for c in chains)

        mean_plddt = _as_opt_float(summary.get("plddt"))
        if mean_plddt is None and plddt:
            mean_plddt = float(np.mean(plddt))

        metrics = PredictionMetrics(
            mean_plddt=mean_plddt,
            ptm=_as_opt_float(summary.get("ptm")),
            iptm=_as_opt_float(summary.get("iptm")),
            mpdockq=None,  # OpenFold3 does not report mpDockQ.
            n_chains=len(chains),
            n_residues=n_residues,
            ranking_score=_as_opt_float(summary.get("ranking_score")),
        )

        raw_files: dict[str, Path] = {"structure": model_path}
        summary_path = _find(sample_dir, _SUMMARY_RE)
        conf_path = _find(sample_dir, _CONF_RE)
        if summary_path is not None:
            raw_files["summary"] = summary_path
        if conf_path is not None:
            raw_files["confidences"] = conf_path

        job = _MODEL_RE.match(model_path.name)["job"]
        return Prediction(
            name=f"{job}_{sample_dir.name}",
            source_tool=self.name,
            structure_path=model_path,
            chains=chains,
            plddt=plddt,
            pae=pae,
            metrics=metrics,
            rank=None,  # assigned after global ranking in parse()
            raw_files=raw_files,
        )

    @staticmethod
    def _iter_sample_dirs(path: Path):
        """Yield seed-*_sample-* directories holding the OpenFold3 confidence files."""
        if not path.is_dir():
            return
        seen: set[Path] = set()
        for entry in path.rglob("*"):
            if not entry.is_dir() or not _SAMPLE_DIR_RE.match(entry.name):
                continue
            if entry in seen:
                continue
            has_summary = any(_SUMMARY_RE.match(f.name) for f in entry.iterdir())
            has_conf = any(_CONF_RE.match(f.name) for f in entry.iterdir())
            if has_summary and has_conf:
                seen.add(entry)
                yield entry


def _per_residue_plddt(conf: dict) -> list[float]:
    """Collapse OpenFold3 per-atom pLDDT to per-residue, or return [] if unavailable."""
    atom_plddts = conf.get("atom_plddts")
    res_ids = conf.get("token_res_ids") or conf.get("atom_res_ids")
    chain_ids = conf.get("token_chain_ids") or conf.get("atom_chain_ids")
    if not atom_plddts or not res_ids or not chain_ids:
        return []
    if not (len(atom_plddts) == len(res_ids) == len(chain_ids)):
        return []
    return base.aggregate_atom_plddt_to_residue(
        [float(v) for v in atom_plddts], list(res_ids), list(chain_ids)
    )


def _find(directory: Path, pattern: re.Pattern) -> Path | None:
    for entry in directory.iterdir():
        if entry.is_file() and pattern.match(entry.name):
            return entry
    return None


def _load_json(path: Path | None) -> dict:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _as_opt_float(value) -> float | None:
    return None if value is None else float(value)


def _as_matrix(value) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    return arr if arr.ndim == 2 else None

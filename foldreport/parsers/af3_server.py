"""Parser for AlphaFold 3 Server output folders.

The AF3 Server download contains, per job, five ranked models plus JSON sidecars::

    fold_<job>_model_0.cif ... fold_<job>_model_4.cif
    fold_<job>_full_data_0.json          -> atom_plddts, pae, token_chain_ids, ...
    fold_<job>_summary_confidences_0.json -> ptm, iptm, ranking_score, ...
    fold_<job>_job_request.json
    terms_of_use.md

pLDDT is stored both per-atom in ``full_data`` and in the mmCIF B-factor column; we
read per-residue pLDDT from the B-factors for consistency with the other parsers.
The PAE matrix (per token) comes from ``full_data``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from foldreport.models import Prediction, PredictionMetrics, Provenance
from foldreport.parsers import base

_MODEL_RE = re.compile(r"^(?P<job>.+)_model_(?P<idx>\d+)\.(?:cif|pdb)$")
_SUMMARY_RE = re.compile(r"^(?P<job>.+)_summary_confidences_(?P<idx>\d+)\.json$")
_FULLDATA_RE = re.compile(r"^(?P<job>.+)_full_data_(?P<idx>\d+)\.json$")


class Af3ServerParser:
    """Parse a folder produced by the AlphaFold 3 Server."""

    name = "af3_server"

    def can_handle(self, path: Path) -> bool:
        path = Path(path)
        if not path.is_dir():
            return False
        has_summary = has_fulldata = False
        for entry in path.iterdir():
            if _SUMMARY_RE.match(entry.name):
                has_summary = True
            elif _FULLDATA_RE.match(entry.name):
                has_fulldata = True
            if has_summary and has_fulldata:
                return True
        return False

    def parse(self, path: Path) -> list[Prediction]:
        path = Path(path)
        models = self._index(path, _MODEL_RE)
        summaries = self._index(path, _SUMMARY_RE)
        fulldata = self._index(path, _FULLDATA_RE)

        # The job request (one per folder) records the seeds and request dialect.
        provenance = _provenance(path)

        predictions: list[Prediction] = []
        for (job, idx), struct_path in sorted(models.items(), key=lambda kv: kv[0][1]):
            summary = _load_json(summaries.get((job, idx)))
            data = _load_json(fulldata.get((job, idx)))

            plddt = base.plddt_from_bfactors(struct_path)
            pae = _as_matrix(data.get("pae"))

            structure = base.read_structure(struct_path)
            chains = base.chains_from_structure(structure)
            n_residues = sum(c.n_residues for c in chains)

            metrics = PredictionMetrics(
                mean_plddt=float(np.mean(plddt)) if plddt else None,
                ptm=_as_opt_float(summary.get("ptm")),
                iptm=_as_opt_float(summary.get("iptm")),
                mpdockq=None,  # AF3 Server does not report mpDockQ.
                n_chains=len(chains),
                n_residues=n_residues,
                ranking_score=_as_opt_float(summary.get("ranking_score")),
            )

            raw_files: dict[str, Path] = {"structure": struct_path}
            if summaries.get((job, idx)):
                raw_files["summary"] = summaries[(job, idx)]
            if fulldata.get((job, idx)):
                raw_files["full_data"] = fulldata[(job, idx)]

            predictions.append(
                Prediction(
                    name=f"{_clean_job(job)}_model_{idx}",
                    source_tool=self.name,
                    structure_path=struct_path,
                    chains=chains,
                    plddt=plddt,
                    pae=pae,
                    metrics=metrics,
                    rank=int(idx) + 1,  # model_0 is the top-ranked model
                    provenance=provenance,
                    raw_files=raw_files,
                )
            )
        return predictions

    @staticmethod
    def _index(path: Path, pattern: re.Pattern) -> dict[tuple[str, str], Path]:
        out: dict[tuple[str, str], Path] = {}
        for entry in path.iterdir():
            m = pattern.match(entry.name)
            if m:
                out[(m["job"], m["idx"])] = entry
        return out


def _clean_job(job: str) -> str:
    return job[len("fold_"):] if job.startswith("fold_") else job


def _provenance(path: Path) -> Provenance:
    """Reproducibility metadata from ``*_job_request.json``.

    The AF3 Server download does not stamp an explicit model version, so we record
    only what the request truly contains (seeds, dialect, request version) and leave
    everything else as ``None``.
    """
    prov = Provenance()
    request = _load_job_request(path)
    if not request:
        return prov

    seeds = request.get("modelSeeds") or request.get("model_seeds")
    if isinstance(seeds, list):
        prov.seeds = [int(s) for s in seeds if _is_intlike(s)]

    dialect = request.get("dialect")
    version = request.get("version")
    if dialect is not None:
        # e.g. "alphafoldserver" (+ schema version) — provenance, not the model version.
        prov.extra["request_dialect"] = (
            f"{dialect} v{version}" if version is not None else str(dialect)
        )
    elif version is not None:
        prov.extra["request_version"] = str(version)
    return prov


def _load_job_request(path: Path) -> dict:
    """Load the per-folder job request, tolerating the list- or dict-shaped form."""
    for entry in path.iterdir():
        if entry.name.endswith("_job_request.json"):
            try:
                with open(entry, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except (json.JSONDecodeError, UnicodeDecodeError, OSError):
                return {}
            if isinstance(data, list):
                return data[0] if data and isinstance(data[0], dict) else {}
            return data if isinstance(data, dict) else {}
    return {}


def _is_intlike(value) -> bool:
    try:
        int(value)
        return True
    except (TypeError, ValueError):
        return False


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

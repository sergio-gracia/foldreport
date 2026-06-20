"""Parser for ColabFold output folders.

ColabFold writes, per job, a set of files like::

    <job>_unrelaxed_rank_001_alphafold2_ptm_model_3_seed_000.pdb
    <job>_relaxed_rank_001_alphafold2_ptm_model_3_seed_000.pdb     (if amber-relaxed)
    <job>_scores_rank_001_alphafold2_ptm_model_3_seed_000.json
    <job>.a3m / config.json / cite.bibtex / *_plddt.png / *_pae.png ...

The per-model ``*_scores_rank_*.json`` holds ``plddt`` (per residue), ``pae``
(N x N matrix), ``ptm`` and, for multimers, ``iptm``. The pLDDT is also stored in the
B-factor column of the structure, which we use as a fallback.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from foldreport.models import Prediction, PredictionMetrics
from foldreport.parsers import base

# <job>_<unrelaxed|relaxed>_rank_<NNN>_<tag>.pdb
_STRUCT_RE = re.compile(
    r"^(?P<job>.+)_(?P<state>unrelaxed|relaxed)_rank_(?P<rank>\d+)_(?P<tag>.+)\.(?:pdb|cif)$"
)
# <job>_scores_rank_<NNN>_<tag>.json
_SCORES_RE = re.compile(r"^(?P<job>.+)_scores_rank_(?P<rank>\d+)_(?P<tag>.+)\.json$")


class ColabFoldParser:
    """Parse a folder produced by ColabFold."""

    name = "colabfold"

    def can_handle(self, path: Path) -> bool:
        path = Path(path)
        if not path.is_dir():
            return False
        for entry in path.iterdir():
            if _SCORES_RE.match(entry.name) or _STRUCT_RE.match(entry.name):
                return True
        return False

    def parse(self, path: Path) -> list[Prediction]:
        path = Path(path)
        scores_by_key = self._index_scores(path)
        structures = self._index_structures(path)

        predictions: list[Prediction] = []
        for key, struct_path in sorted(structures.items(), key=lambda kv: kv[0][1]):
            job, rank, tag = key
            scores_path = scores_by_key.get(key)
            scores = _load_json(scores_path) if scores_path else {}

            plddt = _as_float_list(scores.get("plddt"))
            if not plddt:
                # Fall back to B-factors stored in the structure.
                plddt = base.plddt_from_bfactors(struct_path)

            pae = _as_matrix(scores.get("pae"))

            structure = base.read_structure(struct_path)
            chains = base.chains_from_structure(structure)
            n_residues = sum(c.n_residues for c in chains)

            metrics = PredictionMetrics(
                mean_plddt=float(np.mean(plddt)) if plddt else None,
                ptm=_as_opt_float(scores.get("ptm")),
                iptm=_as_opt_float(scores.get("iptm")),
                mpdockq=None,  # ColabFold does not report mpDockQ.
                n_chains=len(chains),
                n_residues=n_residues,
            )

            raw_files: dict[str, Path] = {"structure": struct_path}
            if scores_path:
                raw_files["scores"] = scores_path

            predictions.append(
                Prediction(
                    name=f"{job}_rank_{int(rank):03d}",
                    source_tool=self.name,
                    structure_path=struct_path,
                    chains=chains,
                    plddt=plddt,
                    pae=pae,
                    metrics=metrics,
                    rank=int(rank),
                    raw_files=raw_files,
                )
            )
        return predictions

    @staticmethod
    def _index_scores(path: Path) -> dict[tuple[str, str, str], Path]:
        out: dict[tuple[str, str, str], Path] = {}
        for entry in path.iterdir():
            m = _SCORES_RE.match(entry.name)
            if m:
                out[(m["job"], m["rank"], m["tag"])] = entry
        return out

    @staticmethod
    def _index_structures(path: Path) -> dict[tuple[str, str, str], Path]:
        """Map (job, rank, tag) -> best structure file, preferring relaxed over unrelaxed."""
        out: dict[tuple[str, str, str], Path] = {}
        preferred_state: dict[tuple[str, str, str], str] = {}
        for entry in path.iterdir():
            m = _STRUCT_RE.match(entry.name)
            if not m:
                continue
            key = (m["job"], m["rank"], m["tag"])
            state = m["state"]
            if key not in out or (state == "relaxed" and preferred_state[key] != "relaxed"):
                out[key] = entry
                preferred_state[key] = state
        return out


def _load_json(path: Path) -> dict:
    """Load a scores sidecar, tolerating a corrupt file by falling back to {}.

    A single malformed JSON must not abort the whole report: the structure's
    B-factor pLDDT still works and the JSON-only metrics simply stay None.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return {}


def _as_float_list(value) -> list[float]:
    if value is None:
        return []
    return [float(v) for v in value]


def _as_opt_float(value) -> float | None:
    return None if value is None else float(value)


def _as_matrix(value) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    return arr if arr.ndim == 2 else None

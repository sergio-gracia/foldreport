"""Parser for Boltz output folders.

Boltz writes, per input, a ``predictions/<name>/`` directory::

    predictions/<name>/<name>_model_0.cif
    predictions/<name>/confidence_<name>_model_0.json  -> ptm, iptm, complex_plddt, ...
    predictions/<name>/plddt_<name>_model_0.npz        -> per-residue pLDDT array
    predictions/<name>/pae_<name>_model_0.npz          -> PAE matrix
    predictions/<name>/pde_<name>_model_0.npz

Boltz reports pLDDT on a 0-1 scale; we rescale to the 0-100 convention used by the
AlphaFold family so figures and rankings are comparable across tools.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from foldreport.models import Prediction, PredictionMetrics
from foldreport.parsers import base

_MODEL_RE = re.compile(r"^(?P<name>.+)_model_(?P<idx>\d+)\.(?:cif|pdb)$")
_CONF_RE = re.compile(r"^confidence_(?P<name>.+)_model_(?P<idx>\d+)\.json$")
_PLDDT_RE = re.compile(r"^plddt_(?P<name>.+)_model_(?P<idx>\d+)\.npz$")
_PAE_RE = re.compile(r"^pae_(?P<name>.+)_model_(?P<idx>\d+)\.npz$")


class BoltzParser:
    """Parse a folder produced by Boltz."""

    name = "boltz"

    def can_handle(self, path: Path) -> bool:
        return any(True for _ in self._iter_model_dirs(Path(path)))

    def parse(self, path: Path) -> list[Prediction]:
        path = Path(path)
        predictions: list[Prediction] = []
        for pred_dir in self._iter_model_dirs(path):
            predictions.extend(self._parse_dir(pred_dir))
        # Stable ordering: by name then model index.
        predictions.sort(key=lambda p: (p.name, p.rank or 0))
        return predictions

    def _parse_dir(self, pred_dir: Path) -> list[Prediction]:
        models = _index(pred_dir, _MODEL_RE)
        confs = _index(pred_dir, _CONF_RE)
        plddts = _index(pred_dir, _PLDDT_RE)
        paes = _index(pred_dir, _PAE_RE)

        out: list[Prediction] = []
        for (name, idx), struct_path in sorted(models.items(), key=lambda kv: kv[0][1]):
            conf = _load_json(confs.get((name, idx)))

            plddt = _rescale_plddt(_load_npz(plddts.get((name, idx))))
            if not plddt:
                plddt = _rescale_plddt(base.plddt_from_bfactors(struct_path))
            pae = _load_npz_matrix(paes.get((name, idx)))

            structure = base.read_structure(struct_path)
            chains = base.chains_from_structure(structure)
            n_residues = sum(c.n_residues for c in chains)

            metrics = PredictionMetrics(
                mean_plddt=float(np.mean(plddt)) if plddt else _complex_plddt(conf),
                ptm=_as_opt_float(conf.get("ptm")),
                iptm=_as_opt_float(conf.get("iptm")),
                mpdockq=None,  # Boltz does not report mpDockQ.
                n_chains=len(chains),
                n_residues=n_residues,
                ranking_score=_as_opt_float(conf.get("confidence_score")),
            )

            raw_files: dict[str, Path] = {"structure": struct_path}
            for role, table in (("confidence", confs), ("plddt", plddts), ("pae", paes)):
                if table.get((name, idx)):
                    raw_files[role] = table[(name, idx)]

            out.append(
                Prediction(
                    name=f"{name}_model_{idx}",
                    source_tool=self.name,
                    structure_path=struct_path,
                    chains=chains,
                    plddt=plddt,
                    pae=pae,
                    metrics=metrics,
                    rank=int(idx) + 1,  # model_0 is the top-ranked model
                    raw_files=raw_files,
                )
            )
        return out

    @staticmethod
    def _iter_model_dirs(path: Path):
        """Yield directories that contain at least one Boltz confidence file.

        Accepts the run root, a ``predictions`` dir, or a single ``<name>`` dir.
        """
        if not path.is_dir():
            return
        seen: set[Path] = set()
        candidates = [path]
        predictions_dir = path / "predictions"
        if predictions_dir.is_dir():
            candidates.append(predictions_dir)
        for base_dir in candidates:
            for entry in base_dir.rglob("*"):
                if entry.is_file() and _CONF_RE.match(entry.name):
                    parent = entry.parent
                    if parent not in seen:
                        seen.add(parent)
                        yield parent


def _index(directory: Path, pattern: re.Pattern) -> dict[tuple[str, str], Path]:
    out: dict[tuple[str, str], Path] = {}
    for entry in directory.iterdir():
        m = pattern.match(entry.name)
        if m:
            out[(m["name"], m["idx"])] = entry
    return out


def _load_json(path: Path | None) -> dict:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_npz(path: Path | None) -> list[float]:
    arr = _load_npz_array(path)
    return arr.astype(float).ravel().tolist() if arr is not None else []


def _load_npz_matrix(path: Path | None) -> np.ndarray | None:
    arr = _load_npz_array(path)
    if arr is None:
        return None
    arr = arr.astype(float)
    return arr if arr.ndim == 2 else None


def _load_npz_array(path: Path | None) -> np.ndarray | None:
    """Load the single array from a Boltz .npz, tolerating the key name used."""
    if path is None:
        return None
    with np.load(path) as data:
        for key in ("plddt", "pae", "pde", "arr_0"):
            if key in data:
                return np.asarray(data[key])
        keys = list(data.keys())
        if keys:
            return np.asarray(data[keys[0]])
    return None


def _rescale_plddt(values: list[float]) -> list[float]:
    """Boltz reports pLDDT in [0, 1]; rescale to the [0, 100] convention."""
    if not values:
        return []
    if max(values) <= 1.5:
        return [v * 100.0 for v in values]
    return values


def _complex_plddt(conf: dict) -> float | None:
    value = conf.get("complex_plddt")
    if value is None:
        return None
    value = float(value)
    return value * 100.0 if value <= 1.5 else value


def _as_opt_float(value) -> float | None:
    return None if value is None else float(value)

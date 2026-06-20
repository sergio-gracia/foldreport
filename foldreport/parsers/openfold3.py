"""Parser for OpenFold3 output, in both layouts the tool emits.

**Local (CLI) layout** — OpenFold3 (aqlaboratory/openfold-3, released March 2026)
writes one directory per job, with one subdirectory per (seed, sample) draw::

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

**Server (API) layout** — the OpenFold3 server returns a single JSON file with the
structures *embedded as mmCIF/PDB strings* and only scalar confidence scores (no PAE
matrix, no per-atom arrays)::

    {"request_id": ..., "outputs": [
        {"input_id": ..., "structures_with_scores": [
            {"structure": "<cif text>", "format": "cif", "name": ...,
             "source": "seed_42", "confidence_score": ..., "complex_plddt_score": ...,
             "ptm_score": ..., "iptm_score": ...}, ...]}]}

For this layout each embedded structure is materialized to a sibling ``.cif``/``.pdb``
file (the report's 3D viewer needs a real structure file), per-residue pLDDT is read
from its B-factors, and PAE is ``None`` since the server does not return it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from foldreport.models import Prediction, PredictionMetrics, Provenance
from foldreport.parsers import base

_SAMPLE_DIR_RE = re.compile(r"^seed-(?P<seed>\d+)_sample-(?P<sample>\d+)$")
_SUMMARY_RE = re.compile(r"^(?P<job>.+)_summary_confidences\.json$")
# Must not also match "<job>_summary_confidences.json"; the negative lookbehind
# keeps the two files distinct regardless of directory iteration order.
_CONF_RE = re.compile(r"^(?P<job>.+)(?<!_summary)_confidences\.json$")
_MODEL_RE = re.compile(r"^(?P<job>.+)_model\.(?:cif|pdb)$")
_EXP_CONFIG_RE = re.compile(r"^experiment_config\.json$")


class OpenFold3Parser:
    """Parse a folder produced by OpenFold3."""

    name = "openfold3"

    def can_handle(self, path: Path) -> bool:
        path = Path(path)
        if any(True for _ in self._iter_sample_dirs(path)):
            return True
        return any(True for _ in _iter_server_json_files(path))

    def parse(self, path: Path) -> list[Prediction]:
        path = Path(path)
        predictions: list[Prediction] = []
        # Local (CLI) layout: one seed-*_sample-* directory per draw.
        for sample_dir in self._iter_sample_dirs(path):
            pred = self._parse_sample(sample_dir)
            if pred is not None:
                predictions.append(pred)
        # Server (API) layout: structures embedded in a single JSON file.
        for json_path in _iter_server_json_files(path):
            predictions.extend(_parse_server_json(json_path, self.name))
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
            provenance=_provenance(sample_dir),
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


def _provenance(sample_dir: Path) -> Provenance:
    """Reproducibility metadata for one OpenFold3 draw.

    The seed and sample index are encoded in the directory name (``seed-S_sample-N``);
    ``experiment_config.json`` at the job root may add the model version / weights.
    """
    prov = Provenance()
    m = _SAMPLE_DIR_RE.match(sample_dir.name)
    if m:
        prov.seeds = [int(m["seed"])]
        prov.extra["sample"] = m["sample"]

    config = _load_json(_find(sample_dir.parent, _EXP_CONFIG_RE))
    version = config.get("version") or config.get("model_version")
    if version is not None:
        prov.model_name = str(version)
    weights = config.get("weights") or config.get("checkpoint")
    if weights is not None:
        prov.extra["weights"] = str(weights)
    return prov


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


# --- Server (API) JSON layout ---------------------------------------------------------


def _iter_server_json_files(path: Path):
    """Yield top-level ``*.json`` files at ``path`` that match the server schema."""
    if not path.is_dir():
        return
    for entry in sorted(path.glob("*.json")):
        if not entry.is_file():
            continue
        try:
            with open(entry, "r", encoding="utf-8") as fh:
                obj = json.load(fh)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        if _is_server_json(obj):
            yield entry


def _is_server_json(obj) -> bool:
    """Recognize the OpenFold3 server response: outputs[].structures_with_scores[]."""
    if not isinstance(obj, dict):
        return False
    outputs = obj.get("outputs")
    if not isinstance(outputs, list):
        return False
    return any(
        isinstance(o, dict) and isinstance(o.get("structures_with_scores"), list)
        for o in outputs
    )


def _parse_server_json(json_path: Path, source_tool: str) -> list[Prediction]:
    """Parse every embedded structure in one OpenFold3 server JSON file."""
    with open(json_path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)

    predictions: list[Prediction] = []
    for output in doc.get("outputs", []):
        if not isinstance(output, dict):
            continue
        input_id = str(output.get("input_id") or json_path.stem)
        for idx, entry in enumerate(output.get("structures_with_scores", [])):
            if not isinstance(entry, dict):
                continue
            pred = _parse_server_entry(json_path, input_id, idx, entry, source_tool)
            if pred is not None:
                predictions.append(pred)
    return predictions


def _parse_server_entry(
    json_path: Path, input_id: str, idx: int, entry: dict, source_tool: str
) -> Prediction | None:
    name = str(entry.get("name") or f"{input_id}_sample_{idx}")
    model_path = _materialize_structure(json_path, name, entry)
    if model_path is None:
        return None

    plddt = base.plddt_from_bfactors(model_path)
    structure = base.read_structure(model_path)
    chains = base.chains_from_structure(structure)
    n_residues = sum(c.n_residues for c in chains)

    mean_plddt = _as_opt_float(entry.get("complex_plddt_score"))
    if mean_plddt is None and plddt:
        mean_plddt = float(np.mean(plddt))

    metrics = PredictionMetrics(
        mean_plddt=mean_plddt,
        ptm=_as_opt_float(entry.get("ptm_score")),
        iptm=_as_opt_float(entry.get("iptm_score")),
        mpdockq=None,  # OpenFold3 does not report mpDockQ.
        n_chains=len(chains),
        n_residues=n_residues,
        ranking_score=_as_opt_float(entry.get("confidence_score")),
    )

    return Prediction(
        name=name,
        source_tool=source_tool,
        structure_path=model_path,
        chains=chains,
        plddt=plddt,
        pae=None,  # The server layout does not return a PAE matrix.
        metrics=metrics,
        rank=None,  # assigned after global ranking in parse()
        provenance=_server_provenance(entry),
        raw_files={"structure": model_path, "server_json": json_path},
    )


def _materialize_structure(json_path: Path, name: str, entry: dict) -> Path | None:
    """Write an embedded structure string to a real file next to the JSON.

    The report's 3D viewer and the B-factor pLDDT reader both need an on-disk
    structure, so we extract each embedded mmCIF/PDB into a ``<json>_structures/``
    sibling directory. Returns None if the entry carries no structure text.
    """
    text = entry.get("structure")
    if not isinstance(text, str) or not text.strip():
        return None
    ext = "pdb" if str(entry.get("format", "cif")).lower() == "pdb" else "cif"
    out_dir = json_path.parent / f"{json_path.stem}_structures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{_safe_name(name)}.{ext}"
    out_path.write_text(text, encoding="utf-8")
    return out_path


def _server_provenance(entry: dict) -> Provenance:
    """Reproducibility metadata for one server draw (the seed is in ``source``)."""
    prov = Provenance()
    source = entry.get("source")
    if isinstance(source, str):
        m = re.search(r"(\d+)", source)
        if m:
            prov.seeds = [int(m.group(1))]
        prov.extra["source"] = source
    return prov


def _safe_name(name: str) -> str:
    safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in name)
    return safe or "structure"


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

"""Generate the example dataset shipped under ``examples/demo`` and rebuild the report.

The demo is a small multi-tool dataset: the same 56-residue two-chain complex predicted
by all four supported tools (ColabFold, AlphaFold 3 Server, Boltz, OpenFold3), two
models each, for eight pooled predictions. Each tool's files live in its own
subdirectory and faithfully mirror that tool's real layout, names and JSON keys.

Running this script regenerates the synthetic inputs under ``examples/demo/<tool>/`` and
the pooled ``examples/demo_report.html`` + ``examples/demo_metrics.csv`` so the README's
quick-start stays reproducible.

Run:  python examples/make_demo.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from foldreport.metrics import ranked_dataframe  # noqa: E402
from foldreport.parsers import parse_folder  # noqa: E402
from foldreport.report import build_report  # noqa: E402

EXAMPLES = Path(__file__).parent
DEMO = EXAMPLES / "demo"

# A 56-residue two-chain complex: chain A (32 residues) + chain B (24 residues).
_SEQ_A = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEV"
_SEQ_B = "GVALDEFTPAELRKLLDTAYKQGY"
_CHAINS = (("A", _SEQ_A), ("B", _SEQ_B))
_ONE_TO_THREE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}
_N = len(_SEQ_A) + len(_SEQ_B)


def _build_structure(plddt: list[float]):
    import gemmi

    structure = gemmi.Structure()
    structure.name = "demo"
    model = gemmi.Model("1")
    ri = 0
    x = 0.0
    for chain_name, seq in _CHAINS:
        chain = gemmi.Chain(chain_name)
        for i, letter in enumerate(seq, start=1):
            residue = gemmi.Residue()
            residue.name = _ONE_TO_THREE[letter]
            residue.seqid = gemmi.SeqId(str(i))
            atom = gemmi.Atom()
            atom.name = "CA"
            atom.element = gemmi.Element("C")
            # A gentle zig-zag so the 3D viewer shows some shape.
            atom.pos = gemmi.Position(x, 2.0 * ((i % 2) - 0.5), 0.0)
            atom.b_iso = plddt[ri]
            residue.add_atom(atom)
            chain.add_residue(residue)
            ri += 1
            x += 3.8
        model.add_chain(chain)
    structure.add_model(model)
    structure.setup_entities()
    return structure


def _plddt(rng: np.random.Generator, rank: int) -> list[float]:
    """Per-residue pLDDT; higher-ranked models are more confident on average."""
    center = 90 - (rank - 1) * 8
    return list(np.round(np.clip(rng.normal(center, 12, size=_N), 25, 99), 2))


def _pae(rng: np.random.Generator) -> np.ndarray:
    m = rng.uniform(0.5, 24.0, size=(_N, _N))
    m = np.round((m + m.T) / 2, 2)
    np.fill_diagonal(m, 0.0)
    return m


# --- Per-tool generators --------------------------------------------------------------


def _make_colabfold(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    job = "demo_complex"
    for rank, model_n, ptm, iptm in [(1, 3, 0.88, 0.81), (2, 1, 0.84, 0.73)]:
        plddt = _plddt(rng, rank)
        pae = _pae(rng)
        tag = f"alphafold2_multimer_v3_model_{model_n}_seed_000"
        _build_structure(plddt).write_pdb(str(out / f"{job}_unrelaxed_rank_{rank:03d}_{tag}.pdb"))
        scores = {
            "plddt": plddt,
            "max_pae": float(pae.max()),
            "pae": pae.tolist(),
            "ptm": ptm,
            "iptm": iptm,
        }
        (out / f"{job}_scores_rank_{rank:03d}_{tag}.json").write_text(
            json.dumps(scores), encoding="utf-8"
        )
    (out / "config.json").write_text(json.dumps({"num_models": 2}), encoding="utf-8")
    (out / "cite.bibtex").write_text("@article{mirdita2022colabfold}\n", encoding="utf-8")


def _make_af3_server(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(8)
    job = "fold_demo_complex"
    token_chain_ids = ["A"] * len(_SEQ_A) + ["B"] * len(_SEQ_B)
    token_res_ids = list(range(1, len(_SEQ_A) + 1)) + list(range(1, len(_SEQ_B) + 1))
    for idx, (ptm, iptm, rscore) in enumerate([(0.86, 0.80, 0.82), (0.82, 0.72, 0.77)]):
        plddt = _plddt(rng, idx + 1)
        pae = _pae(rng)
        _build_structure(plddt).make_mmcif_document().write_file(str(out / f"{job}_model_{idx}.cif"))
        full_data = {
            "atom_plddts": plddt,
            "pae": pae.tolist(),
            "token_chain_ids": token_chain_ids,
            "token_res_ids": token_res_ids,
        }
        (out / f"{job}_full_data_{idx}.json").write_text(json.dumps(full_data), encoding="utf-8")
        summary = {"ptm": ptm, "iptm": iptm, "ranking_score": rscore, "fraction_disordered": 0.08}
        (out / f"{job}_summary_confidences_{idx}.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
    (out / "terms_of_use.md").write_text("Terms of use.\n", encoding="utf-8")


def _make_boltz(out: Path) -> None:
    name = "demo_complex"
    pred_dir = out / "predictions" / name
    pred_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(9)
    for idx, (ptm, iptm, cscore) in enumerate([(0.85, 0.78, 0.81), (0.80, 0.70, 0.74)]):
        plddt01 = np.round(np.clip(_plddt(rng, idx + 1), 25, 99), 3) / 100.0  # Boltz 0-1 scale
        pae = _pae(rng)
        _build_structure(list(plddt01 * 100.0)).make_mmcif_document().write_file(
            str(pred_dir / f"{name}_model_{idx}.cif")
        )
        np.savez_compressed(pred_dir / f"plddt_{name}_model_{idx}.npz", plddt01)
        np.savez_compressed(pred_dir / f"pae_{name}_model_{idx}.npz", pae)
        conf = {
            "confidence_score": cscore,
            "ptm": ptm,
            "iptm": iptm,
            "complex_plddt": float(plddt01.mean()),
        }
        (pred_dir / f"confidence_{name}_model_{idx}.json").write_text(
            json.dumps(conf), encoding="utf-8"
        )


def _make_openfold3(out: Path) -> None:
    job = "demo_complex"
    root = out / job
    rng = np.random.default_rng(11)
    atoms_per_res = 3
    atom_chain_ids: list[str] = []
    atom_res_ids: list[int] = []
    for chain_name, seq in _CHAINS:
        for res_i in range(1, len(seq) + 1):
            atom_chain_ids.extend([chain_name] * atoms_per_res)
            atom_res_ids.extend([res_i] * atoms_per_res)

    specs = [("seed-1_sample-0", 0.87, 0.79, 0.83), ("seed-1_sample-1", 0.83, 0.71, 0.78)]
    for rank, (sample_name, ptm, iptm, rscore) in enumerate(specs, start=1):
        sample_dir = root / sample_name
        sample_dir.mkdir(parents=True, exist_ok=True)
        residue_plddt = _plddt(rng, rank)
        atom_plddts: list[float] = []
        for value in residue_plddt:
            atom_plddts.extend(float(np.round(value + j, 2)) for j in (-1.0, 0.0, 1.0))
        pae = _pae(rng)
        _build_structure(residue_plddt).make_mmcif_document().write_file(
            str(sample_dir / f"{job}_model.cif")
        )
        confidences = {
            "atom_plddts": atom_plddts,
            "pae": pae.tolist(),
            "pde": pae.tolist(),
            "token_chain_ids": atom_chain_ids,
            "token_res_ids": atom_res_ids,
        }
        (sample_dir / f"{job}_confidences.json").write_text(
            json.dumps(confidences), encoding="utf-8"
        )
        summary = {
            "plddt": float(np.mean(residue_plddt)),
            "ptm": ptm,
            "iptm": iptm,
            "ranking_score": rscore,
        }
        (sample_dir / f"{job}_summary_confidences.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
    (root / "experiment_config.json").write_text(json.dumps({"seeds": [1], "samples": 2}), encoding="utf-8")


_TOOL_DIRS = ("colabfold", "af3_server", "boltz", "openfold3")


def main() -> None:
    if DEMO.exists():
        shutil.rmtree(DEMO)
    _make_colabfold(DEMO / "colabfold")
    _make_af3_server(DEMO / "af3_server")
    _make_boltz(DEMO / "boltz")
    _make_openfold3(DEMO / "openfold3")

    # Pool all four tools into one ranked report, exactly as the README command does.
    predictions = []
    for tool in _TOOL_DIRS:
        predictions.extend(parse_folder(DEMO / tool))

    ranked_dataframe(predictions).to_csv(EXAMPLES / "demo_metrics.csv", index=False)
    build_report(predictions, EXAMPLES / "demo_report.html", title="FoldReport demo")
    print(
        f"Demo dataset ({len(predictions)} predictions across {len(_TOOL_DIRS)} tools) "
        f"written under {DEMO}; report + CSV regenerated under {EXAMPLES}."
    )


if __name__ == "__main__":
    main()

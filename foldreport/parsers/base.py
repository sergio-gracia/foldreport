"""Parser contract and structure-reading helpers.

Every parser implements the same two-method contract so the registry can autodetect
the right one and the rest of the pipeline stays format-agnostic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from foldreport.models import Chain, Prediction


@runtime_checkable
class Parser(Protocol):
    """Contract every tool-specific parser must satisfy.

    ``can_handle`` is a cheap structural check used for autodetection; ``parse`` does
    the real work and returns one :class:`~foldreport.models.Prediction` per model.
    """

    name: str

    def can_handle(self, path: Path) -> bool:
        """Return True if this parser recognizes the folder layout at ``path``."""
        ...

    def parse(self, path: Path) -> list[Prediction]:
        """Parse every prediction found under ``path``."""
        ...


# --- Shared structure-reading helpers -------------------------------------------------
#
# Parsers reuse these so that pLDDT-from-B-factor and chain extraction behave
# identically regardless of the source tool.


def read_structure(path: Path):
    """Read a structure file (mmCIF or PDB) into a gemmi.Structure."""
    import gemmi

    return gemmi.read_structure(str(path))


def chains_from_structure(structure) -> list[Chain]:
    """Extract per-chain metadata from the first model of a gemmi.Structure."""
    if len(structure) == 0:
        return []
    model = structure[0]
    chains: list[Chain] = []
    for chain in model:
        residues = list(chain)
        seq = _one_letter_sequence(residues)
        chains.append(
            Chain(
                chain_id=chain.name,
                n_residues=len(residues),
                sequence=seq,
            )
        )
    return chains


def plddt_from_bfactors(path: Path) -> list[float]:
    """Read per-residue pLDDT from a structure's B-factor column.

    AlphaFold-family tools store the per-residue pLDDT in the B-factor of every atom;
    all atoms of a residue share the value, so we take the CA (or first) atom per
    residue. Returns one value per residue across all chains, in file order.
    """
    structure = read_structure(path)
    if len(structure) == 0:
        return []
    model = structure[0]
    plddt: list[float] = []
    for chain in model:
        for residue in chain:
            atom = _representative_atom(residue)
            if atom is not None:
                plddt.append(float(atom.b_iso))
    return plddt


_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def _one_letter_sequence(residues) -> str | None:
    """Best-effort one-letter sequence; returns None if nothing maps."""
    letters = [_THREE_TO_ONE.get(res.name, "X") for res in residues]
    seq = "".join(letters)
    return seq if seq.strip("X") else None


def _representative_atom(residue):
    """Return the CA atom of a residue, or the first atom as a fallback."""
    for atom in residue:
        if atom.name == "CA":
            return atom
    for atom in residue:
        return atom
    return None


def aggregate_atom_plddt_to_residue(
    atom_plddts: list[float], token_res_ids: list[int], token_chain_ids: list[str]
) -> list[float]:
    """Collapse per-atom pLDDT to per-residue by averaging within each residue.

    AlphaFold 3 Server reports ``atom_plddts`` aligned with ``token_res_ids`` /
    ``token_chain_ids``. Consecutive entries sharing the same (chain, residue) belong
    to the same residue. Returns the mean pLDDT per residue in token order.
    """
    if not atom_plddts:
        return []
    values = np.asarray(atom_plddts, dtype=float)
    keys = list(zip(token_chain_ids, token_res_ids))
    residue_means: list[float] = []
    current_key = keys[0]
    bucket: list[float] = []
    for key, value in zip(keys, values):
        if key != current_key:
            residue_means.append(float(np.mean(bucket)))
            bucket = []
            current_key = key
        bucket.append(value)
    if bucket:
        residue_means.append(float(np.mean(bucket)))
    return residue_means

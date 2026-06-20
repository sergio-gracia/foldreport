"""Internal representation shared by every parser.

This is the core abstraction of FoldReport: each parser converts the on-disk output
of a prediction tool into ``list[Prediction]`` objects. Nothing downstream (metrics,
figures, report) knows the original format. Adding a new tool means writing a parser
that produces these dataclasses, not touching the rest of the codebase.

Missing values are always ``None`` — never invented. The report renders them as "N/A".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Chain:
    """A single polymer chain within a predicted structure."""

    chain_id: str
    n_residues: int
    sequence: str | None = None  # one-letter sequence, if known


@dataclass
class PredictionMetrics:
    """Normalized confidence metrics for one prediction.

    Any metric a tool does not provide stays ``None``; downstream code must tolerate
    holes and never fabricate a value.
    """

    mean_plddt: float | None = None
    ptm: float | None = None
    iptm: float | None = None
    mpdockq: float | None = None
    n_chains: int = 0
    n_residues: int = 0
    # Optional ranking score reported by some tools (e.g. AF3 Server "ranking_score").
    ranking_score: float | None = None


@dataclass
class Prediction:
    """A single predicted model, normalized across tools.

    Attributes:
        name: Human-readable identifier (usually derived from the file name).
        source_tool: One of "colabfold", "af3_server", "boltz", ...
        structure_path: Path to the ``.cif``/``.pdb`` holding coordinates.
        chains: Per-chain metadata in canonical order.
        plddt: Per-residue pLDDT in canonical residue order (0-100). Empty if unknown.
        pae: Predicted Aligned Error matrix (N_tokens x N_tokens) or ``None``.
        metrics: Normalized scalar metrics.
        rank: Tool-reported rank (1 = best) when available, else ``None``.
        raw_files: Provenance — maps a logical role to the file it came from.
    """

    name: str
    source_tool: str
    structure_path: Path
    chains: list[Chain] = field(default_factory=list)
    plddt: list[float] = field(default_factory=list)
    pae: np.ndarray | None = None
    metrics: PredictionMetrics = field(default_factory=PredictionMetrics)
    rank: int | None = None
    raw_files: dict[str, Path] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Keep the PAE as a float ndarray for consistent downstream handling.
        if self.pae is not None and not isinstance(self.pae, np.ndarray):
            self.pae = np.asarray(self.pae, dtype=float)

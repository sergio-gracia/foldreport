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
class Provenance:
    """Reproducibility metadata captured from a prediction's own files.

    The "future you" (or a reviewer) should be able to read off *what produced this
    figure*: which model, which MSA, which seed, which database snapshot, on which
    date. Every field is optional and is populated **only** from data found on disk —
    never inferred or invented, exactly like :class:`PredictionMetrics`. Whatever a
    tool records but doesn't map to a common field goes in ``extra`` so we can surface
    it without inventing a fixed schema for every tool.

    Attributes:
        model_name: The model / preset that ran (e.g. "alphafold2_multimer_v3", "v6").
        tool_version: Version string of the predicting tool (e.g. ColabFold "1.5.5").
        seeds: Random seeds used, when recorded.
        msa_mode: How the MSA was built (e.g. "mmseqs2_uniref_env", "single_sequence").
        msa_depth: Number of sequences in the MSA, when known.
        num_recycles: Recycle count, when recorded.
        use_templates: Whether structural templates were used, when recorded.
        database_snapshot: A database version/date stamp, when recorded.
        created_date: Date the prediction was produced, when recorded (as found).
        extra: Tool-specific key/value pairs that don't map to the fields above.
    """

    model_name: str | None = None
    tool_version: str | None = None
    seeds: list[int] = field(default_factory=list)
    msa_mode: str | None = None
    msa_depth: int | None = None
    num_recycles: int | None = None
    use_templates: bool | None = None
    database_snapshot: str | None = None
    created_date: str | None = None
    extra: dict[str, str] = field(default_factory=dict)

    def is_empty(self) -> bool:
        """True when nothing was captured (so the report can render "N/A")."""
        return (
            self.model_name is None
            and self.tool_version is None
            and not self.seeds
            and self.msa_mode is None
            and self.msa_depth is None
            and self.num_recycles is None
            and self.use_templates is None
            and self.database_snapshot is None
            and self.created_date is None
            and not self.extra
        )


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
        provenance: Reproducibility metadata captured from the prediction's files.
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
    provenance: Provenance = field(default_factory=Provenance)
    raw_files: dict[str, Path] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Keep the PAE as a float ndarray for consistent downstream handling.
        if self.pae is not None and not isinstance(self.pae, np.ndarray):
            self.pae = np.asarray(self.pae, dtype=float)

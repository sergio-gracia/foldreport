"""Parser registry and format autodetection.

Each tool-specific parser registers here. ``detect_parser`` asks every parser whether
it recognizes a folder, so the user never has to declare the format.
"""

from __future__ import annotations

from pathlib import Path

from foldreport.models import Prediction
from foldreport.parsers.af3_server import Af3ServerParser
from foldreport.parsers.alphafold_db import AlphaFoldDBParser
from foldreport.parsers.base import Parser
from foldreport.parsers.boltz import BoltzParser
from foldreport.parsers.colabfold import ColabFoldParser
from foldreport.parsers.openfold3 import OpenFold3Parser

# Order matters only as a tie-breaker; can_handle checks should be mutually exclusive
# in practice. More specific layouts come first.
ALL_PARSERS: list[Parser] = [
    ColabFoldParser(),
    Af3ServerParser(),
    OpenFold3Parser(),
    AlphaFoldDBParser(),
    BoltzParser(),
]


def detect_parser(path: Path) -> Parser | None:
    """Return the first parser that recognizes ``path``, or None."""
    path = Path(path)
    for parser in ALL_PARSERS:
        if parser.can_handle(path):
            return parser
    return None


def parse_folder(path: Path) -> list[Prediction]:
    """Autodetect the tool for ``path`` and parse all predictions.

    Raises:
        ValueError: if no registered parser recognizes the folder.
    """
    path = Path(path)
    parser = detect_parser(path)
    if parser is None:
        raise ValueError(
            f"No registered parser recognizes the folder: {path}. "
            f"Supported tools: {', '.join(p.name for p in ALL_PARSERS)}."
        )
    return parser.parse(path)


__all__ = [
    "ALL_PARSERS",
    "Parser",
    "ColabFoldParser",
    "Af3ServerParser",
    "BoltzParser",
    "OpenFold3Parser",
    "AlphaFoldDBParser",
    "detect_parser",
    "parse_folder",
]

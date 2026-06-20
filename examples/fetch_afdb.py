"""Download real predictions from the AlphaFold Protein Structure Database and report.

This pulls a small, varied set of real single-chain human proteins from
https://alphafold.ebi.ac.uk via its public ``/api/prediction/<UniProt>`` endpoint, saves
each entry's structure (mmCIF, pLDDT in B-factors), PAE JSON, and API metadata into
``examples/afdb_demo/`` using the database's own file-naming scheme, then builds a
FoldReport HTML report + CSV from the real data.

This is the only part of the project that touches the network, and it is opt-in: nothing
in the test suite depends on it. Run it to validate the pipeline on genuine biological
predictions.

Run:  python examples/fetch_afdb.py
      python examples/fetch_afdb.py P01308 P0CG48 P00698   # custom UniProt accessions
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from foldreport.metrics import ranked_dataframe  # noqa: E402
from foldreport.parsers import parse_folder  # noqa: E402
from foldreport.report import build_report  # noqa: E402

EXAMPLES = Path(__file__).parent
OUT_DIR = EXAMPLES / "afdb_demo"
API = "https://alphafold.ebi.ac.uk/api/prediction/{acc}"

# A small, varied default set of well-studied human proteins.
DEFAULT_ACCESSIONS = [
    "P01308",  # Insulin
    "P0CG48",  # Polyubiquitin-C
    "P00698",  # Lysozyme C (hen egg white reference entry)
    "P69905",  # Hemoglobin subunit alpha
]


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as resp:
        return resp.read()


def _fetch_one(acc: str) -> None:
    record = json.loads(_get(API.format(acc=acc)).decode("utf-8"))[0]
    entry_id = record["entryId"]  # e.g. AF-P01308-F1
    ver = record.get("latestVersion", 6)

    cif = _get(record["cifUrl"])
    (OUT_DIR / f"{entry_id}-model_v{ver}.cif").write_bytes(cif)

    pae = _get(record["paeDocUrl"])
    (OUT_DIR / f"{entry_id}-predicted_aligned_error_v{ver}.json").write_bytes(pae)

    (OUT_DIR / f"{entry_id}-metadata.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    label = record.get("gene") or record.get("uniprotId") or acc
    print(f"  + {acc}: {label} ({record.get('organismScientificName', '?')})")


def main(accessions: list[str]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Fetching {len(accessions)} entries from AlphaFold DB into {OUT_DIR} ...")
    fetched = 0
    for acc in accessions:
        try:
            _fetch_one(acc)
            fetched += 1
        except Exception as exc:  # network/HTTP/JSON issues: skip, keep going
            print(f"  ! {acc}: skipped ({exc})", file=sys.stderr)

    if not fetched:
        print("No entries downloaded; cannot build a report.", file=sys.stderr)
        sys.exit(1)

    predictions = parse_folder(OUT_DIR)
    ranked_dataframe(predictions).to_csv(EXAMPLES / "afdb_metrics.csv", index=False)
    build_report(
        predictions,
        EXAMPLES / "afdb_report.html",
        title="FoldReport - AlphaFold DB sample",
    )
    print(
        f"Report ({len(predictions)} real predictions): {EXAMPLES / 'afdb_report.html'}\n"
        f"Metrics CSV: {EXAMPLES / 'afdb_metrics.csv'}"
    )


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT_ACCESSIONS)

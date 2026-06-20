"""Command-line interface: ``foldreport <folder> [...] -o report.html``.

Point it at one or more folders of predictions. Each folder's format is autodetected;
all predictions are pooled, ranked by confidence, and written to a single HTML file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

from foldreport import __version__
from foldreport.metrics import ranked_dataframe
from foldreport.parsers import detect_parser, parse_folder
from foldreport.report import build_report


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument(
    "folders",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("foldreport.html"),
    show_default=True,
    help="Path of the self-contained HTML report to write.",
)
@click.option(
    "-t",
    "--title",
    default="FoldReport",
    show_default=True,
    help="Title shown at the top of the report.",
)
@click.option(
    "--csv",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Also write the ranked metrics table to this CSV path.",
)
@click.version_option(__version__, "-V", "--version", prog_name="foldreport")
def main(folders: tuple[Path, ...], output: Path, title: str, csv: Path | None) -> None:
    """Build a single HTML report from prediction FOLDERS.

    Supported tools (autodetected): ColabFold, AlphaFold 3 Server, Boltz, OpenFold3,
    and AlphaFold DB downloads.
    """
    predictions = []
    for folder in folders:
        parser = detect_parser(folder)
        if parser is None:
            click.echo(f"  ! Skipping {folder}: no supported format detected.", err=True)
            continue
        found = parse_folder(folder)
        click.echo(f"  + {folder}: {len(found)} prediction(s) via '{parser.name}'.")
        predictions.extend(found)

    if not predictions:
        click.echo("No predictions found in the given folder(s).", err=True)
        sys.exit(1)

    if csv is not None:
        ranked_dataframe(predictions).to_csv(csv, index=False)
        click.echo(f"  > Metrics table: {csv}")

    out_path = build_report(predictions, output, title=title)
    click.echo(f"  > Report ({len(predictions)} predictions): {out_path}")


if __name__ == "__main__":
    main()

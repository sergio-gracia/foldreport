"""Command-line interface: ``foldreport <folder> [...] -o report.html``.

Point it at one or more folders of predictions. Each folder's format is autodetected;
all predictions are pooled, ranked by confidence, and written to a single HTML file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

from foldreport import __version__, figures
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
@click.option(
    "--figures-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Also export submission-ready pLDDT/PAE figures (one set per prediction) here.",
)
@click.option(
    "--figure-format",
    type=click.Choice(["pdf", "svg", "png"]),
    multiple=True,
    default=("pdf", "png"),
    show_default=True,
    help="Format(s) for exported figures. Repeat to write several.",
)
@click.option(
    "--dpi",
    type=click.IntRange(min=72),
    default=300,
    show_default=True,
    help="Resolution for raster (PNG) exported figures.",
)
@click.option(
    "--colorblind",
    is_flag=True,
    default=False,
    help="Use a colorblind-safe palette for the pLDDT confidence bands.",
)
@click.version_option(__version__, "-V", "--version", prog_name="foldreport")
def main(
    folders: tuple[Path, ...],
    output: Path,
    title: str,
    csv: Path | None,
    figures_dir: Path | None,
    figure_format: tuple[str, ...],
    dpi: int,
    colorblind: bool,
) -> None:
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

    if figures_dir is not None:
        n_files = 0
        for pred in predictions:
            n_files += len(
                figures.save_publication_figures(
                    pred,
                    figures_dir,
                    dpi=dpi,
                    formats=tuple(figure_format),
                    colorblind=colorblind,
                )
            )
        click.echo(f"  > Figures ({n_files} files): {figures_dir}")

    out_path = build_report(predictions, output, title=title, colorblind=colorblind)
    click.echo(f"  > Report ({len(predictions)} predictions): {out_path}")


if __name__ == "__main__":
    main()

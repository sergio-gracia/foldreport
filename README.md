# FoldReport

**Point it at a folder of structure predictions and get one self-contained HTML report
that ranks them all by confidence.**

### ▶ [Try the live demo report →](https://sergio-gracia.github.io/foldreport/)

See exactly what you get before installing anything: four real AlphaFold DB proteins —
insulin, polyubiquitin-C, lysozyme C and hemoglobin subunit alpha — ranked in one page.
Click a row to open its detail card — interactive 3D viewer colored by pLDDT, per-residue
pLDDT plot, PAE heatmap, interface metrics, and a provenance panel (model version, dates,
organism) read straight from the database files. It is the exact file `foldreport` writes,
served as-is.

FoldReport reads the outputs of modern structure-prediction tools — **ColabFold**,
the **AlphaFold 3 Server**, **Boltz**, and **OpenFold3** — as well as entries from the
**AlphaFold Protein Structure Database**, and unifies them into a single navigable
`.html` file: a confidence-ranked table on top (filterable by tool,
name, and confidence), and a detail card per prediction with an embedded 3D viewer
(colored by pLDDT), a per-residue pLDDT plot, an interactive PAE heatmap, and interface
metrics (pTM, ipTM, …). Each card also carries a **Provenance & reproducibility** panel
— model version, seeds, MSA mode/depth, recycles, database snapshot — read straight from
each tool's own files (never invented; missing fields show as "N/A") and embedded in the
HTML in machine-readable form so a reviewer or the future you can reproduce any figure.

The report is **one file**. No server, no notebook, no internet connection, and no
adjacent assets — open it in any browser and share it as a single attachment.

## Why

Running AlphaFold is solved. The bottleneck moved *downstream*: you end up with dozens
or hundreds of output folders, in slightly different formats, and have to decide *what
to look at*. FoldReport answers "300 outputs from 3 tools — which ones matter?" in one
command.

## Install

```bash
pip install foldreport
```

Or from a checkout:

```bash
pip install .
```

## Quick start (copy-paste)

A ready-to-run example dataset ships in the repo: the same complex predicted by all
four supported tools (two models each, eight pooled predictions), one folder per tool.

```bash
foldreport examples/demo/colabfold examples/demo/af3_server examples/demo/boltz examples/demo/openfold3 -o report.html
```

Open `report.html` in your browser. That's it.

Point it at a single run the same way — the format is autodetected:

```bash
foldreport path/to/colabfold_run -o report.html
```

Pool several runs (even from different tools) into one ranked report:

```bash
foldreport run_colabfold/ run_af3/ run_boltz/ run_openfold3/ -o combined.html
```

### Options

| Flag | Description |
|------|-------------|
| `-o, --output` | Path of the HTML report to write (default `foldreport.html`). |
| `-t, --title` | Title shown at the top of the report. |
| `--csv` | Also write the ranked metrics table as CSV. |
| `--figures-dir` | Also export submission-ready pLDDT/PAE figures (one set per prediction) to this folder. |
| `--figure-format` | Format(s) for exported figures: `pdf`, `svg`, `png` (repeat to write several; default `pdf` + `png`). |
| `--dpi` | Resolution for raster (PNG) exported figures (default `300`). |
| `--colorblind` | Use a colorblind-safe palette for the pLDDT confidence bands. |
| `-V, --version` | Print version. |

### Submission-ready figures

`--figures-dir` writes standalone figures for every prediction, designed to drop into a
manuscript:

```bash
foldreport run_af3/ -o report.html --figures-dir figures/ --figure-format pdf --dpi 600
```

- **Vector PDF/SVG with embedded fonts** so a figure renders identically on any machine.
- **One shared PAE colour scale** (the AlphaFold 0–31.75 Å convention) across every
  prediction, so a darker panel always means lower error — comparisons stay visually
  honest instead of reflecting a per-figure `vmax`.
- **`--colorblind`** switches the categorical pLDDT bands to a colorblind-safe palette
  (applied consistently to the plot, the legend, and the 3D viewer).

## Supported tools

| Tool | Detected from | pLDDT | PAE | pTM / ipTM |
|------|---------------|:-----:|:---:|:----------:|
| ColabFold | `*_scores_rank_*.json` + `*_rank_*.pdb` | ✓ | ✓ | ✓ |
| AlphaFold 3 Server | `*_summary_confidences_*.json` + `*_full_data_*.json` | ✓ | ✓ | ✓ |
| Boltz | `confidence_*_model_*.json` + `*.npz` | ✓ | ✓ | ✓ |
| OpenFold3 | `seed-*_sample-*/` + `*_confidences.json` + `*_summary_confidences.json` | ✓ | ✓ | ✓ |
| AlphaFold DB | `AF-<ACC>-F*-model_v*.cif` + `*_predicted_aligned_error_v*.json` | ✓ | ✓ | — |

Metrics a tool does not provide are shown as **N/A** — never fabricated.

## How it works

Every parser converts a tool's on-disk output into a common internal representation
(`foldreport/models.py`). Nothing downstream — metrics, figures, report — knows the
original format, so adding a tool means writing one parser, not touching the rest.
Format detection is automatic; you never declare which tool produced a folder.

## Development

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"   # Windows
.venv/bin/python    -m pip install -e ".[dev]"     # macOS/Linux
.venv/Scripts/python -m pytest                      # run tests
```

The test fixtures are synthetic but faithful to each tool's real file layout, names,
and JSON keys; regenerate them with `python tests/make_fixtures.py`. Regenerate the
example dataset with `python examples/make_demo.py`.

### Try it on real biological data

To validate the pipeline on genuine predictions, download a small set of real proteins
from the [AlphaFold Protein Structure Database](https://alphafold.ebi.ac.uk):

```bash
python examples/fetch_afdb.py                      # default set: INS, UBC, LYZ, HBA1
python examples/fetch_afdb.py P01308 P0CG48 P00698 # or any UniProt accessions
```

This writes the real structures + PAE into `examples/afdb_demo/` and builds
`examples/afdb_report.html`. It is the only network-touching part of the project and is
fully opt-in — the test suite never hits the network.

## Scope

FoldReport processes *existing* predictions only. It does not run inference (no GPU, no
model), predict mutation effects, or edit structures. The deliverable is a CLI plus a
static HTML file — no backend, database, or accounts.

## License

MIT.

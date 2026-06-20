# CLAUDE.md

Guidance for working in this repository.

## Project

**FoldReport** — point the tool at a folder of structure predictions (ColabFold,
AlphaFold 3 Server, Boltz, OpenFold3) and get a single self-contained HTML report
that ranks all predictions by confidence and lets you explore each one (interactive
PAE, per-residue pLDDT, interface metrics) without installing anything else or
opening a notebook.

The full specification lives in [PLAN.md](PLAN.md).

## Language policy

**All code, comments, docstrings, identifiers, commit messages, documentation, and
user-facing strings MUST be written in English.** This applies to everything in the
repository regardless of the language used in conversation or in PLAN.md (which is in
Spanish). Do not introduce non-English text into the codebase.

## Architecture

- The heart of the project is the internal representation in `foldreport/models.py`.
  Every parser produces `list[Prediction]`; nothing downstream knows the original
  format. Adding a tool means writing a parser, not touching the rest.
- Missing metrics are `None`, never invented. The report renders "N/A".
- Format detection is automatic via each parser's `can_handle()`.

## Dev workflow

- Use the project virtualenv at `.venv`.
- Install in editable mode: `.venv\Scripts\python.exe -m pip install -e ".[dev]"`
- Run tests: `.venv\Scripts\python.exe -m pytest`
- Keep dependencies minimal — one-command install is core to adoption.

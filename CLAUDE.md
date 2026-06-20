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

## Releasing (PyPI)

The version has a **single source of truth: the git tag**. Do NOT hardcode a version
anywhere. `pyproject.toml` uses `dynamic = ["version"]` with `setuptools-scm`, and
`foldreport/__init__.py` reads `__version__` from the installed package metadata via
`importlib.metadata`. A tag `vX.Y.Z` produces version `X.Y.Z`.

To publish a new release, just create a GitHub Release — nothing to edit:

```
gh release create vX.Y.Z --generate-notes
```

This triggers `.github/workflows/publish.yml`, which builds on a clean runner (so
stale `dist/` artifacts can never be re-uploaded), runs `twine check`, and publishes
to PyPI via Trusted Publishing (OIDC, no stored token). The checkout uses
`fetch-depth: 0` so setuptools-scm can see the tags.

One-time PyPI setup: project Settings -> Publishing -> trusted publisher with owner
`sergio-gracia`, repo `foldreport`, workflow `publish.yml`, environment `pypi`.

Notes / gotchas:
- Tags must match `vX.Y.Z` exactly. A malformed tag like `v.0.2.0` (extra dot) is
  silently ignored by setuptools-scm and will not set the version.
- Never run `twine upload` by hand. If you ever do a local build, `setuptools-scm`
  yields a dev version (e.g. `0.1.1.devN+g<hash>`) when the tree is not on a tag —
  do not upload those.
- `File already exists` from PyPI means you tried to re-upload an existing version
  (filenames can't be reused). Cut a new tag/release instead.

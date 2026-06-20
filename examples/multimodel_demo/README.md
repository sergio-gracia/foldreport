# Multi-model demo — one protein, several predictors

This dataset shows the **same protein predicted by several tools**, so the report
ranks genuinely comparable predictions side by side instead of unrelated proteins.

- **Protein:** ubiquitin, 76 aa (UniProt **P62975**; the sequence is identical to
  human ubiquitin and to PDB `1UBQ` — ubiquitin is one of the most conserved
  proteins known, so every species' entry shares this exact sequence).
- **Sequence:** [`ubiquitin.fasta`](ubiquitin.fasta)

```
MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG
```

## Folder layout

Each tool gets its own subfolder; FoldReport autodetects the format of each:

```
multimodel_demo/
├── alphafold_db/   # ready — real AlphaFold DB download (AF-P62975-F1)
├── colabfold/      # drop the unzipped ColabFold result here
├── af3_server/     # drop the unzipped AlphaFold 3 Server result here
├── boltz/          # drop the Boltz output here
└── openfold3/      # optional — drop the OpenFold3 output here
```

`alphafold_db/` already contains the real AlphaFold DB prediction (structure with
per-residue pLDDT in the B-factors, PAE JSON, and provenance metadata). The other
folders are empty until you generate those predictions for the **same sequence**.

## How to generate the other predictions (all free)

**ColabFold** — Google Colab, free T4 GPU
1. Open <https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb>
2. Paste the sequence above, run all cells, download the result `.zip`.
3. Unzip into `colabfold/` (it contains `*_scores_rank_*.json` + `*_unrelaxed_rank_*.pdb`).

**AlphaFold 3 Server** — <https://alphafoldserver.com> (free, non-commercial; gives real PAE)
1. Sign in, "New job", paste the sequence, submit.
2. Download the result `.zip` and unzip into `af3_server/`
   (needs `fold_*_full_data_0.json`, `*_model_0.cif`, `*_summary_confidences_0.json`).

**Boltz** — any GPU (local or Colab)
1. `pip install boltz`
2. `boltz predict ubiquitin.fasta --out_dir boltz`

**OpenFold3** (optional, heaviest — needs GPU + weights): drop its output into `openfold3/`.

## Build the report locally

```bash
foldreport examples/multimodel_demo/alphafold_db \
           examples/multimodel_demo/colabfold \
           examples/multimodel_demo/af3_server \
           examples/multimodel_demo/boltz \
           -o report.html
```

Pass only the folders you have populated. The GitHub Pages workflow
(`.github/workflows/pages.yml`) does this automatically: once at least two tool
folders here contain results, the live demo switches to this multi-model set.

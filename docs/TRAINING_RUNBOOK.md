# Training runbook (Kaggle)

## Local tests and demo app (Windows PowerShell)

Use Python 3.10–3.13. These development dependencies run the tests and demo server without Torch:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
make test
make app
```

Install a Windows CPU-only Torch wheel only if you later want the local app to load trained weights and embed sequences:

```powershell
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-app.txt
```

1. Run `python scripts/make_kaggle_zip.py`, extract `protfunc-source.zip`, and upload its contents as a Kaggle dataset named `protfunc-source` so `src/protfunc/models.py` is visible under the dataset root. Attach it and the CAFA 5 competition dataset. Start a Kaggle notebook with GPU enabled and run `notebooks/01_prepare_and_embed.ipynb` from top to bottom. Its config cell uses the required `/kaggle/input/competitions/cafa-5-protein-function-prediction` paths and checks each required input exists before work begins; `IA.txt` is checked and reported as optional.
2. Save/download the notebook output directory `/kaggle/working/protfunc`. It contains `splits.json`, `term_lists.json`, `labels.json`, `embedding_config.json`, the sampled FASTA, and `embeddings/emb_*.npy` plus matching `ids_*.json` shard manifests. Embedding shards already present are skipped on rerun.
3. Start notebook 02 with GPU enabled, attach the same source dataset and the saved preparation output as `/kaggle/input/protfunc-prepared/protfunc` (or place prep output at `/kaggle/working/protfunc`), and run its verification cell.
4. Run the two-model training and evaluation cells. Expected outputs are `models/config.json`, `models/terms.json`, `models/names.json`, `models/parent_child.npy`, `models/flat.pt`, `models/hierarchy.pt`, `results/metrics.csv`, and `results/fmax_by_model_aspect.png`. The CSV columns are `model, aspect, fmax, micro_fmax, mean_term_ap, rare_fmax, frequent_fmax, split_method`. The optional DIAMOND section is separate and may be skipped.
5. Download the `models/` and `results/` directories into this repository. Run `python scripts/make_results_table.py` to refresh the README from the actual CSV, then run local `make test` and `make app`.

## If a Kaggle session disconnects

- Notebook 01: preserve `/kaggle/working/protfunc` as a Kaggle output before stopping. On a new session, attach that output and set `WORK` to its directory, or rerun the notebook; complete embedding shards are detected and skipped.
- Notebook 02: preserve `/kaggle/working/models` and `/kaggle/working/results`. If it disconnects during training, rerun training (the basic notebook does not checkpoint optimizer state). Do not combine metrics from different runs.
- If MMseqs2 cannot be installed or clustering fails, `splits.json` records `method: random_fallback`; this weakens leakage control and must be reported.
- Check the printed missing-file message before troubleshooting anything else. The notebook deliberately fails early if any listed CAFA input is absent.

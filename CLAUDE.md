# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A thin hydrology layer on top of the Torch Spatiotemporal library (tsl). Project files at the
top level; everything else is the tsl framework:

- `hydrology_dataset.py` — `IJCNNHydrologyDataset(TabularDataset)`: loads a USGS
  gauge adjacency-matrix CSV and a discharge CSV and adapts them to tsl's dataset API.
- `spin_hydrology.yaml` — Hydra config for training the SPIN imputation model on that
  dataset. It composes on top of `default` (i.e. `tsl/examples/imputation/config/default.yaml`),
  so it is meant to live in, or be pointed at, that config directory.
- `tutorial.ipynb` — end-to-end walkthrough (Korean) on a synthetic 8-gauge river network:
  CSV → dataset → connectivity → masks → `ImputationDataset` → DataModule → `Imputer`/GRIN →
  eval, plus a forecasting variant and how to wire the Hydra config. It is generated from a
  script and was executed cleanly; its `HydrologyDataset` subclass is the reference for the
  fixes listed under "Dataset gotchas" below. Run it from the repo root with the `hydrotgnn` kernel.
- `tsl/` — a full clone of https://github.com/scdmlab/tsl (fork of TorchSpatiotemporal/tsl,
  v0.9.6). It is a **nested git repo, not a submodule, and is untracked** by the outer repo.
  `git status` inside `tsl/` shows ~278 modified files; all but the three files listed under
  "Local patches to tsl" are CRLF/LF line-ending changes only. Do not "fix" or commit them.

The raw data (`IJCNN/adj_matrix.csv`, `IJCNN/merged_discharge_data.csv`) is **not in the
repo**; the README says to email tsui5@wisc.edu for it. `spin_hydrology.yaml` hardcodes
Windows absolute paths (`E:/WISC/...`) for both files, so they must be overridden on Linux.
`tutorial.ipynb` writes same-format synthetic CSVs to `data/synthetic/` (gitignored).

## Environment

**Always use the conda env `hydrotgnn`** (`/home/hydro/miniconda3/envs/hydrotgnn`, Python 3.11).
Prefix commands with `conda run -n hydrotgnn` or call its `bin/python` / `bin/pip` directly.
Installed and verified: torch 2.11+cu128 (RTX 5090 needs cu128), torch_geometric 2.8,
torch_scatter / torch_sparse (from `https://data.pyg.org/whl/torch-2.11.0+cu128.html`),
lightning 2.6, hydra-core, jupyter, pytest, and tsl as an editable install. A Jupyter kernel
named `hydrotgnn` is registered.

**tsl must be installed with `--config-settings editable_mode=compat`.** From the repo root,
`import tsl` otherwise resolves to the `tsl/` clone directory as a namespace package (no
`__version__`, no submodules) because the default editable finder runs after the path scan.
If that happens, reinstall:

```bash
/home/hydro/miniconda3/envs/hydrotgnn/bin/pip install --no-deps --config-settings editable_mode=compat -e tsl
```

## Commands

All tsl commands run from `tsl/`.

```bash
# Tests (pytest.ini defines markers: slow, integration). Fast suite: 44 pass in ~4 s.
cd tsl && python -m pytest -q -m 'not slow'
python -m pytest tests/test_metrics.py::test_name -v     # single test
python -m pytest tests/test_example_imputation.py -v    # 1-batch imputation smoke test (slow, integration)

# Lint/format (tsl's pre-commit: isort, yapf, flake8 --max-line-length=80)
cd tsl && pre-commit run --all-files

# Run an experiment (Hydra). `config=` and `config_path=` are tsl shorthands for
# --config-name / --config-path; any other key=value overrides the composed config.
cd tsl/examples/imputation && python run_imputation_experiment.py config=spin dataset.name=la
cd tsl/examples/forecasting && python run_traffic_experiment.py model=dcrnn dataset=la

# Tutorial notebook (from repo root); regenerate outputs headlessly with:
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=hydrotgnn --inplace tutorial.ipynb

# Smoke-test the hydrology dataset alone (expects ./IJCNN/*.csv relative to cwd)
python hydrology_dataset.py
```

Experiment outputs go to `logs/imputation/<model>/<date>/<time>/` relative to the cwd
(`hydra.run.dir` in `default.yaml`): a TensorBoard log, the best checkpoint, and the
resolved config. tsl downloads benchmark datasets into `tsl/tsl/.storage` by default
(`tsl.config.data_dir`, overridable via `data_dir=` in the config). `logs/` and `data/` are gitignored.

## Local patches to tsl (needed for PyG 2.8 / torch 2.6+)

tsl 0.9.6 predates these library versions. Three files in `tsl/tsl/` carry small, commented
compatibility patches; keep them if `tsl/` is ever re-cloned or updated:

- `transforms/imputation.py`, `transforms/rearrange.py`, `transforms/masked_subgraph.py` —
  PyG ≥2.7 makes `BaseTransform.forward` abstract and has `__call__` shallow-copy the `Data`.
  The shallow copy breaks tsl's `Data.input` / `Data.target` views (they keep pointing at the
  old storage, so after `.to('cuda')` the view tensors stay on CPU → "Expected all tensors to be
  on the same device"). Each transform now implements `forward` and overrides `__call__` to
  call it in place.
- `engines/predictor.py` `load_model` — passes `weights_only=False` to `torch.load`
  (torch ≥2.6 default `True` rejects the pickled tsl metric/model-class objects in checkpoints).

## Architecture: how a run flows through tsl

```
CSV files --> HydrologyDataset (DatetimeDataset; pandas, columns = MultiIndex(nodes, channels))
          --> ImputationDataset (torch; sliding windows of `window`/`stride`, connectivity, masks)
          --> SpatioTemporalDataModule (StandardScaler on target, temporal splitter val/test)
          --> Imputer (LightningModule wrapping the model class, loss = MaskedMAE)
          --> pytorch_lightning.Trainer (EarlyStopping + ModelCheckpoint on val_mae)
```

`tsl.experiment.Experiment` wraps `hydra.main`: it seeds, sets `cfg.run.dir`, and calls
the `run_fn`. Models are selected by string in each example script's `get_model_class`
(`rnni`, `birnni`, `grin`, `spin`, `spin-h` for imputation); datasets by string in
`get_dataset`.

### The hydrology wiring is not finished

`spin_hydrology.yaml` sets `dataset.name: hydrology` plus `adj_matrix_path` /
`discharge_data_path`, but `run_imputation_experiment.py`'s `get_dataset` only knows
`air*`, `la`, `bay` and raises `ValueError` for anything else, and nothing in `tsl/`
imports `hydrology_dataset.py`. Section 11 of `tutorial.ipynb` shows the `get_dataset`
branch to add. The dataset also needs the fixes below before the runner works on it.

### Dataset gotchas in `IJCNNHydrologyDataset` (all fixed in the notebook's `HydrologyDataset`)

- **Mask polarity is inverted.** tsl's `mask` is True where *observed*; `create_mask()` returns
  `df.isna()` (True where missing) and stores a DataFrame on `self.mask`. Pass `mask=~df.isna()`
  to the base constructor instead.
- **Edge direction is reversed.** tsl/PyG read `A[i, j]` as edge *j → i*
  (`adj_to_edge_index` transposes). The CSV means `a[i, j] > 0` = *i → j* (upstream → downstream).
  `prepare_distance_matrix()` does not transpose, so messages flow downstream → upstream. Store `dist.T`.
- **`compute_similarity` is unimplemented** and `similarity_options` is unset, so
  `get_connectivity(method='distance')` raises. Use `gaussian_kernel(self.dist, theta)`. With
  binary adjacency (distances 0/1/inf) the MetrLA-style `theta = std(finite distances)` ≈ 0.5
  gives weights ≈ 0.02, which `threshold: 0.1` in the yaml prunes entirely → **empty graph, no error**.
  Use `theta=1.0` (weights ≈ 0.37) and assert the edge count.
- **`datetime_encoded` lives on `DatetimeDataset`**, not `TabularDataset`; subclass
  `DatetimeDataset` (as every built-in tsl dataset does) so the runner's day-of-time covariate works.
- **`eval_mask` / `training_mask`** come from wrapping with `tsl.ops.imputation.add_missing_values`
  (as the runner does for `la`/`bay`).
- **GRIN needs node-level exog.** `GRINModel` concatenates `u` with per-node tensors, so a global
  `(T, 2)` covariate (what the example script passes) fails with "Tensors must have same number of
  dimensions". Broadcast to `(T, N, 2)`. GRIN also requires `embedding_size` when `merge_mode='mlp'`.
- If the adjacency and discharge CSVs disagree on node count, the dataset silently truncates
  both to the first `min(n)` nodes; check the printed warning rather than assuming alignment.

## Notes

- `README.md` starts with a stray `readme = """` and is truncated mid-"Usage Example";
  the pending `M README.md` in git status is a CRLF→LF change only.
- Chinese comments in `spin_hydrology.yaml` just mark "absolute path" and "parameters
  tuned for hydrology data" (`epochs: 10`, `batch_size: 4`).

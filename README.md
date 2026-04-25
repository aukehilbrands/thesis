## Thesis — Crime prediction with spatial spillovers (CBS neighbourhoods)

This repository contains a **notebook-based, end-to-end pipeline** that predicts neighbourhood-level **registered crime intensity** in the Netherlands using CBS neighbourhood (*buurt*) covariates, optionally augmented with **exogenous spatial spillover features** (Queen contiguity).

### Pipeline overview (run order)
- **`1 - Merging Datasets.ipynb`**: merges raw crime + neighbourhood covariates, constructs join keys, attaches neighbourhood polygons (GeoPackage), exports merged artifacts to `datasets/pre-processing/`.
- **`2 - Data Cleaning.ipynb`**: coerces types/encodings and drops unused metadata columns, exports `datasets/pre-processing/cleaned_crime_nbh_2024.csv`.
- **`3 - EDA.ipynb`**: exploratory analysis for thesis reporting (distributions, checks, missingness).
- **`4 - Data Pre-processing.ipynb`**: produces the final modeling datasets:
  - `datasets/model_ready_base.csv`
  - `datasets/model_ready_spatial.csv` (+ `datasets/model_ready_spatial.gpkg`)
- **`5 - Modeling MVP.ipynb`**: creates a **municipality-grouped held-out test split** and runs MVP cross-validation on the **training split only**.
- **`6 - Modeling Improvement and Tuning.ipynb`**: runs **CV + hyperparameter tuning on the training split only**, then refits on full train and reports **final results on the held-out test set** (tables/figures/SHAP).
- **`7 - Ethics Bias and Error Analysis.ipynb`**: performs bias/error stratification using the **held-out test predictions**.

### Data layout
See `datasets/README.md` for the expected raw input files and where intermediate/final datasets are written.

### Reproducibility & evaluation protocol (academic standard)
- **Target**: `log_crime_count = log1p(crime_count)` (fixed across base/spatial datasets).
- **Held-out test set**: municipality-grouped split (≈20% of municipalities), `random_state=42`.
- **Model selection / tuning**: 5-fold **GroupKFold by municipality** on the training split only.
- **Leakage safety**: all imputation (median) is done **inside sklearn Pipelines**, fit on training folds only.
- **Final reporting**: figures/tables/SHAP are generated from **held-out test set predictions** after refitting on the full training split.

### Environment setup
This project uses a local virtual environment under `.venv/` (not committed). Two ways to recreate the environment:

#### Option A — Conda (recommended for GeoPandas stacks)
Create the environment from `environment.yml`:

```bash
conda env create -f environment.yml
conda activate thesis-spatial-crime
python -m ipykernel install --user --name thesis-spatial-crime --display-name "thesis-spatial-crime"
```

#### Option B — pip (works if system geo deps are available)

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For exact package versions from the author’s environment, see `requirements-lock.txt`.

### Running the pipeline
Open Jupyter and run the notebooks in order:

```bash
jupyter lab
```

If you start from scratch, delete cached outputs under `outputs/modeling_improvement/` before re-running notebook 6 (tuning caches are intentionally written to disk for speed).


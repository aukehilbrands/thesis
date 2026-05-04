#!/usr/bin/env python3
"""
Read-only extraction of modeling and ethics results into a single markdown file
for external writers. Does not modify modeling notebooks or training code.

MVP CV and RandomizedSearchCV results are loaded from disk when present:
  - outputs/writer_extraction_cache/mvp_cv_summary.csv
  - outputs/writer_extraction_cache/tuning_results.json
If missing, the script recomputes them (tuning is slow). You can populate
tuning_results.json from notebook 5 tuning cell outputs (best_params + CV RMSE).
"""
from __future__ import annotations

import json
import os
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# --- Paths ---
ROOT = Path(__file__).resolve().parent
DATASETS = ROOT / "datasets"
OUT = ROOT / "outputs"
MODELING_OUT = OUT / "modeling"
ETHICS_OUT = OUT / "ethics_bias_error_analysis"
CACHE_DIR = OUT / "writer_extraction_cache"
RESULT_MD = OUT / "results_summary_for_writing.md"

MVP_CACHE_CSV = CACHE_DIR / "mvp_cv_summary.csv"
TUNING_CACHE_JSON = CACHE_DIR / "tuning_results.json"

TARGET_COL = "log_crime_count"
GROUP_COL = "gm_naam"
RANDOM_STATE = 42

SECTION2_EXCLUDE = {
    "gwb_code_10",
    "regio",
    "gm_naam",
    "log_crime_count",
    "crime_count",
}

ERRORS: List[str] = []
WARNINGS: List[str] = []


def _rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def _warn(msg: str) -> None:
    WARNINGS.append(msg)


def _err(msg: str) -> None:
    ERRORS.append(msg)


def fmt_metric(x: Any) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    return f"{float(x):,.4f}"


def fmt_int(n: int) -> str:
    return f"{int(n):,}"


def fmt_pct(x: float) -> str:
    return f"{float(x):,.4f}%"


def md_table(headers: List[str], rows: List[List[str]]) -> str:
    if not rows:
        return "| " + " | ".join(headers) + " |\n| " + " | ".join(["—"] * len(headers)) + " |\n"
    w = [len(h) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            w[i] = max(w[i], len(c))
    sep = "| " + " | ".join(h.ljust(w[i]) for i, h in enumerate(headers)) + " |\n"
    line = "| " + " | ".join("-" * w[i] for i in range(len(headers))) + " |\n"
    body = "".join(
        "| " + " | ".join(str(r[i]).ljust(w[i]) for i in range(len(headers))) + " |\n" for r in rows
    )
    return sep + line + body


# --- XGBoost (match notebook 5 env fix) ---
_libomp_candidates = [
    "/usr/local/opt/libomp/lib",
    "/opt/homebrew/opt/libomp/lib",
]
for p in _libomp_candidates:
    if os.path.isdir(p):
        os.environ["DYLD_LIBRARY_PATH"] = p + ":" + os.environ.get("DYLD_LIBRARY_PATH", "")
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = (
            p + ":" + os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")
        )

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
    XGBOOST_IMPORT_ERROR = ""
except Exception as e:
    xgb = None  # type: ignore
    XGBOOST_AVAILABLE = False
    XGBOOST_IMPORT_ERROR = repr(e)


# --- Feature selection (mirrors notebook 5) ---
ID_COLS_BASE = [
    "gwb_code_10",
    "gwb_code_8",
    "gwb_code",
    "merge_key",
    "regio",
    "gm_naam",
    "ID",
    "Perioden",
    "SoortMisdrijf",
    "recs",
    "centroid_x",
    "centroid_y",
]
ID_COLS_SPATIAL = ID_COLS_BASE.copy()
ID_COLS_BOUNDARY = ID_COLS_BASE.copy()

RAW_TARGET_LIKE = {
    "crime_count",
    "crime_rate_per_1000",
    "GeregistreerdeMisdrijven_1",
}

LEAKAGE_PATTERNS = [
    r"spillover_spec",
    r"spillover_target",
    r"target_lag",
    r"lag_y",
    r"spillover_lag",
    r"lag_.*crime",
    r"crime.*lag",
]
LEAKAGE_REGEX = re.compile("|".join(LEAKAGE_PATTERNS), flags=re.IGNORECASE)


def select_numeric_predictors(
    df: pd.DataFrame, target: str, id_cols: List[str]
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    df = df.copy()
    leak = [c for c in df.columns if LEAKAGE_REGEX.search(c)]
    exclude = set([c for c in id_cols if c in df.columns]) | set(leak) | set(RAW_TARGET_LIKE) | {target}
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    feature_cols = [c for c in num_cols if c not in exclude]
    X = df[feature_cols].copy()
    y = df[target].copy()
    info = {
        "n_features": len(feature_cols),
        "queen_lag_cols": [c for c in feature_cols if c.startswith("queen_lag_")],
        "boundary_lag_cols": [c for c in feature_cols if c.startswith("boundary_lag_")],
        "excluded_leakage_cols": leak,
    }
    return X, y, info


def make_ols_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )


def make_rf_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_leaf=2,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def make_xgb_pipeline() -> Pipeline:
    if not XGBOOST_AVAILABLE:
        raise ImportError(f"xgboost unavailable: {XGBOOST_IMPORT_ERROR}")
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                xgb.XGBRegressor(
                    n_estimators=500,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.0,
                    reg_lambda=1.0,
                    objective="reg:squarederror",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_cv_oof(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv,
    groups: Optional[pd.Series],
    model_name: str,
    dataset_name: str,
    id_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    oof_pred = np.full(len(y), np.nan, dtype=float)
    fold_rows = []
    for fold, (tr, va) in enumerate(cv.split(X, y, groups)):
        X_tr, y_tr = X.iloc[tr], y.iloc[tr]
        X_va, y_va = X.iloc[va], y.iloc[va]
        model_fold = Pipeline(steps=model.steps)
        model_fold.fit(X_tr, y_tr)
        pred_tr = model_fold.predict(X_tr)
        pred_va = model_fold.predict(X_va)
        oof_pred[va] = pred_va
        fold_rows.append(
            {
                "dataset": dataset_name,
                "model": model_name,
                "fold": fold,
                "n_train": len(tr),
                "n_val": len(va),
                "rmse_train": rmse(y_tr, pred_tr),
                "rmse_val": rmse(y_va, pred_va),
                "mae_train": float(mean_absolute_error(y_tr, pred_tr)),
                "mae_val": float(mean_absolute_error(y_va, pred_va)),
                "r2_train": float(r2_score(y_tr, pred_tr)),
                "r2_val": float(r2_score(y_va, pred_va)),
            }
        )
    fold_df = pd.DataFrame(fold_rows)
    oof_df = id_df.copy()
    oof_df["dataset"] = dataset_name
    oof_df["model"] = model_name
    oof_df["y_true"] = y.values
    oof_df["y_pred_oof"] = oof_pred
    oof_df["residual"] = oof_df["y_true"] - oof_df["y_pred_oof"]
    oof_df["abs_error"] = oof_df["residual"].abs()
    if np.isnan(oof_pred).any():
        raise RuntimeError("OOF predictions contain NaN")
    return fold_df, oof_df


def summarize_folds(fold_df: pd.DataFrame) -> pd.DataFrame:
    return (
        fold_df.groupby(["dataset", "model"])
        .agg(
            rmse_val_mean=("rmse_val", "mean"),
            rmse_val_std=("rmse_val", "std"),
            mae_val_mean=("mae_val", "mean"),
            mae_val_std=("mae_val", "std"),
            r2_val_mean=("r2_val", "mean"),
            r2_val_std=("r2_val", "std"),
        )
        .reset_index()
        .sort_values(["dataset", "rmse_val_mean"])
    )


def run_random_search(
    estimator: Pipeline, param_grid: dict, X, y, cv, groups, n_iter: int, label: str
):
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=1,
        random_state=RANDOM_STATE,
        verbose=0,
        return_train_score=False,
    )
    search.fit(X, y, **({"groups": groups} if groups is not None else {}))
    best_rmse = float(-search.best_score_)
    return search.best_estimator_, search.best_params_, best_rmse


def json_serialize_params(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if v is None or isinstance(v, (bool, str, int, float)):
            out[k] = v
        elif isinstance(v, np.integer):
            out[k] = int(v)
        elif isinstance(v, np.floating):
            out[k] = float(v)
        else:
            out[k] = str(v)
    return out


def section1_eda(lines: List[str]) -> None:
    lines.append("## Section 1: EDA summary\n")
    path = DATASETS / "model_ready_base.csv"
    if not path.exists():
        _err(f"Missing file: {path}")
        lines.append("*Dataset not found; section skipped.*\n")
        return
    df = pd.read_csv(path)
    ylog = df[TARGET_COL].dropna()
    n = len(df)
    lines.append(f"- **n (total rows):** {fmt_int(n)}\n")
    lines.append(
        f"- **log_crime_count — mean:** {fmt_metric(ylog.mean())}, "
        f"**median:** {fmt_metric(ylog.median())}, **std:** {fmt_metric(ylog.std())}, "
        f"**min:** {fmt_metric(ylog.min())}, **max:** {fmt_metric(ylog.max())}, "
        f"**25th pct:** {fmt_metric(ylog.quantile(0.25))}, **75th pct:** {fmt_metric(ylog.quantile(0.75))}\n"
    )
    cc = df["crime_count"].dropna()
    lines.append(
        f"- **crime_count (raw) — skewness:** {fmt_metric(stats.skew(cc))}, "
        f"**kurtosis:** {fmt_metric(stats.kurtosis(cc, fisher=True))}\n"
    )
    lines.append(
        f"- **log_crime_count — skewness:** {fmt_metric(stats.skew(ylog))}, "
        f"**kurtosis:** {fmt_metric(stats.kurtosis(ylog, fisher=True))}\n"
    )

    X, y, _ = select_numeric_predictors(df, TARGET_COL, ID_COLS_BASE)
    corrs = []
    for c in X.columns:
        pair = pd.concat([X[c], y], axis=1).dropna()
        if len(pair) < 3:
            continue
        r = pair[c].corr(pair[TARGET_COL], method="pearson")
        if r is not None and not np.isnan(r):
            corrs.append((c, float(r)))
    corrs.sort(key=lambda t: abs(t[1]), reverse=True)
    top = corrs[:15]
    rows = [[a, fmt_metric(b)] for a, b in top]
    lines.append("\n**Top 15 Pearson correlations with log_crime_count (numeric predictors):**\n\n")
    lines.append(md_table(["Feature", "Correlation"], rows))


def section2_dataset_sizes(lines: List[str]) -> None:
    lines.append("\n## Section 2: Dataset sizes\n")
    for tag, fname in [
        ("base", "model_ready_base.csv"),
        ("spatial", "model_ready_spatial.csv"),
        ("boundary", "model_ready_boundary.csv"),
    ]:
        path = DATASETS / fname
        if not path.exists():
            _err(f"Missing file: {path}")
            lines.append(f"\n### {tag}\n*File missing.*\n")
            continue
        df = pd.read_csv(path)
        feat_cols = [c for c in df.columns if c not in SECTION2_EXCLUDE]
        n_feat = len(feat_cols)
        n_lag = sum(
            1
            for c in feat_cols
            if c.startswith("queen_lag_") or c.startswith("boundary_lag_")
        )
        rows_na = int(df.isna().any(axis=1).sum())
        total_cells = len(df) * len(df.columns)
        miss_rate = 100.0 * float(df.isna().sum().sum()) / total_cells if total_cells else 0.0
        lines.append(f"\n### {tag}\n")
        lines.append(f"- **Total rows:** {fmt_int(len(df))}\n")
        lines.append(
            f"- **Feature columns (excl. gwb_code_10, regio, gm_naam, log_crime_count, crime_count):** {fmt_int(n_feat)}\n"
        )
        lines.append(f"- **Spatial lag feature columns (queen_lag_*, boundary_lag_*):** {fmt_int(n_lag)}\n")
        lines.append(f"- **Rows with ≥1 missing value:** {fmt_int(rows_na)}\n")
        lines.append(f"- **Overall missing value rate (% of all cells):** {fmt_pct(miss_rate)}\n")


def section3_split(lines: List[str]) -> None:
    lines.append("\n## Section 3: Train/test split summary (base dataset)\n")
    trp = DATASETS / "model_ready_base_train.csv"
    tep = DATASETS / "model_ready_base_test.csv"
    if not trp.exists() or not tep.exists():
        _err(f"Missing train/test: {trp} or {tep}")
        lines.append("*Split files not found.*\n")
        return
    tr = pd.read_csv(trp)
    te = pd.read_csv(tep)
    n_tr, n_te = len(tr), len(te)
    m_tr = tr[GROUP_COL].nunique()
    m_te = te[GROUP_COL].nunique()
    pct_row = 100.0 * n_tr / (n_tr + n_te) if (n_tr + n_te) else 0.0
    pct_te_row = 100.0 * n_te / (n_tr + n_te) if (n_tr + n_te) else 0.0
    m_all = m_tr + m_te
    pct_m_tr = 100.0 * m_tr / m_all if m_all else 0.0
    pct_m_te = 100.0 * m_te / m_all if m_all else 0.0
    set_tr = set(tr[GROUP_COL].dropna().astype(str))
    set_te = set(te[GROUP_COL].dropna().astype(str))
    disjoint = len(set_tr & set_te) == 0
    lines.append(f"- **Train rows:** {fmt_int(n_tr)}\n")
    lines.append(f"- **Test rows:** {fmt_int(n_te)}\n")
    lines.append(f"- **Unique municipalities (train):** {fmt_int(m_tr)}\n")
    lines.append(f"- **Unique municipalities (test):** {fmt_int(m_te)}\n")
    lines.append(
        f"- **Train/test split by rows:** {fmt_metric(pct_row)}% train, {fmt_metric(pct_te_row)}% test\n"
    )
    lines.append(
        f"- **Train/test split by municipalities:** {fmt_metric(pct_m_tr)}% train, {fmt_metric(pct_m_te)}% test\n"
    )
    lines.append(f"- **Municipality sets non-overlapping:** {'TRUE' if disjoint else 'FALSE'}\n")


def load_or_compute_mvp(lines: List[str]) -> pd.DataFrame:
    lines.append("\n## Section 4: MVP cross-validated results\n")
    if MVP_CACHE_CSV.exists():
        _warn(f"Loaded MVP CV from cache: {_rel(MVP_CACHE_CSV)}")
        summary = pd.read_csv(MVP_CACHE_CSV)
    else:
        train_files = {
            "base": DATASETS / "model_ready_base_train.csv",
            "spatial": DATASETS / "model_ready_spatial_train.csv",
            "boundary": DATASETS / "model_ready_boundary_train.csv",
        }
        for p in train_files.values():
            if not p.exists():
                _err(f"Missing train split for MVP: {p}")
                return pd.DataFrame()
        KEEP_CONTEXT = [c for c in ["gwb_code_10", "gm_naam", "regio", "a_inw", "crime_count"]]
        fold_tables = []
        models = {"OLS": make_ols_pipeline(), "RF": make_rf_pipeline()}
        if XGBOOST_AVAILABLE:
            models["XGB"] = make_xgb_pipeline()
        else:
            _warn("XGBoost not available; MVP table will omit XGB.")

        for ds_name, path in train_files.items():
            df = pd.read_csv(path)
            id_cols = ID_COLS_BASE if ds_name == "base" else (ID_COLS_SPATIAL if ds_name == "spatial" else ID_COLS_BOUNDARY)
            X, y, _ = select_numeric_predictors(df, TARGET_COL, id_cols)
            groups = df[GROUP_COL] if GROUP_COL in df.columns else None
            cv = GroupKFold(n_splits=5)
            kcc = [c for c in KEEP_CONTEXT if c in df.columns]
            id_df = df[kcc].copy() if kcc else pd.DataFrame(index=df.index)
            for mname, pipe in models.items():
                fdf, _ = evaluate_cv_oof(pipe, X, y, cv, groups, mname, ds_name, id_df)
                fold_tables.append(fdf)
        fold_metrics = pd.concat(fold_tables, ignore_index=True)
        summary = summarize_folds(fold_metrics)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        summary.to_csv(MVP_CACHE_CSV, index=False)
        _warn(f"Computed MVP CV and wrote cache: {_rel(MVP_CACHE_CSV)}")

    if summary.empty:
        lines.append("*MVP CV could not be computed.*\n")
        return summary

    rows = []
    for _, r in summary.sort_values("rmse_val_mean").iterrows():
        rows.append(
            [
                str(r["model"]),
                str(r["dataset"]),
                "untuned (MVP)",
                fmt_metric(r["rmse_val_mean"]),
                fmt_metric(r["rmse_val_std"]),
                fmt_metric(r["mae_val_mean"]),
                fmt_metric(r["r2_val_mean"]),
            ]
        )
    lines.append(
        md_table(
            [
                "Model",
                "Feature set",
                "Tuning status",
                "Mean CV RMSE",
                "Std CV RMSE",
                "Mean CV MAE",
                "Mean CV R²",
            ],
            rows,
        )
    )
    return summary


def load_or_compute_tuning(
    mvp_summary: pd.DataFrame, lines: List[str]
) -> Optional[Dict[str, Any]]:
    lines.append("\n## Section 5: Hyperparameter tuning results\n")
    if not XGBOOST_AVAILABLE:
        lines.append("*XGBoost unavailable; RF tuning may still run; XGB section skipped.*\n")

    if TUNING_CACHE_JSON.exists():
        _warn(f"Loaded tuning results from cache: {_rel(TUNING_CACHE_JSON)}")
        try:
            with open(TUNING_CACHE_JSON, "r", encoding="utf-8") as f:
                tuning = json.load(f)
        except Exception as ex:
            _err(f"Could not read tuning cache: {ex}")
            tuning = None
    else:
        train_files = {
            "base": DATASETS / "model_ready_base_train.csv",
            "spatial": DATASETS / "model_ready_spatial_train.csv",
            "boundary": DATASETS / "model_ready_boundary_train.csv",
        }
        for p in train_files.values():
            if not p.exists():
                _err(f"Missing train file for tuning: {p}")
                return None

        rf_param_grid = {
            "model__n_estimators": [300, 500, 800],
            "model__max_depth": [None, 10, 20, 40],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 5, 10],
            "model__max_features": ["sqrt", "log2", 0.5, 0.8],
            "model__bootstrap": [True],
        }
        xgb_param_grid = {
            "model__n_estimators": [300, 500, 800, 1000],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "model__max_depth": [3, 4, 5, 6, 8],
            "model__min_child_weight": [1, 3, 5, 10],
            "model__subsample": [0.6, 0.8, 1.0],
            "model__colsample_bytree": [0.6, 0.8, 1.0],
            "model__reg_alpha": [0, 0.01, 0.1, 1],
            "model__reg_lambda": [0.5, 1, 2, 5],
        }

        rf_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)),
            ]
        )
        xgb_pipe = None
        if XGBOOST_AVAILABLE:
            xgb_pipe = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        xgb.XGBRegressor(
                            objective="reg:squarederror",
                            random_state=RANDOM_STATE,
                            n_jobs=-1,
                        ),
                    ),
                ]
            )

        tuning: Dict[str, Any] = {"RF": {}, "XGB": {}}
        cv = GroupKFold(n_splits=5)

        for ds in ["base", "spatial", "boundary"]:
            df = pd.read_csv(train_files[ds])
            id_cols = ID_COLS_BASE if ds == "base" else (ID_COLS_SPATIAL if ds == "spatial" else ID_COLS_BOUNDARY)
            X, y, _ = select_numeric_predictors(df, TARGET_COL, id_cols)
            groups = df[GROUP_COL] if GROUP_COL in df.columns else None
            _, bp, cv_rmse = run_random_search(
                Pipeline(rf_pipe.steps), rf_param_grid, X, y, cv, groups, 30, f"RF ({ds})"
            )
            tuning["RF"][ds] = {"best_params": json_serialize_params(bp), "cv_rmse": cv_rmse}

        if xgb_pipe is not None:
            for ds in ["base", "spatial", "boundary"]:
                df = pd.read_csv(train_files[ds])
                id_cols = ID_COLS_BASE if ds == "base" else (ID_COLS_SPATIAL if ds == "spatial" else ID_COLS_BOUNDARY)
                X, y, _ = select_numeric_predictors(df, TARGET_COL, id_cols)
                groups = df[GROUP_COL] if GROUP_COL in df.columns else None
                _, bp, cv_rmse = run_random_search(
                    Pipeline(xgb_pipe.steps), xgb_param_grid, X, y, cv, groups, 40, f"XGB ({ds})"
                )
                tuning["XGB"][ds] = {"best_params": json_serialize_params(bp), "cv_rmse": cv_rmse}

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(TUNING_CACHE_JSON, "w", encoding="utf-8") as f:
            json.dump(tuning, f, indent=2)
        _warn(f"Ran RandomizedSearchCV and wrote cache: {_rel(TUNING_CACHE_JSON)}")

    if tuning is None:
        lines.append("\n*Tuning results unavailable (see extraction log).*\n")
        return None

    # Best params text
    for model in ["RF", "XGB"]:
        if model not in tuning:
            continue
        lines.append(f"\n### {model} — best hyperparameters (RandomizedSearchCV)\n")
        for ds in ["base", "spatial", "boundary"]:
            if ds not in tuning[model]:
                continue
            lines.append(f"\n**{ds}:**\n```json\n{json.dumps(tuning[model][ds]['best_params'], indent=2)}\n```\n")

    # Delta vs MVP
    delta_rows = []
    if mvp_summary is not None and not mvp_summary.empty:
        for model in ["RF", "XGB"]:
            if model not in tuning:
                continue
            for ds in ["base", "spatial", "boundary"]:
                if ds not in tuning[model]:
                    continue
                sub = mvp_summary[(mvp_summary["model"] == model) & (mvp_summary["dataset"] == ds)]
                if len(sub) != 1:
                    continue
                mvp_rmse = float(sub["rmse_val_mean"].iloc[0])
                tuned_rmse = float(tuning[model][ds]["cv_rmse"])
                d_rmse = mvp_rmse - tuned_rmse
                pct_imp = 100.0 * d_rmse / mvp_rmse if mvp_rmse else 0.0
                delta_rows.append([model, ds, fmt_metric(mvp_rmse), fmt_metric(tuned_rmse), fmt_metric(d_rmse), fmt_metric(pct_imp)])
                lines.append(
                    f"\n**ΔRMSE ({model}, {ds}):** MVP CV RMSE − tuned CV RMSE = "
                    f"{fmt_metric(mvp_rmse)} − {fmt_metric(tuned_rmse)} = **{fmt_metric(d_rmse)}** "
                    f"({fmt_metric(pct_imp)}% improvement)\n"
                )

    lines.append("\n**Summary (CV RMSE):**\n\n")
    lines.append(
        md_table(
            ["Model", "Feature set", "MVP RMSE", "Tuned RMSE", "ΔRMSE", "% improvement"],
            delta_rows,
        )
    )
    return tuning


def section6_test(lines: List[str]) -> pd.DataFrame:
    lines.append("\n## Section 6: Final held-out test set results\n")
    path = MODELING_OUT / "final_test_set_results.csv"
    if not path.exists():
        _err(f"Missing: {path}")
        lines.append("*File not found.*\n")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df.sort_values("rmse")
    rows = []
    for _, r in df.iterrows():
        rows.append(
            [
                str(r["model"]),
                str(r["dataset"]),
                str(r["tuning"]),
                fmt_metric(r["rmse"]),
                fmt_metric(r["mae"]),
                fmt_metric(r["r2"]),
            ]
        )
    lines.append(
        md_table(
            ["Model", "Feature set", "Tuning status", "Test RMSE", "Test MAE", "Test R²"],
            rows,
        )
    )
    best = df.iloc[0]
    worst = df.iloc[-1]
    lines.append(
        f"\n**Best configuration (lowest test RMSE):** {best['model']}, {best['dataset']}, {best['tuning']} — "
        f"RMSE {fmt_metric(best['rmse'])}, MAE {fmt_metric(best['mae'])}, R² {fmt_metric(best['r2'])}\n"
    )
    lines.append(
        f"**Worst configuration:** {worst['model']}, {worst['dataset']}, {worst['tuning']} — "
        f"RMSE {fmt_metric(worst['rmse'])}, MAE {fmt_metric(worst['mae'])}, R² {fmt_metric(worst['r2'])}\n"
    )
    tree = df[df["model"].isin(["RF", "XGB"])]
    ols = df[df["model"] == "OLS"]
    best_tree_rmse = float(tree["rmse"].min()) if len(tree) else float("nan")
    best_ols_rmse = float(ols["rmse"].min()) if len(ols) else float("nan")
    gap = best_ols_rmse - best_tree_rmse
    lines.append(
        f"**RMSE gap (best OLS − best tree-based):** {fmt_metric(best_ols_rmse)} − {fmt_metric(best_tree_rmse)} = **{fmt_metric(gap)}**\n"
    )

    for fam in ["OLS", "RF", "XGB"]:
        sub = df[df["model"] == fam].set_index("dataset").reindex(["base", "spatial", "boundary"])
        if sub["rmse"].isna().all():
            continue
        lines.append(f"\n### {fam} — test RMSE by feature set\n")
        rb = float(sub.loc["base", "rmse"]) if "base" in sub.index and pd.notna(sub.loc["base", "rmse"]) else float("nan")
        rs = float(sub.loc["spatial", "rmse"]) if "spatial" in sub.index and pd.notna(sub.loc["spatial", "rmse"]) else float("nan")
        rbd = float(sub.loc["boundary", "rmse"]) if "boundary" in sub.index and pd.notna(sub.loc["boundary", "rmse"]) else float("nan")
        lines.append(f"- **base:** {fmt_metric(rb)}\n")
        lines.append(f"- **spatial:** {fmt_metric(rs)} (Δ vs base: {fmt_metric(rs - rb)})\n")
        lines.append(f"- **boundary:** {fmt_metric(rbd)} (Δ vs base: {fmt_metric(rbd - rb)})\n")
    return df


def _build_fitted_best_pipe(
    best: pd.Series, tuning: Optional[Dict[str, Any]]
) -> Tuple[Pipeline, pd.DataFrame, pd.Series]:
    """Refit the winning test configuration on the corresponding training split."""
    ds = str(best["dataset"])
    model_name = str(best["model"])
    tuning_status = str(best["tuning"])
    tr_path = DATASETS / f"model_ready_{ds}_train.csv"
    if not tr_path.exists():
        raise FileNotFoundError(str(tr_path))
    df_tr = pd.read_csv(tr_path)
    id_cols = ID_COLS_BASE if ds == "base" else (ID_COLS_SPATIAL if ds == "spatial" else ID_COLS_BOUNDARY)
    X, y, _ = select_numeric_predictors(df_tr, TARGET_COL, id_cols)

    if model_name == "OLS":
        pipe = make_ols_pipeline()
    elif model_name == "RF":
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)),
            ]
        )
        if tuning_status == "tuned" and tuning and "RF" in tuning and ds in tuning["RF"]:
            bp = tuning["RF"][ds]["best_params"]
            pipe.set_params(**bp)
    elif model_name == "XGB":
        if not XGBOOST_AVAILABLE:
            raise RuntimeError("XGBoost not available")
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    xgb.XGBRegressor(
                        objective="reg:squarederror",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        if tuning_status == "tuned" and tuning and "XGB" in tuning and ds in tuning["XGB"]:
            bp = tuning["XGB"][ds]["best_params"]
            pipe.set_params(**bp)
        else:
            pipe = make_xgb_pipeline()
    else:
        raise ValueError(f"Unknown model {model_name}")

    pipe.fit(X, y)
    return pipe, X, y


def section7_shap(
    test_df: pd.DataFrame, tuning: Optional[Dict[str, Any]], lines: List[str]
) -> None:
    lines.append("\n## Section 7: SHAP feature importance\n")
    shap_files = sorted(MODELING_OUT.glob("*shap_mean_abs*"))
    shap_csv = shap_files[0] if shap_files else None
    imp_df = None

    if shap_csv and shap_csv.suffix.lower() == ".csv":
        imp_df = pd.read_csv(shap_csv)
        lines.append(f"*Loaded:* `{shap_csv.relative_to(ROOT)}`\n")
    else:
        if not shap_files:
            _warn(
                "No SHAP mean |SHAP| CSV in outputs/modeling; computing from best test configuration."
            )
        try:
            import shap
        except ImportError:
            _err("shap package not installed; cannot compute SHAP table.")
            lines.append("*SHAP skipped.*\n")
            return

        if test_df.empty:
            _err("Cannot compute SHAP: missing test results.")
            return
        best = test_df.sort_values("rmse").iloc[0]
        try:
            pipe, X, _y = _build_fitted_best_pipe(best, tuning)
        except Exception as ex:
            _err(f"SHAP refit failed: {ex}")
            lines.append("*SHAP computation skipped.*\n")
            return

        feat_names = list(X.columns)
        model_name = str(best["model"])
        eval_rows = min(2000, len(X))
        X_eval = X.sample(eval_rows, random_state=RANDOM_STATE) if len(X) > eval_rows else X
        X_imp_eval = pipe.named_steps["imputer"].transform(X_eval)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if model_name in ("XGB", "RF"):
                explainer = shap.TreeExplainer(pipe.named_steps["model"])
                sv = explainer.shap_values(X_imp_eval)
            else:
                bg = shap.sample(X, min(256, len(X)))
                explainer = shap.Explainer(pipe.predict, bg)
                exp = explainer(X_eval)
                sv = exp.values
        mean_abs = np.mean(np.abs(sv), axis=0)
        imp_df = pd.DataFrame({"feature": feat_names, "mean_abs_shap": mean_abs})
        imp_df = imp_df.sort_values("mean_abs_shap", ascending=False)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        out_csv = CACHE_DIR / "shap_mean_abs_computed.csv"
        imp_df.to_csv(out_csv, index=False)
        lines.append(
            f"*Computed mean |SHAP| on full **{best['dataset']}** training split for **{best['model']}** ({best['tuning']}); "
            f"saved to `{out_csv.relative_to(ROOT)}`.*\n"
        )

    # Normalize column names
    col_feat = "feature" if "feature" in imp_df.columns else imp_df.columns[0]
    col_val = "mean_abs_shap" if "mean_abs_shap" in imp_df.columns else imp_df.columns[-1]
    imp_df = imp_df[[col_feat, col_val]].copy()
    imp_df.columns = ["feature", "mean_abs_shap"]
    imp_df = imp_df.sort_values("mean_abs_shap", ascending=False)

    rows = [[r["feature"], fmt_metric(r["mean_abs_shap"])] for _, r in imp_df.iterrows()]
    lines.append("\n**Full table (mean |SHAP|, descending):**\n\n")
    lines.append(md_table(["Feature", "Mean |SHAP|"], rows))

    imp_df = imp_df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    imp_df["rank_overall"] = np.arange(1, len(imp_df) + 1)
    lag_mask = imp_df["feature"].str.startswith("queen_lag_") | imp_df["feature"].str.startswith(
        "boundary_lag_"
    )
    lag_df = imp_df.loc[lag_mask].copy()
    lines.append("\n**Spatial lag features only (rank = overall rank in full table):**\n\n")
    if lag_df.empty:
        lines.append(
            "*No `queen_lag_*` or `boundary_lag_*` predictors in this feature matrix "
            "(e.g. **base** feature set only has non-spatial lags).*\n"
        )
    else:
        lag_rows = [
            [fmt_int(int(r["rank_overall"])), r["feature"], fmt_metric(r["mean_abs_shap"])]
            for _, r in lag_df.iterrows()
        ]
        lines.append(md_table(["Rank", "Feature", "Mean |SHAP|"], lag_rows))

    tot = float(imp_df["mean_abs_shap"].sum())
    lag_sum = float(imp_df.loc[lag_mask, "mean_abs_shap"].sum())
    base_sum = tot - lag_sum
    pct_lag = 100.0 * lag_sum / tot if tot else 0.0
    pct_base = 100.0 * base_sum / tot if tot else 0.0
    lines.append(
        f"\n**Combined mean |SHAP| — spatial lag vs base features:** "
        f"{fmt_metric(pct_lag)}% lag, {fmt_metric(pct_base)}% non-lag (of total mean |SHAP| mass).\n"
    )


def section8_ethics(lines: List[str]) -> None:
    lines.append("\n## Section 8: Error and bias analysis\n")
    if not ETHICS_OUT.is_dir():
        _err(f"Missing directory: {ETHICS_OUT}")
        return

    oem = ETHICS_OUT / "overall_error_metrics.csv"
    if oem.exists():
        o = pd.read_csv(oem).iloc[0]
        lines.append("### overall_error_metrics.csv\n")
        lines.append(
            f"- **RMSE:** {fmt_metric(o['RMSE'])}, **MAE:** {fmt_metric(o['MAE'])}, **R²:** {fmt_metric(o['R2'])}\n"
        )
        lines.append(f"- **Mean residual (bias check):** {fmt_metric(o['mean_residual'])}\n")
    else:
        _err(f"Missing {oem}")

    fp = ETHICS_OUT / "flagged_bias_error_patterns.csv"
    if fp.exists():
        fl = pd.read_csv(fp)
        lines.append("\n### flagged_bias_error_patterns.csv\n\n")
        lines.append(md_table(list(fl.columns), [[str(fl.iloc[i, j]) for j in range(len(fl.columns))] for i in range(len(fl))]))
    else:
        _err(f"Missing {fp}")

    pop_path = ETHICS_OUT / "subgroup_error_by_population_decile.csv"
    if pop_path.exists():
        pop = pd.read_csv(pop_path)
        dec_col = "pop_decile" if "pop_decile" in pop.columns else pop.columns[0]
        pop = pop.sort_values(dec_col)
        lines.append("\n### subgroup_error_by_population_decile.csv\n\n")
        pr = pop[[dec_col, "n", "MAE", "mean_residual"]]
        lines.append(
            md_table(
                ["Decile", "n", "MAE", "Mean residual"],
                [
                    [str(int(r[dec_col])), fmt_int(int(r["n"])), fmt_metric(r["MAE"]), fmt_metric(r["mean_residual"])]
                    for _, r in pr.iterrows()
                ],
            )
        )
        low = pop[pop[dec_col] == pop[dec_col].min()]
        high = pop[pop[dec_col] == pop[dec_col].max()]
        if len(low) and len(high):
            mae_low = float(low["MAE"].iloc[0])
            mae_high = float(high["MAE"].iloc[0])
            ratio = mae_low / mae_high if mae_high else float("nan")
            lines.append(
                f"\n**MAE ratio (lowest population decile / highest population decile):** "
                f"{fmt_metric(mae_low)} / {fmt_metric(mae_high)} = **{fmt_metric(ratio)}**\n"
            )
    else:
        _err(f"Missing {pop_path}")

    tgt_path = ETHICS_OUT / "subgroup_error_by_target_decile.csv"
    if tgt_path.exists():
        tg = pd.read_csv(tgt_path)
        dec_col = "target_decile" if "target_decile" in tg.columns else tg.columns[0]
        tg = tg.sort_values(dec_col)
        lines.append("\n### subgroup_error_by_target_decile.csv\n\n")
        lines.append(
            md_table(
                ["Decile", "n", "MAE", "Mean residual"],
                [
                    [str(int(r[dec_col])), fmt_int(int(r["n"])), fmt_metric(r["MAE"]), fmt_metric(r["mean_residual"])]
                    for _, r in tg.iterrows()
                ],
            )
        )
        top = tg[tg[dec_col] == tg[dec_col].max()]
        if len(top):
            lines.append(
                f"\n**Mean residual in top target decile:** {fmt_metric(float(top['mean_residual'].iloc[0]))}\n"
            )
    else:
        _err(f"Missing {tgt_path}")

    lines.append("\n*Other files in ethics output folder (for reference):*\n")
    for p in sorted(ETHICS_OUT.iterdir()):
        if p.is_file() and p.suffix.lower() == ".csv" and p.name not in {
            "overall_error_metrics.csv",
            "flagged_bias_error_patterns.csv",
            "subgroup_error_by_population_decile.csv",
            "subgroup_error_by_target_decile.csv",
        }:
            lines.append(f"- `{p.relative_to(ROOT)}`\n")


def find_figure(keywords: List[str]) -> Optional[Path]:
    for p in ROOT.rglob("*.png"):
        name = p.name.lower()
        if all(k.lower() in name or k.lower() in str(p).lower() for k in keywords):
            if ".ipynb_checkpoints" in str(p):
                continue
            return p
    return None


def section9_figures(lines: List[str]) -> None:
    lines.append("\n## Section 9: Figures needed (checklist)\n")
    checks = [
        (
            "1. Side-by-side histogram of raw crime_count and log_crime_count",
            find_figure(["hist", "crime"]) or find_figure(["histogram", "crime"]),
        ),
        ("2. Choropleth map (neighbourhood) shaded by log_crime_count", find_figure(["choropleth"]) or find_figure(["map", "crime"])),
        ("3. Horizontal bar chart — top 15 predictor correlations with log_crime_count", find_figure(["corr", "bar"]) or find_figure(["correlation"])),
        ("4. Grouped bar chart — test RMSE, 9 configurations by feature set", find_figure(["rmse", "test"]) or find_figure(["test", "rmse"])),
        ("5. Grouped bar chart — test R², 9 configurations by feature set", find_figure(["r2", "test"])),
        (
            "6. Scatter plot — predicted vs actual log_crime_count (best model, test)",
            ETHICS_OUT / "figures" / "pred_vs_actual.png" if (ETHICS_OUT / "figures" / "pred_vs_actual.png").exists() else None,
        ),
        (
            "7. Bar chart — MAE by population decile with overall MAE reference",
            ETHICS_OUT / "figures" / "abs_error_by_population_decile.png"
            if (ETHICS_OUT / "figures" / "abs_error_by_population_decile.png").exists()
            else None,
        ),
        (
            "8. Bar chart — mean residual by target decile",
            ETHICS_OUT / "figures" / "mean_residual_by_target_decile.png"
            if (ETHICS_OUT / "figures" / "mean_residual_by_target_decile.png").exists()
            else None,
        ),
        ("9. SHAP summary beeswarm (best model)", find_figure(["shap", "beeswarm"]) or find_figure(["shap", "summary"])),
    ]
    for desc, path in checks:
        if path and path.exists():
            lines.append(f"- [x] **{desc}** — `{path.resolve().relative_to(ROOT)}`\n")
        else:
            lines.append(f"- [ ] **{desc}** — *not found as saved file; extract manually when needed*\n")


def section10_validation(lines: List[str], test_df: pd.DataFrame) -> None:
    lines.append("\n## Section 10: Validation checks\n")
    # Train muni sets across feature sets
    trs = {}
    for ds in ["base", "spatial", "boundary"]:
        p = DATASETS / f"model_ready_{ds}_train.csv"
        if p.exists():
            trs[ds] = set(pd.read_csv(p)[GROUP_COL].dropna().astype(str))
        else:
            trs[ds] = set()
            _err(f"Missing {p}")
    match = trs.get("base") == trs.get("spatial") == trs.get("boundary") if len(trs) == 3 else False
    lines.append(f"- **Same municipalities in train splits (base/spatial/boundary):** {'TRUE' if match else 'FALSE'}\n")

    disjoint_all = True
    for ds in ["base", "spatial", "boundary"]:
        trp = DATASETS / f"model_ready_{ds}_train.csv"
        tep = DATASETS / f"model_ready_{ds}_test.csv"
        if not trp.exists() or not tep.exists():
            disjoint_all = False
            continue
        tr_m = set(pd.read_csv(trp)[GROUP_COL].dropna().astype(str))
        te_m = set(pd.read_csv(tep)[GROUP_COL].dropna().astype(str))
        if tr_m & te_m:
            disjoint_all = False
    lines.append(f"- **No municipality in both train and test (all feature sets):** {'TRUE' if disjoint_all else 'FALSE'}\n")

    best = None
    if test_df is not None and not test_df.empty:
        best = test_df.sort_values("rmse").iloc[0]
        lines.append(
            f"- **SHAP / best model alignment:** Best test configuration = **{best['model']}**, "
            f"feature set **{best['dataset']}**, tuning **{best['tuning']}** "
            f"(SHAP table targets this configuration when CSV is absent and values are recomputed).\n"
        )
    else:
        lines.append("- **SHAP / best model:** Could not verify (test results missing).\n")

    pred_path = MODELING_OUT / "test_set_predictions_all_models.csv"
    if pred_path.exists():
        pr = pd.read_csv(pred_path)
        ncfg = len(pr.groupby(["model", "dataset", "tuning"]))
        ok9 = ncfg == 9
        lines.append(f"- **Test set predictions exist for all 9 configurations:** {'TRUE' if ok9 else 'FALSE'} (found {ncfg} unique configs)\n")
        if "gwb_code_10" in pr.columns:
            one_cfg = pr[(pr["model"] == "XGB") & (pr["dataset"] == "base") & (pr["tuning"] == "tuned")]
            n_nbhd = int(one_cfg["gwb_code_10"].nunique()) if len(one_cfg) else int(pr["gwb_code_10"].nunique())
        else:
            n_nbhd = len(pr) // max(ncfg, 1)
        lines.append(f"- **Number of neighbourhoods in test set (unique gwb_code_10):** {fmt_int(int(n_nbhd))}\n")
    else:
        _err(f"Missing {pred_path}")
        lines.append("- **Test predictions / neighbourhood count:** Could not verify.\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
    lines: List[str] = [
        "# Results summary for writing\n",
        f"\n*Extraction run (UTC): {ts}*\n",
        "\n---\n",
    ]
    section1_eda(lines)
    section2_dataset_sizes(lines)
    section3_split(lines)
    mvp = load_or_compute_mvp(lines)
    tuning = load_or_compute_tuning(mvp, lines)
    test_df = section6_test(lines)
    section7_shap(test_df, tuning, lines)
    section8_ethics(lines)
    section9_figures(lines)
    section10_validation(lines, test_df)

    lines.append("\n---\n\n## Extraction log\n")
    if WARNINGS:
        lines.append("**Warnings:**\n")
        for w in WARNINGS:
            lines.append(f"- {w}\n")
    else:
        lines.append("*No warnings.*\n")
    if ERRORS:
        lines.append("\n**Errors / missing files:**\n")
        for e in ERRORS:
            lines.append(f"- {e}\n")
    else:
        lines.append("\n*No errors reported.*\n")

    RESULT_MD.parent.mkdir(parents=True, exist_ok=True)
    RESULT_MD.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {RESULT_MD}")
    if ERRORS:
        print("Completed with errors; see results file extraction log.")
    else:
        print("Extraction complete.")


if __name__ == "__main__":
    main()

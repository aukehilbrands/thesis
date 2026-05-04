# Results summary for writing

*Extraction run (UTC): 2026-05-02 11:27:49 UTC*

---
## Section 1: EDA summary
- **n (total rows):** 14,494
- **log_crime_count — mean:** 3.0936, **median:** 3.1355, **std:** 1.3972, **min:** 0.0000, **max:** 8.2201, **25th pct:** 2.1972, **75th pct:** 4.0604
- **crime_count (raw) — skewness:** 11.6254, **kurtosis:** 241.4104
- **log_crime_count — skewness:** -0.0867, **kurtosis:** -0.2306

**Top 15 Pearson correlations with log_crime_count (numeric predictors):**

| Feature     | Correlation |
| ----------- | ----------- |
| a_hh        | 0.6650      |
| a_woning    | 0.6641      |
| a_bedv      | 0.6578      |
| a_man       | 0.6472      |
| a_inw       | 0.6462      |
| a_ongeh     | 0.6456      |
| a_vrouw     | 0.6440      |
| a_25_44     | 0.6422      |
| a_bed_oq    | 0.6402      |
| a_hh_z_k    | 0.6383      |
| a_1p_hh     | 0.6314      |
| a_gesch     | 0.6284      |
| a_bed_ru    | 0.6277      |
| ste_mvs     | -0.6207     |
| p_1gezw_hvw | -0.6167     |

## Section 2: Dataset sizes

### base
- **Total rows:** 14,494
- **Feature columns (excl. gwb_code_10, regio, gm_naam, log_crime_count, crime_count):** 97
- **Spatial lag feature columns (queen_lag_*, boundary_lag_*):** 0
- **Rows with ≥1 missing value:** 3,036
- **Overall missing value rate (% of all cells):** 3.1620%

### spatial
- **Total rows:** 14,494
- **Feature columns (excl. gwb_code_10, regio, gm_naam, log_crime_count, crime_count):** 109
- **Spatial lag feature columns (queen_lag_*, boundary_lag_*):** 12
- **Rows with ≥1 missing value:** 3,041
- **Overall missing value rate (% of all cells):** 2.8364%

### boundary
- **Total rows:** 14,494
- **Feature columns (excl. gwb_code_10, regio, gm_naam, log_crime_count, crime_count):** 103
- **Spatial lag feature columns (queen_lag_*, boundary_lag_*):** 6
- **Rows with ≥1 missing value:** 3,041
- **Overall missing value rate (% of all cells):** 2.9902%

## Section 3: Train/test split summary (base dataset)
- **Train rows:** 11,735
- **Test rows:** 2,759
- **Unique municipalities (train):** 273
- **Unique municipalities (test):** 68
- **Train/test split by rows:** 80.9645% train, 19.0355% test
- **Train/test split by municipalities:** 80.0587% train, 19.9413% test
- **Municipality sets non-overlapping:** TRUE

## Section 4: MVP cross-validated results
| Model | Feature set | Tuning status | Mean CV RMSE | Std CV RMSE | Mean CV MAE | Mean CV R² |
| ----- | ----------- | ------------- | ------------ | ----------- | ----------- | ---------- |
| XGB   | spatial     | untuned (MVP) | 0.5780       | 0.0082      | 0.4252      | 0.8267     |
| XGB   | boundary    | untuned (MVP) | 0.5809       | 0.0097      | 0.4254      | 0.8251     |
| XGB   | base        | untuned (MVP) | 0.5829       | 0.0120      | 0.4264      | 0.8240     |
| RF    | spatial     | untuned (MVP) | 0.5967       | 0.0079      | 0.4396      | 0.8155     |
| RF    | boundary    | untuned (MVP) | 0.6031       | 0.0110      | 0.4406      | 0.8116     |
| RF    | base        | untuned (MVP) | 0.6031       | 0.0118      | 0.4400      | 0.8117     |
| OLS   | spatial     | untuned (MVP) | 0.7041       | 0.0242      | 0.5268      | 0.7434     |
| OLS   | boundary    | untuned (MVP) | 0.7191       | 0.0265      | 0.5374      | 0.7323     |
| OLS   | base        | untuned (MVP) | 0.7248       | 0.0256      | 0.5402      | 0.7280     |

## Section 5: Hyperparameter tuning results

### RF — best hyperparameters (RandomizedSearchCV)

**base:**
```json
{
  "model__n_estimators": 500,
  "model__min_samples_split": 2,
  "model__min_samples_leaf": 2,
  "model__max_features": 0.5,
  "model__max_depth": 40,
  "model__bootstrap": true
}
```

**spatial:**
```json
{
  "model__n_estimators": 500,
  "model__min_samples_split": 2,
  "model__min_samples_leaf": 2,
  "model__max_features": 0.5,
  "model__max_depth": null,
  "model__bootstrap": true
}
```

**boundary:**
```json
{
  "model__n_estimators": 800,
  "model__min_samples_split": 2,
  "model__min_samples_leaf": 5,
  "model__max_features": 0.5,
  "model__max_depth": 20,
  "model__bootstrap": true
}
```

### XGB — best hyperparameters (RandomizedSearchCV)

**base:**
```json
{
  "model__subsample": 0.6,
  "model__reg_lambda": 0.5,
  "model__reg_alpha": 0.01,
  "model__n_estimators": 500,
  "model__min_child_weight": 3,
  "model__max_depth": 6,
  "model__learning_rate": 0.03,
  "model__colsample_bytree": 0.6
}
```

**spatial:**
```json
{
  "model__subsample": 0.6,
  "model__reg_lambda": 0.5,
  "model__reg_alpha": 0.01,
  "model__n_estimators": 500,
  "model__min_child_weight": 3,
  "model__max_depth": 6,
  "model__learning_rate": 0.03,
  "model__colsample_bytree": 0.6
}
```

**boundary:**
```json
{
  "model__subsample": 0.6,
  "model__reg_lambda": 2,
  "model__reg_alpha": 0.1,
  "model__n_estimators": 800,
  "model__min_child_weight": 5,
  "model__max_depth": 8,
  "model__learning_rate": 0.01,
  "model__colsample_bytree": 0.8
}
```

**ΔRMSE (RF, base):** MVP CV RMSE − tuned CV RMSE = 0.6031 − 0.5961 = **0.0070** (1.1684% improvement)

**ΔRMSE (RF, spatial):** MVP CV RMSE − tuned CV RMSE = 0.5967 − 0.5914 = **0.0053** (0.8875% improvement)

**ΔRMSE (RF, boundary):** MVP CV RMSE − tuned CV RMSE = 0.6031 − 0.5978 = **0.0053** (0.8811% improvement)

**ΔRMSE (XGB, base):** MVP CV RMSE − tuned CV RMSE = 0.5829 − 0.5771 = **0.0058** (0.9937% improvement)

**ΔRMSE (XGB, spatial):** MVP CV RMSE − tuned CV RMSE = 0.5780 − 0.5707 = **0.0073** (1.2713% improvement)

**ΔRMSE (XGB, boundary):** MVP CV RMSE − tuned CV RMSE = 0.5809 − 0.5764 = **0.0045** (0.7810% improvement)

**Summary (CV RMSE):**

| Model | Feature set | MVP RMSE | Tuned RMSE | ΔRMSE  | % improvement |
| ----- | ----------- | -------- | ---------- | ------ | ------------- |
| RF    | base        | 0.6031   | 0.5961     | 0.0070 | 1.1684        |
| RF    | spatial     | 0.5967   | 0.5914     | 0.0053 | 0.8875        |
| RF    | boundary    | 0.6031   | 0.5978     | 0.0053 | 0.8811        |
| XGB   | base        | 0.5829   | 0.5771     | 0.0058 | 0.9937        |
| XGB   | spatial     | 0.5780   | 0.5707     | 0.0073 | 1.2713        |
| XGB   | boundary    | 0.5809   | 0.5764     | 0.0045 | 0.7810        |

## Section 6: Final held-out test set results
| Model | Feature set | Tuning status | Test RMSE | Test MAE | Test R² |
| ----- | ----------- | ------------- | --------- | -------- | ------- |
| XGB   | base        | tuned         | 0.5546    | 0.4047   | 0.8392  |
| XGB   | spatial     | tuned         | 0.5579    | 0.4040   | 0.8373  |
| XGB   | boundary    | tuned         | 0.5594    | 0.4087   | 0.8364  |
| RF    | base        | tuned         | 0.5723    | 0.4178   | 0.8287  |
| RF    | boundary    | tuned         | 0.5774    | 0.4204   | 0.8257  |
| RF    | spatial     | tuned         | 0.5814    | 0.4200   | 0.8233  |
| OLS   | spatial     | baseline      | 0.6778    | 0.5138   | 0.7598  |
| OLS   | boundary    | baseline      | 0.6840    | 0.5179   | 0.7554  |
| OLS   | base        | baseline      | 0.6852    | 0.5160   | 0.7545  |

**Best configuration (lowest test RMSE):** XGB, base, tuned — RMSE 0.5546, MAE 0.4047, R² 0.8392
**Worst configuration:** OLS, base, baseline — RMSE 0.6852, MAE 0.5160, R² 0.7545
**RMSE gap (best OLS − best tree-based):** 0.6778 − 0.5546 = **0.1232**

### OLS — test RMSE by feature set
- **base:** 0.6852
- **spatial:** 0.6778 (Δ vs base: -0.0074)
- **boundary:** 0.6840 (Δ vs base: -0.0012)

### RF — test RMSE by feature set
- **base:** 0.5723
- **spatial:** 0.5814 (Δ vs base: 0.0090)
- **boundary:** 0.5774 (Δ vs base: 0.0051)

### XGB — test RMSE by feature set
- **base:** 0.5546
- **spatial:** 0.5579 (Δ vs base: 0.0032)
- **boundary:** 0.5594 (Δ vs base: 0.0047)

## Section 7: SHAP feature importance
*Computed mean |SHAP| on full **base** training split for **XGB** (tuned); saved to `outputs/writer_extraction_cache/shap_mean_abs_computed.csv`.*

**Full table (mean |SHAP|, descending):**

| Feature      | Mean |SHAP| |
| ------------ | ----------- |
| a_bedv       | 0.2527      |
| a_1p_hh      | 0.2460      |
| a_vastg      | 0.1477      |
| ste_oad      | 0.1410      |
| a_neu_al     | 0.1318      |
| a_bed_gi     | 0.1024      |
| g_3km_sc     | 0.0880      |
| a_lan_ha     | 0.0862      |
| a_gbl_ne     | 0.0855      |
| a_bed_ru     | 0.0849      |
| a_opp_ha     | 0.0653      |
| pst_mvp      | 0.0470      |
| a_eur_al     | 0.0441      |
| bev_dich     | 0.0326      |
| g_ele_tr     | 0.0319      |
| a_gbl_eu     | 0.0304      |
| g_wozbag     | 0.0253      |
| a_woning     | 0.0237      |
| g_afs_gs     | 0.0222      |
| g_afs_sc     | 0.0213      |
| a_bed_oq     | 0.0208      |
| p_koopw      | 0.0180      |
| a_gesch      | 0.0168      |
| ste_mvs      | 0.0164      |
| p_huurw      | 0.0157      |
| a_bed_a      | 0.0142      |
| a_m2w        | 0.0135      |
| p_1gezw      | 0.0134      |
| g_hhgro      | 0.0131      |
| g_ele        | 0.0121      |
| a_geb_ne     | 0.0117      |
| a_bed_mn     | 0.0117      |
| a_ongeh      | 0.0114      |
| p_1gezw_hvw  | 0.0110      |
| g_afs_hp     | 0.0108      |
| p_1gezw_2w   | 0.0098      |
| a_15_24      | 0.0097      |
| a_bed_bf     | 0.0096      |
| a_gehuwd     | 0.0095      |
| a_hh         | 0.0093      |
| p_ov_hw      | 0.0084      |
| a_wat_ha     | 0.0081      |
| a_bst_b      | 0.0079      |
| a_25_44      | 0.0078      |
| a_bed_hj     | 0.0078      |
| a_vrouw      | 0.0077      |
| a_man        | 0.0077      |
| p_leegsw     | 0.0077      |
| g_gas        | 0.0074      |
| g_afs_kv     | 0.0072      |
| a_bed_kl     | 0.0068      |
| a_pau        | 0.0067      |
| a_geb_eu     | 0.0067      |
| a_inw        | 0.0067      |
| a_nl_all     | 0.0064      |
| a_ll         | 0.0064      |
| a_bst_nb     | 0.0062      |
| pst_dekp     | 0.0061      |
| a_verwed     | 0.0060      |
| a_ons_hbo    | 0.0060      |
| a_65_oo      | 0.0059      |
| a_hh_m_k     | 0.0056      |
| a_nb_won     | 0.0055      |
| a_geb_nl     | 0.0055      |
| a_ons_mbo    | 0.0053      |
| a_ons_vovavo | 0.0053      |
| a_hh_z_k     | 0.0052      |
| a_45_64      | 0.0046      |
| p_geb        | 0.0044      |
| a_nb_vastg   | 0.0044      |
| p_1gezw_tw   | 0.0044      |
| p_ste        | 0.0043      |
| p_mgezw      | 0.0041      |
| p_wcorpw     | 0.0038      |
| p_1gezw_hw   | 0.0038      |
| a_00_14      | 0.0030      |
| p_bj_me10    | 0.0028      |
| a_ste        | 0.0026      |
| a_ons_wo     | 0.0023      |
| p_bj_mi10    | 0.0017      |
| a_ons_po     | 0.0017      |
| a_geb        | 0.0010      |
| ind_wbi      | 0.0007      |

**Spatial lag features only (rank = overall rank in full table):**

*No `queen_lag_*` or `boundary_lag_*` predictors in this feature matrix (e.g. **base** feature set only has non-spatial lags).*

**Combined mean |SHAP| — spatial lag vs base features:** 0.0000% lag, 100.0000% non-lag (of total mean |SHAP| mass).

## Section 8: Error and bias analysis
### overall_error_metrics.csv
- **RMSE:** 0.5546, **MAE:** 0.4047, **R²:** 0.8392
- **Mean residual (bias check):** -0.0083

### flagged_bias_error_patterns.csv

| issue                      | evidence                                         | interpretation                                                                       |
| -------------------------- | ------------------------------------------------ | ------------------------------------------------------------------------------------ |
| High-crime underprediction | mean residual in top target decile = 0.2882 (>0) | Model tends to underpredict high-intensity neighborhoods (systematic error pattern). |
| Low-population instability | low-pop MAE = 0.6830 vs overall MAE = 0.4047     | Model errors are substantially larger in small-population neighborhoods.             |

### subgroup_error_by_population_decile.csv

| Decile | n   | MAE    | Mean residual |
| ------ | --- | ------ | ------------- |
| 0      | 281 | 0.6830 | 0.0194        |
| 1      | 286 | 0.5655 | -0.0027       |
| 2      | 271 | 0.5563 | -0.0666       |
| 3      | 268 | 0.4662 | -0.0557       |
| 4      | 275 | 0.3716 | 0.0226        |
| 5      | 274 | 0.3607 | -0.0326       |
| 6      | 276 | 0.2926 | -0.0462       |
| 7      | 277 | 0.2973 | 0.0140        |
| 8      | 275 | 0.2243 | 0.0470        |
| 9      | 276 | 0.2226 | 0.0151        |

**MAE ratio (lowest population decile / highest population decile):** 0.6830 / 0.2226 = **3.0688**

### subgroup_error_by_target_decile.csv

| Decile | n   | MAE    | Mean residual |
| ------ | --- | ------ | ------------- |
| 0      | 293 | 0.6615 | -0.6125       |
| 1      | 264 | 0.4657 | -0.1637       |
| 2      | 313 | 0.4123 | -0.0799       |
| 3      | 245 | 0.4228 | -0.0585       |
| 4      | 285 | 0.3893 | 0.0718        |
| 5      | 255 | 0.3516 | 0.0922        |
| 6      | 290 | 0.3513 | 0.1532        |
| 7      | 269 | 0.2984 | 0.1123        |
| 8      | 271 | 0.3043 | 0.1535        |
| 9      | 274 | 0.3723 | 0.2882        |

**Mean residual in top target decile:** 0.2882

*Other files in ethics output folder (for reference):*
- `outputs/ethics_bias_error_analysis/ethics_checklist.csv`
- `outputs/ethics_bias_error_analysis/subgroup_error_by_bev_dich_decile.csv`
- `outputs/ethics_bias_error_analysis/subgroup_error_by_socioeconomic_proxy.csv`
- `outputs/ethics_bias_error_analysis/subgroup_error_by_ste_oad_decile.csv`

## Section 9: Figures needed (checklist)
- [ ] **1. Side-by-side histogram of raw crime_count and log_crime_count** — *not found as saved file; extract manually when needed*
- [ ] **2. Choropleth map (neighbourhood) shaded by log_crime_count** — *not found as saved file; extract manually when needed*
- [ ] **3. Horizontal bar chart — top 15 predictor correlations with log_crime_count** — *not found as saved file; extract manually when needed*
- [ ] **4. Grouped bar chart — test RMSE, 9 configurations by feature set** — *not found as saved file; extract manually when needed*
- [ ] **5. Grouped bar chart — test R², 9 configurations by feature set** — *not found as saved file; extract manually when needed*
- [x] **6. Scatter plot — predicted vs actual log_crime_count (best model, test)** — `outputs/ethics_bias_error_analysis/figures/pred_vs_actual.png`
- [x] **7. Bar chart — MAE by population decile with overall MAE reference** — `outputs/ethics_bias_error_analysis/figures/abs_error_by_population_decile.png`
- [x] **8. Bar chart — mean residual by target decile** — `outputs/ethics_bias_error_analysis/figures/mean_residual_by_target_decile.png`
- [ ] **9. SHAP summary beeswarm (best model)** — *not found as saved file; extract manually when needed*

## Section 10: Validation checks
- **Same municipalities in train splits (base/spatial/boundary):** TRUE
- **No municipality in both train and test (all feature sets):** TRUE
- **SHAP / best model alignment:** Best test configuration = **XGB**, feature set **base**, tuning **tuned** (SHAP table targets this configuration when CSV is absent and values are recomputed).
- **Test set predictions exist for all 9 configurations:** TRUE (found 9 unique configs)
- **Number of neighbourhoods in test set (unique gwb_code_10):** 2,759

---

## Extraction log
**Warnings:**
- Loaded MVP CV from cache: outputs/writer_extraction_cache/mvp_cv_summary.csv
- Loaded tuning results from cache: outputs/writer_extraction_cache/tuning_results.json
- No SHAP mean |SHAP| CSV in outputs/modeling; computing from best test configuration.

*No errors reported.*

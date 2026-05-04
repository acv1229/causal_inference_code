# External Dataset Validation: LaLonde IRM + Local R2 Diagnostics

## Dataset and Design

This pilot uses the standard LaLonde dataset as a non-IHDP covariate base. The cached file is `processed/lalonde.csv`, downloaded from a public CSV copy of the LaLonde data.

- 614 rows
- 8 pre-treatment covariates: `age`, `educ`, `black`, `hispan`, `married`, `nodegree`, `re74`, `re75`
- observed LaLonde `treat` and `re78` are not used in the validation
- treatment and outcomes are generated semi-synthetically
- true ATE = 4.0 by construction

The goal is not to estimate the empirical LaLonde treatment effect. The dataset is used as an external covariate geometry to test whether the IRM local-R2 diagnostic behavior observed on IHDP carries over to a different causal-inference benchmark dataset.

Command run:

```bash
python3 src/validate_external_dataset.py \
  --n-reps 5 \
  --n-folds 3 \
  --overlap-strengths 0 2 4 \
  --surfaces linear localized_nonlinear \
  --learners ridge lasso_logistic rf \
  --output output/external_validation/lalonde_irm_local_results.csv \
  --auc-output output/external_validation/lalonde_irm_local_auc.csv \
  --cell-output output/external_validation/lalonde_irm_local_cells.csv
```

This produced 90 IRM fits:

- 2 outcome surfaces
- 3 overlap levels
- 3 learners
- 5 replications

## Result

Diagnostic AUC for predicting whether the nominal 95% CI covers the true ATE:

| External covariate base | Surface | R_D alone | R_D + global R_Y0/R_Y1 | R_D + local R_Y0 | R_D + global R_Y0 + local R_Y0 | R_D + local R_Y0/R_Y1 |
|---|---:|---:|---:|---:|---:|---:|
| LaLonde | linear | 0.6858 | 0.7466 | 0.7850 | 0.7800 | 0.7650 |
| LaLonde | localized_nonlinear | 0.7300 | 0.7700 | 0.7429 | 0.7929 | 0.7786 |

Coverage by surface:

| Surface | Coverage |
|---|---:|
| linear | 0.8222 |
| localized_nonlinear | 0.8889 |

Coverage by surface, overlap, and learner:

| Surface | Overlap | Lasso+Logistic | RF | Ridge |
|---|---:|---:|---:|---:|
| linear | 0.0 | 1.0 | 1.0 | 1.0 |
| linear | 2.0 | 0.8 | 0.8 | 0.6 |
| linear | 4.0 | 1.0 | 0.2 | 1.0 |
| localized_nonlinear | 0.0 | 1.0 | 1.0 | 1.0 |
| localized_nonlinear | 2.0 | 1.0 | 0.8 | 1.0 |
| localized_nonlinear | 4.0 | 1.0 | 0.2 | 1.0 |

## Interpretation

The LaLonde pilot gives partial external support for the local diagnostic idea, but it is still a small validation:

- Linear surface: local R_Y0 improves over global arm R2, 0.7850 vs 0.7466.
- Localized nonlinear surface: global-plus-local R_Y0 performs best, 0.7929 vs 0.7700 for global arm R2 alone.
- Both-arm local R2 improves over global on the localized nonlinear surface, 0.7786 vs 0.7700, but is below control-side local alone on the linear surface.

The strongest signal again comes from the control-side local metric. Adding the treated-side local metric is not uniformly helpful, which supports the current cautionary framing: compute both arms, report local sample sizes, and inspect which arm actually carries signal before treating the diagnostic triple mechanically.

## Takeaway

This is a useful external validation on a canonical causal-inference dataset, but not yet a broad generality claim. It supports the paper's practical recommendation with a careful qualification:

- IRM makes outcome diagnostics interpretable on a non-IHDP benchmark covariate base.
- Local R2 can add predictive information beyond R_D and global arm R2.
- The most informative arm can be dataset- and DGP-dependent.
- Larger replication counts and additional external covariate bases are still needed before claiming universal external validation.

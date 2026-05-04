# Reproducing R_D Experiment Results

This doc points to the visualizations and gives commands to recreate the Monte Carlo runs behind the R_D diagnostic findings.

## Visuals (PLR, from v7 joint diagnostic run)

Location: `output/figures/joint/`

- `joint_2d_scatter.png` — (R_D, R_Y) colored by coverage; shows learner separation
- `joint_ry_marginal_value.png` — left panel: coverage vs R_D per learner (the headline plot); right panel: coverage vs R_Y at fixed R_D bands
- `joint_predictive_comparison.png` — AUC bar chart showing R_Y adds no predictive value beyond R_D
- `joint_coverage_heatmap.png` — coverage heatmap in (R_D, R_Y) bins
- `joint_decision_boundary.png` — logistic decision surface over (R_D, R_Y)

## Summary tables

Location: `output/tables/joint/`

- `joint_summary.csv` — per-cell bias, RMSE, coverage, R_D, R_Y
- `predictive_comparison.csv` — AUC comparison (R_D alone vs R_Y alone vs R_D + R_Y)

## Commands to recreate

All commands run from the `src/` directory.

### Primary structural PLR run (v3) — Lasso vs XGBoost

5,600 estimations (7 overlap strengths × 2 surfaces × 2 learners × 200 reps).

```bash
python3 monte_carlo.py --propensity-model structural \
  --overlap-strengths 0.0 0.5 1.0 1.5 2.0 3.0 5.0 \
  --complexities linear nonlinear \
  --learners lasso xgboost --n-reps 200 \
  --input ../processed/ihdp_sim_processed.csv \
  --output ../output/mc_results_v3_structural.csv
```

### Joint (R_D, R_Y) PLR run (v7) — adds lasso_logistic, records R_Y

8,400 estimations.

```bash
python3 monte_carlo.py --propensity-model structural \
  --overlap-strengths 0.0 0.5 1.0 1.5 2.0 3.0 5.0 \
  --complexities linear nonlinear \
  --learners lasso xgboost lasso_logistic --n-reps 200 \
  --input ../processed/ihdp_sim_processed.csv \
  --output ../output/mc_results_v7_structural_joint.csv
```

### Regenerate figures from v7 results

```bash
python3 analyze_joint_diagnostic.py \
  --input ../output/mc_results_v7_structural_joint.csv \
  --output-dir ../output
```

## IRM results (balanced grid, v8c)

IRM figures at `output/figures/joint_irm/` are generated from a balanced grid — all 4 learners at every overlap × surface cell (2,400 estimations).

### Command to reproduce the IRM run

```bash
python3 monte_carlo.py --framework irm --propensity-model structural \
  --overlap-strengths 0.0 2.0 5.0 \
  --complexities linear nonlinear \
  --learners lasso_logistic xgboost ridge rf --n-reps 100 \
  --input ../processed/ihdp_sim_processed.csv \
  --output ../output/mc_results_v8c_irm_balanced.csv
```

### Regenerate IRM figures

```bash
python3 -c "
import sys; sys.path.insert(0, '.')
from pathlib import Path
from analyze_joint_diagnostic import (
    load_results, build_summary, _setup_style,
    plot_2d_scatter, plot_coverage_heatmap, plot_ry_marginal_value,
    predictive_comparison, plot_decision_boundary, save_joint_summary,
)
fig_dir = Path('../output/figures/joint_irm')
table_dir = Path('../output/tables/joint_irm')
fig_dir.mkdir(parents=True, exist_ok=True)
table_dir.mkdir(parents=True, exist_ok=True)
df = load_results('../output/mc_results_v8c_irm_balanced.csv')
summary = build_summary(df)
_setup_style()
plot_2d_scatter(summary, fig_dir)
plot_coverage_heatmap(df, fig_dir)
plot_ry_marginal_value(df, fig_dir)
predictive_comparison(df, table_dir, fig_dir)
plot_decision_boundary(df, fig_dir)
save_joint_summary(summary, table_dir)
"
```

### Key IRM AUC numbers (from v8c balanced grid)

| Surface | R_D alone | R_Y0 alone | R_D + R_Y0 |
|---------|-----------|-------------|-------------|
| Linear | 0.804 | 0.588 | 0.826 |
| Nonlinear | 0.734 | 0.608 | 0.767 |

Compare to PLR on the same DGP (v7): R_D alone = 0.566 / 0.649, R_D + R_Y = 0.564 / 0.645. IRM makes both R_D and R_Y0 substantially more useful.

## Project context

- Full findings: [findings.md](findings.md)
- Implementation log: [implementation_status.md](implementation_status.md)
- Project overview: [README.md](README.md)

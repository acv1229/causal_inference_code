"""
Deep dive: can we find a sharp R_D threshold from the flexible learner alone?
Focus on XGBoost results, especially nonlinear surface where real degradation occurs.
"""

import pandas as pd
import numpy as np

df = pd.read_csv("output/archive/mc_results_v3_structural.csv")
TRUE_ATE = 4.0

# ── 1. XGBoost only, split by surface ────────────────────────────────────
xgb = df[df["learner"] == "xgboost"].copy()
xgb["bias"] = xgb["estimate"] - TRUE_ATE
xgb["abs_bias"] = xgb["bias"].abs()
xgb["ci_width"] = xgb["ci_upper"] - xgb["ci_lower"]
xgb["bias_over_se"] = xgb["bias"] / xgb["se"]  # t-stat-like: how many SEs off?

print("=" * 70)
print("XGBoost: COVERAGE & BIAS BY R_D DECILES, split by surface")
print("=" * 70)

for surface in ["linear", "nonlinear"]:
    sub = xgb[xgb["complexity"] == surface].copy()
    sub["r_d_bin"] = pd.qcut(sub["r_d"], q=10, duplicates="drop")
    grouped = sub.groupby("r_d_bin", observed=True).agg(
        n=("covers", "count"),
        coverage=("covers", "mean"),
        mean_bias=("bias", "mean"),
        mean_abs_bias=("abs_bias", "mean"),
        median_abs_bias=("abs_bias", "median"),
        rmse=("bias", lambda x: np.sqrt((x**2).mean())),
        mean_se=("se", "mean"),
        mean_bias_over_se=("bias_over_se", lambda x: x.mean()),
        mean_ci_width=("ci_width", "mean"),
    ).round(3)
    print(f"\n{surface.upper()} surface:")
    print(grouped.to_string())

# ── 2. Finer threshold search for XGBoost nonlinear ──────────────────────
print("\n" + "=" * 70)
print("XGBoost NONLINEAR: fine-grained threshold search")
print("=" * 70)

xgb_nl = xgb[xgb["complexity"] == "nonlinear"].copy()

print(f"\n  {'R_D thresh':>10} | {'N below':>7} | {'Cov below':>9} | {'RMSE below':>10} | "
      f"{'|Bias| below':>11} | {'N above':>7} | {'Cov above':>9} | {'RMSE above':>10}")
print(f"  {'-'*10} | {'-'*7} | {'-'*9} | {'-'*10} | {'-'*11} | {'-'*7} | {'-'*9} | {'-'*10}")

for thresh in np.arange(0.15, 1.05, 0.05):
    below = xgb_nl[xgb_nl["r_d"] < thresh]
    above = xgb_nl[xgb_nl["r_d"] >= thresh]
    if len(below) >= 20 and len(above) >= 20:
        cov_b = below["covers"].mean()
        cov_a = above["covers"].mean()
        rmse_b = np.sqrt((below["bias"]**2).mean())
        rmse_a = np.sqrt((above["bias"]**2).mean())
        abias_b = below["abs_bias"].mean()
        abias_a = above["abs_bias"].mean()
        print(f"  {thresh:>10.2f} | {len(below):>7} | {cov_b:>9.3f} | {rmse_b:>10.3f} | "
              f"{abias_b:>11.3f} | {len(above):>7} | {cov_a:>9.3f} | {rmse_a:>10.3f}")

# ── 3. Same for XGBoost linear (the "easy" surface) ──────────────────────
print("\n" + "=" * 70)
print("XGBoost LINEAR: fine-grained threshold search")
print("=" * 70)

xgb_lin = xgb[xgb["complexity"] == "linear"].copy()

print(f"\n  {'R_D thresh':>10} | {'N below':>7} | {'Cov below':>9} | {'RMSE below':>10} | "
      f"{'|Bias| below':>11} | {'N above':>7} | {'Cov above':>9} | {'RMSE above':>10}")
print(f"  {'-'*10} | {'-'*7} | {'-'*9} | {'-'*10} | {'-'*11} | {'-'*7} | {'-'*9} | {'-'*10}")

for thresh in np.arange(0.15, 1.05, 0.05):
    below = xgb_lin[xgb_lin["r_d"] < thresh]
    above = xgb_lin[xgb_lin["r_d"] >= thresh]
    if len(below) >= 20 and len(above) >= 20:
        cov_b = below["covers"].mean()
        cov_a = above["covers"].mean()
        rmse_b = np.sqrt((below["bias"]**2).mean())
        rmse_a = np.sqrt((above["bias"]**2).mean())
        abias_b = below["abs_bias"].mean()
        abias_a = above["abs_bias"].mean()
        print(f"  {thresh:>10.2f} | {len(below):>7} | {cov_b:>9.3f} | {rmse_b:>10.3f} | "
              f"{abias_b:>11.3f} | {len(above):>7} | {cov_a:>9.3f} | {rmse_a:>10.3f}")

# ── 4. Ratio of RMSE to mean SE (calibration check) ──────────────────────
# If RMSE >> SE, inference is overconfident
print("\n" + "=" * 70)
print("SE CALIBRATION: RMSE / mean_SE by overlap strength")
print("=" * 70)
print("(If ratio >> 1, SEs are too small → CI undercoverage)\n")

for surface in ["linear", "nonlinear"]:
    sub = xgb[xgb["complexity"] == surface]
    print(f"{surface.upper()}:")
    for strength in sorted(sub["overlap_strength"].unique()):
        s = sub[sub["overlap_strength"] == strength]
        rmse = np.sqrt((s["bias"]**2).mean())
        mean_se = s["se"].mean()
        ratio = rmse / mean_se
        print(f"  strength={strength:.1f}: RMSE={rmse:.3f}, mean_SE={mean_se:.3f}, "
              f"ratio={ratio:.2f}, R_D={s['r_d'].mean():.3f}, cov={s['covers'].mean():.3f}")
    print()

# ── 5. What if we frame it as: R_D predicts RMSE/SE ratio? ───────────────
print("=" * 70)
print("R_D vs RMSE/SE RATIO (pooled across surfaces)")
print("=" * 70)
print("(This is the 'miscalibration' view — when does DML become overconfident?)\n")

# Bin by R_D and compute group-level RMSE/SE
bins = [0, 0.15, 0.25, 0.35, 0.50, 0.70, 0.90, 1.5]
xgb["r_d_coarse"] = pd.cut(xgb["r_d"], bins=bins)
for r_d_bin, group in xgb.groupby("r_d_coarse", observed=True):
    rmse = np.sqrt((group["bias"]**2).mean())
    mean_se = group["se"].mean()
    ratio = rmse / mean_se
    cov = group["covers"].mean()
    print(f"  R_D in {str(r_d_bin):>15}: n={len(group):>4}, RMSE/SE={ratio:.2f}, "
          f"coverage={cov:.3f}, mean_|bias|={group['abs_bias'].mean():.3f}")

# ── 6. What about the PROPORTION of variance explained? ───────────────────
# R_D = Var(D_tilde)/Var(D). So 1-R_D = proportion of treatment variance
# explained by covariates. This might be more intuitive.
print("\n" + "=" * 70)
print("REFRAMING: 1-R_D = 'propensity predictability' (proportion explained)")
print("=" * 70)

summary = xgb.groupby(["overlap_strength", "complexity"]).agg(
    mean_r_d=("r_d", "mean"),
    coverage=("covers", "mean"),
).reset_index()
summary["predictability"] = 1 - summary["mean_r_d"]
summary = summary.round(3)
print(summary[["overlap_strength", "complexity", "mean_r_d", "predictability", "coverage"]].to_string(index=False))

"""
Robustness test: does R_D < 0.7 (from a flexible learner) reliably predict
DML coverage failure across multiple DGPs?

Pools results from:
- v2 logistic propensity (4 linear covariates)
- v3 structural propensity (race × bw interaction)
- v4 highdim propensity (6 covariates + 3 interactions)
- v5 threshold propensity (step function in bw × marriage)
"""

import pandas as pd
import numpy as np
from pathlib import Path

TRUE_ATE = 4.0
THRESHOLD = 0.70

# ── Load all results ──────────────────────────────────────────────────────
datasets = {
    "logistic": "output/archive/mc_results_v2_clip001_overlap0-10.csv",
    "structural": "output/archive/mc_results_v3_structural.csv",
    "highdim": "output/mc_results_v4_highdim.csv",
    "threshold": "output/mc_results_v5_threshold.csv",
}

frames = []
for dgp_name, path in datasets.items():
    p = Path(path)
    if not p.exists():
        print(f"  SKIPPING {dgp_name}: {path} not found yet")
        continue
    df = pd.read_csv(p)
    df["dgp"] = dgp_name
    df["bias"] = df["estimate"] - TRUE_ATE
    df["abs_bias"] = df["bias"].abs()
    frames.append(df)
    print(f"  Loaded {dgp_name}: {len(df)} rows")

if not frames:
    print("No data found!")
    exit(1)

all_data = pd.concat(frames, ignore_index=True)
print(f"\nTotal pooled: {len(all_data)} rows across {all_data['dgp'].nunique()} DGPs")

# ══════════════════════════════════════════════════════════════════════════
# ANALYSIS 1: The R_D < 0.7 rule, per DGP, XGBoost only
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 75)
print(f"TEST 1: R_D < {THRESHOLD} rule — XGBoost only, by DGP")
print("=" * 75)

xgb = all_data[all_data["learner"] == "xgboost"].copy()

print(f"\n  {'DGP':<15} | {'N<thr':>6} | {'Cov<thr':>8} | {'RMSE<thr':>9} | "
      f"{'N>=thr':>6} | {'Cov>=thr':>8} | {'RMSE>=thr':>9} | {'Separation':>10}")
print(f"  {'-'*15} | {'-'*6} | {'-'*8} | {'-'*9} | {'-'*6} | {'-'*8} | {'-'*9} | {'-'*10}")

for dgp in sorted(xgb["dgp"].unique()):
    sub = xgb[xgb["dgp"] == dgp]
    below = sub[sub["r_d"] < THRESHOLD]
    above = sub[sub["r_d"] >= THRESHOLD]
    if len(below) > 0 and len(above) > 0:
        cov_b = below["covers"].mean()
        cov_a = above["covers"].mean()
        rmse_b = np.sqrt((below["bias"]**2).mean())
        rmse_a = np.sqrt((above["bias"]**2).mean())
        sep = cov_a - cov_b
        print(f"  {dgp:<15} | {len(below):>6} | {cov_b:>8.3f} | {rmse_b:>9.3f} | "
              f"{len(above):>6} | {cov_a:>8.3f} | {rmse_a:>9.3f} | {sep:>+10.3f}")
    else:
        n_b = len(below)
        n_a = len(above)
        print(f"  {dgp:<15} | {n_b:>6} | {'N/A':>8} | {'N/A':>9} | "
              f"{n_a:>6} | {'N/A':>8} | {'N/A':>9} | {'N/A':>10}")

# ══════════════════════════════════════════════════════════════════════════
# ANALYSIS 2: Same split, but by DGP × surface
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 75)
print(f"TEST 2: R_D < {THRESHOLD} rule — XGBoost, by DGP × surface")
print("=" * 75)

print(f"\n  {'DGP':<15} {'Surface':<12} | {'Cov<thr':>8} {'(n)':>5} | {'Cov>=thr':>8} {'(n)':>5} | {'Gap':>7}")
print(f"  {'-'*15} {'-'*12} | {'-'*8} {'-'*5} | {'-'*8} {'-'*5} | {'-'*7}")

for dgp in sorted(xgb["dgp"].unique()):
    for surface in ["linear", "nonlinear"]:
        sub = xgb[(xgb["dgp"] == dgp) & (xgb["complexity"] == surface)]
        below = sub[sub["r_d"] < THRESHOLD]
        above = sub[sub["r_d"] >= THRESHOLD]
        if len(below) > 0 and len(above) > 0:
            cov_b = below["covers"].mean()
            cov_a = above["covers"].mean()
            print(f"  {dgp:<15} {surface:<12} | {cov_b:>8.3f} {len(below):>5} | "
                  f"{cov_a:>8.3f} {len(above):>5} | {cov_a-cov_b:>+7.3f}")
        else:
            print(f"  {dgp:<15} {surface:<12} | {'--':>8} {len(below):>5} | "
                  f"{'--':>8} {len(above):>5} | {'--':>7}")

# ══════════════════════════════════════════════════════════════════════════
# ANALYSIS 3: Fine-grained threshold search across ALL DGPs pooled
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 75)
print("TEST 3: Threshold search — XGBoost, pooled across all DGPs")
print("=" * 75)

print(f"\n  {'Threshold':>9} | {'N below':>7} | {'Cov below':>9} | {'N above':>7} | {'Cov above':>9} | {'Sep':>7}")
print(f"  {'-'*9} | {'-'*7} | {'-'*9} | {'-'*7} | {'-'*9} | {'-'*7}")

for thresh in np.arange(0.30, 1.01, 0.05):
    below = xgb[xgb["r_d"] < thresh]
    above = xgb[xgb["r_d"] >= thresh]
    if len(below) >= 20 and len(above) >= 20:
        cov_b = below["covers"].mean()
        cov_a = above["covers"].mean()
        print(f"  {thresh:>9.2f} | {len(below):>7} | {cov_b:>9.3f} | {len(above):>7} | {cov_a:>9.3f} | {cov_a-cov_b:>+7.3f}")

# ══════════════════════════════════════════════════════════════════════════
# ANALYSIS 4: Per-DGP threshold search (find optimal per DGP)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 75)
print("TEST 4: Optimal threshold per DGP (max coverage separation)")
print("=" * 75)

for dgp in sorted(xgb["dgp"].unique()):
    sub = xgb[xgb["dgp"] == dgp]
    best_thresh, best_sep = None, -999
    for thresh in np.arange(0.30, 1.01, 0.05):
        below = sub[sub["r_d"] < thresh]
        above = sub[sub["r_d"] >= thresh]
        if len(below) >= 20 and len(above) >= 20:
            sep = above["covers"].mean() - below["covers"].mean()
            if sep > best_sep:
                best_sep = sep
                best_thresh = thresh
    if best_thresh is not None:
        below = sub[sub["r_d"] < best_thresh]
        above = sub[sub["r_d"] >= best_thresh]
        print(f"  {dgp:<15}: optimal R_D = {best_thresh:.2f} "
              f"(cov_below={below['covers'].mean():.3f}, cov_above={above['covers'].mean():.3f}, "
              f"sep={best_sep:+.3f})")

# ══════════════════════════════════════════════════════════════════════════
# ANALYSIS 5: R_D by overlap strength for each new DGP (sanity check)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 75)
print("SANITY CHECK: Mean R_D and coverage by DGP × overlap strength (XGBoost)")
print("=" * 75)

summary = xgb.groupby(["dgp", "overlap_strength"]).agg(
    mean_r_d=("r_d", "mean"),
    coverage=("covers", "mean"),
    mean_bias=("bias", "mean"),
    n=("covers", "count"),
).round(3)
print(summary.to_string())

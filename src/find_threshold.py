"""
Find a simple R_D threshold rule for when DML inference is unreliable.
Analyzes rep-level MC results to identify actionable cutoffs.
"""

import pandas as pd
import numpy as np

df = pd.read_csv("output/archive/mc_results_v3_structural.csv")

print(f"Total obs: {len(df)}")
print(f"Learners: {df['learner'].unique()}")
print(f"Overlap strengths: {sorted(df['overlap_strength'].unique())}")
print()

# ── 1. Coverage by R_D bins, within each learner ──────────────────────────
print("=" * 70)
print("COVERAGE BY R_D BINS (rep-level)")
print("=" * 70)

for learner in ["lasso", "xgboost"]:
    sub = df[df["learner"] == learner].copy()
    # Create R_D bins
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5]
    sub["r_d_bin"] = pd.cut(sub["r_d"], bins=bins)
    grouped = sub.groupby("r_d_bin", observed=True).agg(
        n=("covers", "count"),
        coverage=("covers", "mean"),
        mean_bias=("estimate", lambda x: (x - 4.0).mean()),
        mean_abs_bias=("estimate", lambda x: (x - 4.0).abs().mean()),
    ).round(3)
    print(f"\n{learner.upper()}:")
    print(grouped.to_string())

# ── 2. For each learner, find R_D threshold where coverage drops below 90% ──
print("\n" + "=" * 70)
print("THRESHOLD SEARCH: R_D value where coverage < 90%")
print("=" * 70)

for learner in ["lasso", "xgboost"]:
    sub = df[df["learner"] == learner].copy()
    # Sort by R_D and compute rolling coverage
    sub = sub.sort_values("r_d")
    # Try thresholds from 0.1 to 1.0
    print(f"\n{learner.upper()}:")
    print(f"  {'Threshold':>10} | {'N below':>8} | {'Cov below':>10} | {'N above':>8} | {'Cov above':>10}")
    print(f"  {'-'*10} | {'-'*8} | {'-'*10} | {'-'*8} | {'-'*10}")
    for thresh in np.arange(0.1, 1.05, 0.05):
        below = sub[sub["r_d"] < thresh]
        above = sub[sub["r_d"] >= thresh]
        if len(below) > 0 and len(above) > 0:
            cov_below = below["covers"].mean()
            cov_above = above["covers"].mean()
            print(f"  {thresh:>10.2f} | {len(below):>8} | {cov_below:>10.3f} | {len(above):>8} | {cov_above:>10.3f}")

# ── 3. The R_D gap diagnostic ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("R_D GAP DIAGNOSTIC (lasso R_D vs xgboost R_D)")
print("=" * 70)

# Merge lasso and xgboost by (overlap_strength, complexity, seed)
lasso = df[df["learner"] == "lasso"][["overlap_strength", "complexity", "seed", "r_d", "covers", "estimate"]].copy()
xgb = df[df["learner"] == "xgboost"][["overlap_strength", "complexity", "seed", "r_d", "covers", "estimate"]].copy()
merged = lasso.merge(xgb, on=["overlap_strength", "complexity", "seed"], suffixes=("_lasso", "_xgb"))
merged["r_d_gap"] = merged["r_d_lasso"] - merged["r_d_xgb"]
merged["r_d_ratio"] = merged["r_d_lasso"] / merged["r_d_xgb"]

print("\nMean R_D gap by overlap strength:")
gap_summary = merged.groupby(["overlap_strength", "complexity"]).agg(
    mean_gap=("r_d_gap", "mean"),
    mean_ratio=("r_d_ratio", "mean"),
    lasso_cov=("covers_lasso", "mean"),
    xgb_cov=("covers_xgb", "mean"),
    lasso_rd=("r_d_lasso", "mean"),
    xgb_rd=("r_d_xgb", "mean"),
).round(3)
print(gap_summary.to_string())

# ── 4. Combined rule: R_D threshold + gap ─────────────────────────────────
print("\n" + "=" * 70)
print("EVALUATING CANDIDATE RULES")
print("=" * 70)

# For each rep, define "unreliable" as covers=0
# Test various rules on the LASSO results (the vulnerable learner)
lasso_df = df[df["learner"] == "lasso"].copy()
lasso_merged = lasso_df.merge(
    df[df["learner"] == "xgboost"][["overlap_strength", "complexity", "seed", "r_d"]].rename(columns={"r_d": "r_d_xgb"}),
    on=["overlap_strength", "complexity", "seed"]
)
lasso_merged["r_d_gap"] = lasso_merged["r_d"] - lasso_merged["r_d_xgb"]

print("\nRules applied to LASSO reps (the vulnerable learner):")
print(f"  Base rate: {lasso_merged['covers'].mean():.3f} coverage across all reps\n")

rules = {
    "R_D(own) < 0.80": lasso_merged["r_d"] < 0.80,
    "R_D(own) < 0.70": lasso_merged["r_d"] < 0.70,
    "R_D(own) < 0.60": lasso_merged["r_d"] < 0.60,
    "R_D(xgb) < 0.50": lasso_merged["r_d_xgb"] < 0.50,
    "R_D(xgb) < 0.30": lasso_merged["r_d_xgb"] < 0.30,
    "Gap > 0.10": lasso_merged["r_d_gap"] > 0.10,
    "Gap > 0.15": lasso_merged["r_d_gap"] > 0.15,
    "Gap > 0.20": lasso_merged["r_d_gap"] > 0.20,
    "R_D(own)<0.80 OR Gap>0.10": (lasso_merged["r_d"] < 0.80) | (lasso_merged["r_d_gap"] > 0.10),
    "R_D(own)<0.80 AND Gap>0.10": (lasso_merged["r_d"] < 0.80) & (lasso_merged["r_d_gap"] > 0.10),
}

print(f"  {'Rule':<35} | {'Flagged':>7} | {'Cov if flagged':>14} | {'Cov if safe':>11} | {'True pos rate':>13}")
print(f"  {'-'*35} | {'-'*7} | {'-'*14} | {'-'*11} | {'-'*13}")

for name, mask in rules.items():
    n_flagged = mask.sum()
    if n_flagged > 0 and (~mask).sum() > 0:
        cov_flagged = lasso_merged.loc[mask, "covers"].mean()
        cov_safe = lasso_merged.loc[~mask, "covers"].mean()
        # True positive rate: of reps that actually failed (covers=0), how many did we flag?
        failures = lasso_merged["covers"] == 0
        tpr = mask[failures].mean() if failures.sum() > 0 else float("nan")
        print(f"  {name:<35} | {n_flagged:>7} | {cov_flagged:>14.3f} | {cov_safe:>11.3f} | {tpr:>13.3f}")

# ── 5. Same analysis for XGBoost ──────────────────────────────────────────
print("\n\nRules applied to XGBOOST reps:")
xgb_df = df[df["learner"] == "xgboost"].copy()
print(f"  Base rate: {xgb_df['covers'].mean():.3f} coverage across all reps\n")

xgb_rules = {
    "R_D(own) < 0.50": xgb_df["r_d"] < 0.50,
    "R_D(own) < 0.40": xgb_df["r_d"] < 0.40,
    "R_D(own) < 0.30": xgb_df["r_d"] < 0.30,
    "R_D(own) < 0.20": xgb_df["r_d"] < 0.20,
}

print(f"  {'Rule':<35} | {'Flagged':>7} | {'Cov if flagged':>14} | {'Cov if safe':>11}")
print(f"  {'-'*35} | {'-'*7} | {'-'*14} | {'-'*11}")

for name, mask in xgb_rules.items():
    n_flagged = mask.sum()
    if n_flagged > 0 and (~mask).sum() > 0:
        cov_flagged = xgb_df.loc[mask, "covers"].mean()
        cov_safe = xgb_df.loc[~mask, "covers"].mean()
        print(f"  {name:<35} | {n_flagged:>7} | {cov_flagged:>14.3f} | {cov_safe:>11.3f}")

print("\n" + "=" * 70)
print("BOTTOM LINE")
print("=" * 70)
print("""
The question: can we find a single R_D threshold below which DML is unreliable?

The answer depends on whether we trust the propensity model:

1. IF propensity model is flexible (XGBoost-class):
   R_D directly reflects true overlap. Look at coverage vs R_D for XGBoost above.

2. IF propensity model may be misspecified:
   R_D from that model is UNRELIABLE. The gap between learners is the signal.

Practical recommendation: run DML with both a linear and a flexible learner.
If their R_D values diverge, the linear model is misspecified — trust the
flexible learner's R_D and estimates.
""")

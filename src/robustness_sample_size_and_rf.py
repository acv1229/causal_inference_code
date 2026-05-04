"""
Robustness checks:
1. Sample size: does the anti-conservative inflation persist at n=500 and n=2000?
2. Alternative flexible learner: does Random Forest show the same pattern as XGBoost?

Runs on the structural DGP (the clearest anti-conservative case).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from generate_ihdp_synthetic import generate_dataset
from dml_simulation import estimate_ate


def run_grid(base_df, sample_sizes, learners, overlap_strengths, complexities, n_reps):
    results = []
    total = len(sample_sizes) * len(overlap_strengths) * len(complexities) * len(learners) * n_reps
    done = 0

    for n in sample_sizes:
        for alpha in overlap_strengths:
            for complexity in complexities:
                for seed in range(1, n_reps + 1):
                    synth = generate_dataset(
                        base_df, sample_size=n, overlap_strength=alpha,
                        complexity=complexity, seed=seed,
                        propensity_model="structural",
                    )
                    for learner in learners:
                        t0 = time.time()
                        result = estimate_ate(synth, learner=learner)
                        elapsed = time.time() - t0
                        result["elapsed"] = elapsed
                        result["sample_size"] = n
                        results.append(result)
                        done += 1
                        if done % 50 == 0:
                            print(f"  [{done}/{total}] n={n}, alpha={alpha}, {complexity}, {learner}")

    return pd.DataFrame(results)


def main():
    base_df = pd.read_csv("../processed/ihdp_sim_processed.csv")

    # ── Check 1: Sample size robustness ───────────────────────────────────
    print("=" * 70)
    print("CHECK 1: Sample size robustness (lasso vs xgboost, structural DGP)")
    print("=" * 70)

    sample_sizes = [500, 985, 2000]
    overlap_strengths = [0.0, 2.0, 5.0]
    n_reps = 100

    total = len(sample_sizes) * len(overlap_strengths) * 2 * 2 * n_reps
    print(f"Running {total} estimations...")

    df_size = run_grid(
        base_df,
        sample_sizes=sample_sizes,
        learners=["lasso", "xgboost"],
        overlap_strengths=overlap_strengths,
        complexities=["linear", "nonlinear"],
        n_reps=n_reps,
    )

    df_size.to_csv("../output/robustness_sample_size.csv", index=False)

    print("\nR_D gap (Lasso - XGBoost) by sample size (nonlinear surface):")
    print(f"  {'n':>5} | {'strength':>8} | {'Lasso R_D':>10} | {'XGB R_D':>8} | {'Gap':>8} | {'Lasso Cov':>9} | {'XGB Cov':>7}")
    print(f"  {'-'*5} | {'-'*8} | {'-'*10} | {'-'*8} | {'-'*8} | {'-'*9} | {'-'*7}")
    for n in sample_sizes:
        for alpha in overlap_strengths:
            lasso = df_size[(df_size["sample_size"]==n) & (df_size["overlap_strength"]==alpha) &
                            (df_size["complexity"]=="nonlinear") & (df_size["learner"]=="lasso")]
            xgb = df_size[(df_size["sample_size"]==n) & (df_size["overlap_strength"]==alpha) &
                          (df_size["complexity"]=="nonlinear") & (df_size["learner"]=="xgboost")]
            if len(lasso) > 0 and len(xgb) > 0:
                l_rd = lasso["r_d"].mean()
                x_rd = xgb["r_d"].mean()
                print(f"  {n:>5} | {alpha:>8.1f} | {l_rd:>10.3f} | {x_rd:>8.3f} | {l_rd-x_rd:>+8.3f} | "
                      f"{lasso['covers'].mean():>9.3f} | {xgb['covers'].mean():>7.3f}")

    # ── Check 2: Random Forest as alternative flexible learner ────────────
    print("\n" + "=" * 70)
    print("CHECK 2: Random Forest vs XGBoost vs Lasso (n=985, structural DGP)")
    print("=" * 70)

    total_rf = len(overlap_strengths) * 2 * 3 * n_reps
    print(f"Running {total_rf} estimations...")

    df_rf = run_grid(
        base_df,
        sample_sizes=[985],
        learners=["lasso", "xgboost", "rf"],
        overlap_strengths=overlap_strengths,
        complexities=["linear", "nonlinear"],
        n_reps=n_reps,
    )

    df_rf.to_csv("../output/robustness_rf.csv", index=False)

    print("\nAll three learners (nonlinear surface):")
    print(f"  {'strength':>8} | {'learner':>8} | {'R_D':>6} | {'|Bias|':>7} | {'Cov':>5}")
    print(f"  {'-'*8} | {'-'*8} | {'-'*6} | {'-'*7} | {'-'*5}")
    for alpha in overlap_strengths:
        for learner in ["lasso", "rf", "xgboost"]:
            sub = df_rf[(df_rf["overlap_strength"] == alpha) &
                        (df_rf["complexity"] == "nonlinear") &
                        (df_rf["learner"] == learner)]
            if len(sub) > 0:
                print(f"  {alpha:>8.1f} | {learner:>8} | {sub['r_d'].mean():>6.3f} | "
                      f"{(sub['estimate']-4.0).abs().mean():>7.3f} | {sub['covers'].mean():>5.3f}")

    print("\nR_D gap: Lasso minus each flexible learner (nonlinear, strength=5.0):")
    for flex in ["xgboost", "rf"]:
        lasso = df_rf[(df_rf["overlap_strength"]==5.0) & (df_rf["complexity"]=="nonlinear") & (df_rf["learner"]=="lasso")]
        flex_df = df_rf[(df_rf["overlap_strength"]==5.0) & (df_rf["complexity"]=="nonlinear") & (df_rf["learner"]==flex)]
        if len(lasso) > 0 and len(flex_df) > 0:
            gap = lasso["r_d"].mean() - flex_df["r_d"].mean()
            print(f"  Lasso R_D - {flex:>7} R_D = {lasso['r_d'].mean():.3f} - {flex_df['r_d'].mean():.3f} = {gap:+.3f}")

    # ── Check 3: Isolate propensity misspecification ──────────────────────
    print("\n" + "=" * 70)
    print("CHECK 3: Lasso+Lasso vs Lasso+Logistic (same outcome, different propensity)")
    print("=" * 70)

    total_iso = len(overlap_strengths) * 2 * 2 * n_reps
    print(f"Running {total_iso} estimations...")

    df_iso = run_grid(
        base_df,
        sample_sizes=[985],
        learners=["lasso", "lasso_logistic"],
        overlap_strengths=overlap_strengths,
        complexities=["linear", "nonlinear"],
        n_reps=n_reps,
    )

    df_iso.to_csv("../output/robustness_propensity_isolation.csv", index=False)

    print("\nLasso vs Lasso+Logistic (nonlinear surface):")
    print(f"  {'strength':>8} | {'learner':>15} | {'R_D':>6} | {'|Bias|':>7} | {'Cov':>5}")
    print(f"  {'-'*8} | {'-'*15} | {'-'*6} | {'-'*7} | {'-'*5}")
    for alpha in overlap_strengths:
        for learner in ["lasso", "lasso_logistic"]:
            sub = df_iso[(df_iso["overlap_strength"] == alpha) &
                         (df_iso["complexity"] == "nonlinear") &
                         (df_iso["learner"] == learner)]
            if len(sub) > 0:
                print(f"  {alpha:>8.1f} | {learner:>15} | {sub['r_d'].mean():>6.3f} | "
                      f"{(sub['estimate']-4.0).abs().mean():>7.3f} | {sub['covers'].mean():>5.3f}")

    print("\nR_D gap at strength=5.0 (nonlinear):")
    for surface in ["linear", "nonlinear"]:
        l = df_iso[(df_iso["overlap_strength"]==5.0) & (df_iso["complexity"]==surface) & (df_iso["learner"]=="lasso")]
        ll = df_iso[(df_iso["overlap_strength"]==5.0) & (df_iso["complexity"]==surface) & (df_iso["learner"]=="lasso_logistic")]
        if len(l) > 0 and len(ll) > 0:
            print(f"  {surface:>10}: Lasso R_D={l['r_d'].mean():.3f} (cov={l['covers'].mean():.3f})  "
                  f"Lasso+Logistic R_D={ll['r_d'].mean():.3f} (cov={ll['covers'].mean():.3f})  "
                  f"Gap={l['r_d'].mean() - ll['r_d'].mean():+.3f}")


if __name__ == "__main__":
    main()

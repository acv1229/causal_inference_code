"""
Empirical verification of the anti-conservative decomposition:

    R_D = R_D* + Var(delta) / Var(D)

where R_D* = Var(V)/Var(D) is the oracle diagnostic and
delta(X) = m_0(X) - m_hat(X) is the misspecification error.

Regenerates datasets from known seeds to access true propensity m_0(X),
runs DML to get m_hat(X), and verifies the decomposition holds.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import doubleml as dml
from sklearn.linear_model import LassoCV
from xgboost import XGBClassifier, XGBRegressor

from generate_ihdp_synthetic import generate_dataset
from clean_ihdp import SIM_COVARIATE_COLUMNS


def estimate_with_decomposition(
    df: pd.DataFrame,
    learner: str,
    n_folds: int = 5,
    true_ate: float = 4.0,
) -> dict:
    """Run DML and return full decomposition components."""
    covariate_cols = [c for c in SIM_COVARIATE_COLUMNS if c in df.columns and c != "treat"]

    dml_data = dml.DoubleMLData(df, y_col="y", d_cols="sim_treat", x_cols=covariate_cols)

    if learner == "lasso":
        ml_l = LassoCV(cv=5, max_iter=5000, random_state=0)
        ml_m = LassoCV(cv=5, max_iter=5000, random_state=0)
    else:
        ml_l = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05,
                            subsample=0.8, random_state=0, verbosity=0)
        ml_m = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05,
                             subsample=0.8, random_state=0, verbosity=0, eval_metric="logloss")

    model = dml.DoubleMLPLR(dml_data, ml_l=ml_l, ml_m=ml_m, n_folds=n_folds, score="partialling out")
    model.fit()

    estimate = model.coef[0]
    se = model.se[0]
    ci = model.confint(level=0.95)
    covers = int(ci.iloc[0, 0] <= true_ate <= ci.iloc[0, 1])

    # Raw components for decomposition
    D = df["sim_treat"].to_numpy()
    m_0 = df["propensity"].to_numpy()  # true propensity from DGP
    m_hat = model.predictions["ml_m"].flatten()

    V = D - m_0          # true treatment residual
    delta = m_0 - m_hat  # misspecification error
    D_tilde = D - m_hat  # observed residual = V + delta

    var_D = D.var()
    var_V = V.var()
    var_delta = delta.var()
    var_D_tilde = D_tilde.var()
    cov_V_delta = np.cov(V, delta)[0, 1]

    r_d = var_D_tilde / var_D if var_D > 0 else np.nan
    r_d_star = var_V / var_D if var_D > 0 else np.nan
    inflation = var_delta / var_D if var_D > 0 else np.nan

    return {
        "estimate": estimate,
        "se": se,
        "covers": covers,
        "bias": estimate - true_ate,
        "r_d": r_d,
        "r_d_star": r_d_star,
        "inflation": inflation,
        "var_D": var_D,
        "var_V": var_V,
        "var_delta": var_delta,
        "var_D_tilde": var_D_tilde,
        "cov_V_delta": cov_V_delta,
        "decomp_check": var_V + var_delta + 2 * cov_V_delta,  # should equal var_D_tilde
        "learner": learner,
        "overlap_strength": df["overlap_strength"].iloc[0],
        "complexity": df["complexity"].iloc[0],
        "seed": df["seed"].iloc[0],
    }


def main():
    base_df = pd.read_csv("../processed/ihdp_sim_processed.csv")

    # Run decomposition for structural DGP, 50 reps per setting
    overlap_strengths = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    complexities = ["linear", "nonlinear"]
    learners = ["lasso", "xgboost"]
    n_reps = 50

    total = len(overlap_strengths) * len(complexities) * len(learners) * n_reps
    print(f"Running {total} estimations for decomposition verification...")

    results = []
    done = 0
    for alpha in overlap_strengths:
        for complexity in complexities:
            for seed in range(1, n_reps + 1):
                synth = generate_dataset(
                    base_df, sample_size=None, overlap_strength=alpha,
                    complexity=complexity, seed=seed, propensity_model="structural",
                )
                for learner in learners:
                    t0 = time.time()
                    result = estimate_with_decomposition(synth, learner=learner)
                    elapsed = time.time() - t0
                    result["elapsed"] = elapsed
                    results.append(result)
                    done += 1
                    if done % 100 == 0:
                        print(f"  [{done}/{total}] alpha={alpha}, {complexity}, {learner}")

    df = pd.DataFrame(results)

    # Save raw results
    out_dir = Path("../output")
    df.to_csv(out_dir / "decomposition_results.csv", index=False)
    print(f"\nSaved {len(df)} results to {out_dir / 'decomposition_results.csv'}")

    # ── Verification 1: Decomposition holds ───────────────────────────────
    print("\n" + "=" * 70)
    print("VERIFICATION 1: Var(D_tilde) = Var(V) + Var(delta) + 2*Cov(V,delta)")
    print("=" * 70)

    df["decomp_error"] = (df["var_D_tilde"] - df["decomp_check"]).abs()
    print(f"\nMax |decomp error| across all {len(df)} reps: {df['decomp_error'].max():.2e}")
    print(f"Mean Cov(V, delta) across all reps: {df['cov_V_delta'].mean():.6f}")
    print("(Should be ~0 by iterated expectations)")

    # ── Verification 2: R_D = R_D* + inflation ────────────────────────────
    print("\n" + "=" * 70)
    print("VERIFICATION 2: R_D = R_D* + Var(delta)/Var(D)")
    print("=" * 70)

    df["r_d_reconstructed"] = df["r_d_star"] + df["inflation"]
    df["r_d_recon_error"] = (df["r_d"] - df["r_d_reconstructed"]).abs()
    print(f"\nMax |R_D - (R_D* + inflation)| across all reps: {df['r_d_recon_error'].max():.2e}")

    # ── Summary table by learner × overlap ────────────────────────────────
    print("\n" + "=" * 70)
    print("DECOMPOSITION TABLE: Mean values by learner × overlap (nonlinear surface)")
    print("=" * 70)

    nl = df[df["complexity"] == "nonlinear"]
    summary = nl.groupby(["learner", "overlap_strength"]).agg(
        R_D=("r_d", "mean"),
        R_D_star=("r_d_star", "mean"),
        inflation=("inflation", "mean"),
        cov_V_delta=("cov_V_delta", "mean"),
        abs_bias=("bias", lambda x: x.abs().mean()),
        coverage=("covers", "mean"),
    ).round(4)
    print(summary.to_string())

    # ── Figure: Inflation scatter ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: R_D vs R_D* + inflation (should be 45-degree line)
    ax = axes[0]
    for learner, color in [("lasso", "#d62728"), ("xgboost", "#1f77b4")]:
        sub = df[df["learner"] == learner]
        ax.scatter(sub["r_d_star"] + sub["inflation"], sub["r_d"],
                   alpha=0.3, s=10, color=color, label=learner.capitalize())
    lims = [0, 1.2]
    ax.plot(lims, lims, "k--", alpha=0.5, linewidth=1)
    ax.set_xlabel("R_D* + Var(δ)/Var(D)")
    ax.set_ylabel("Observed R_D")
    ax.set_title("Decomposition verification")
    ax.legend()
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # Panel 2: Inflation vs |Bias|
    ax = axes[1]
    for learner, color in [("lasso", "#d62728"), ("xgboost", "#1f77b4")]:
        sub = df[df["learner"] == learner]
        ax.scatter(sub["inflation"], sub["bias"].abs(),
                   alpha=0.3, s=10, color=color, label=learner.capitalize())
    ax.set_xlabel("Var(δ)/Var(D) (misspecification inflation)")
    ax.set_ylabel("|Bias|")
    ax.set_title("Inflation drives bias")
    ax.legend()

    # Panel 3: R_D vs |Bias| colored by learner
    ax = axes[2]
    for learner, color in [("lasso", "#d62728"), ("xgboost", "#1f77b4")]:
        sub = df[df["learner"] == learner]
        ax.scatter(sub["r_d"], sub["bias"].abs(),
                   alpha=0.3, s=10, color=color, label=learner.capitalize())
    ax.set_xlabel("R_D")
    ax.set_ylabel("|Bias|")
    ax.set_title("R_D vs |Bias| (the anti-conservative pattern)")
    ax.legend()

    fig.suptitle("Empirical Verification of Anti-Conservative Decomposition", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / "figures" / "decomposition_verification.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_dir / 'figures' / 'decomposition_verification.png'}")


if __name__ == "__main__":
    main()

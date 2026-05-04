"""Pre-MC validation checks for the IHDP DML overlap simulation.

Run this before committing to the full 200-rep Monte Carlo grid.
It checks four things:

1. ATE calibration: mean true_tau ≈ 4 across many seeds (both surfaces)
2. Propensity distributions: summary stats at each overlap level
3. Baseline coverage: ~95% coverage at overlap=0 with 50 reps
4. Seed independence: different seeds produce different datasets
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_ihdp_synthetic import generate_dataset
from dml_simulation import estimate_ate


def load_base_data() -> pd.DataFrame:
    """Load the cleaned IHDP data."""
    path = Path(__file__).resolve().parent.parent / "processed" / "ihdp_sim_processed.csv"
    if not path.exists():
        # Fall back to raw sim file
        path = Path(__file__).resolve().parent.parent / "hill_data" / "ihdp_sim.csv"
    df = pd.read_csv(path)
    print(f"Loaded base data: {path.name} ({df.shape[0]} rows, {df.shape[1]} cols)\n")
    return df


# ---------------------------------------------------------------------------
# Check 1: ATE calibration
# ---------------------------------------------------------------------------
def check_ate_calibration(base_df: pd.DataFrame, n_seeds: int = 50) -> bool:
    """Verify mean true_tau ≈ 4 across seeds for both surfaces."""
    print("=" * 60)
    print("CHECK 1: ATE calibration (mean true_tau across seeds)")
    print("=" * 60)
    passed = True

    for complexity in ["linear", "nonlinear"]:
        ates = []
        for seed in range(1, n_seeds + 1):
            synth = generate_dataset(
                base_df,
                sample_size=None,
                overlap_strength=0.0,
                complexity=complexity,
                seed=seed,
            )
            ates.append(synth["true_tau"].mean())

        mean_ate = np.mean(ates)
        std_ate = np.std(ates)
        min_ate = np.min(ates)
        max_ate = np.max(ates)

        ok = abs(mean_ate - 4.0) < 0.1
        status = "PASS" if ok else "FAIL"
        if not ok:
            passed = False

        print(f"\n  Surface: {complexity} ({n_seeds} seeds)")
        print(f"  Mean ATE:  {mean_ate:.4f}  (target: 4.0)")
        print(f"  Std ATE:   {std_ate:.4f}")
        print(f"  Range:     [{min_ate:.4f}, {max_ate:.4f}]")
        print(f"  [{status}] |mean - 4| = {abs(mean_ate - 4.0):.4f} (threshold: 0.1)")

    print()
    return passed


# ---------------------------------------------------------------------------
# Check 2: Propensity distributions
# ---------------------------------------------------------------------------
def check_propensity_distributions(base_df: pd.DataFrame) -> bool:
    """Inspect propensity distributions at each overlap level."""
    print("=" * 60)
    print("CHECK 2: Propensity distributions at each overlap level")
    print("=" * 60)
    passed = True

    for alpha in [0.0, 2.0, 3.0, 5.0]:
        synth = generate_dataset(
            base_df,
            sample_size=None,
            overlap_strength=alpha,
            complexity="linear",
            seed=1,
        )
        p = synth["propensity"]
        pct_clipped = ((p <= 0.021) | (p >= 0.979)).mean() * 100
        treat_rate = synth["sim_treat"].mean()

        print(f"\n  overlap_strength = {alpha}")
        print(f"  Mean propensity:    {p.mean():.4f}")
        print(f"  Std propensity:     {p.std():.4f}")
        print(f"  Min/Max:            [{p.min():.4f}, {p.max():.4f}]")
        print(f"  % near clip bounds: {pct_clipped:.1f}%")
        print(f"  Treatment rate:     {treat_rate:.3f}")
        print(f"  Quartiles:          {p.quantile(0.25):.4f} / {p.quantile(0.50):.4f} / {p.quantile(0.75):.4f}")

        # At overlap=0, propensity should be constant (~0.38)
        if alpha == 0.0:
            ok = p.std() < 0.01
            status = "PASS" if ok else "FAIL"
            if not ok:
                passed = False
            print(f"  [{status}] Std should be ~0 at overlap=0 (got {p.std():.4f})")

        # At overlap=5, expect substantial clipping
        if alpha == 5.0:
            ok = pct_clipped > 30
            status = "PASS" if ok else "FAIL"
            if not ok:
                passed = False
            print(f"  [{status}] Expected >30% clipped at overlap=5 (got {pct_clipped:.1f}%)")

    print()
    return passed


# ---------------------------------------------------------------------------
# Check 3: Baseline coverage at overlap=0
# ---------------------------------------------------------------------------
def check_baseline_coverage(base_df: pd.DataFrame, n_reps: int = 50) -> bool:
    """Run 50 reps at overlap=0 and confirm coverage is ~95%."""
    print("=" * 60)
    print(f"CHECK 3: Baseline coverage at overlap=0 ({n_reps} reps)")
    print("=" * 60)
    passed = True

    for complexity in ["linear", "nonlinear"]:
        for learner in ["lasso", "xgboost"]:
            covers = []
            t0 = time.time()
            for seed in range(1, n_reps + 1):
                synth = generate_dataset(
                    base_df,
                    sample_size=None,
                    overlap_strength=0.0,
                    complexity=complexity,
                    seed=seed,
                )
                result = estimate_ate(synth, learner=learner)
                covers.append(result["covers"])

            elapsed = time.time() - t0
            coverage = np.mean(covers) * 100
            se = np.sqrt(coverage / 100 * (1 - coverage / 100) / n_reps) * 100

            # Coverage should be at least 85% (allowing for MC noise)
            ok = coverage >= 85
            status = "PASS" if ok else "FAIL"
            if not ok:
                passed = False

            print(f"\n  {complexity} / {learner}: coverage = {coverage:.0f}% ± {se:.1f}pp  ({elapsed:.1f}s)")
            print(f"  [{status}] Coverage >= 85% threshold")

    print()
    return passed


# ---------------------------------------------------------------------------
# Check 4: Seed independence
# ---------------------------------------------------------------------------
def check_seed_independence(base_df: pd.DataFrame) -> bool:
    """Confirm different seeds produce different treatment vectors and outcomes."""
    print("=" * 60)
    print("CHECK 4: Seed independence")
    print("=" * 60)
    passed = True

    for complexity in ["linear", "nonlinear"]:
        synth1 = generate_dataset(base_df, sample_size=None, overlap_strength=2.0, complexity=complexity, seed=1)
        synth2 = generate_dataset(base_df, sample_size=None, overlap_strength=2.0, complexity=complexity, seed=2)
        synth1_dup = generate_dataset(base_df, sample_size=None, overlap_strength=2.0, complexity=complexity, seed=1)

        treat_differ = (synth1["sim_treat"] != synth2["sim_treat"]).sum()
        y_differ = (synth1["y"] != synth2["y"]).sum()
        treat_same = (synth1["sim_treat"] == synth1_dup["sim_treat"]).all()
        y_same = np.allclose(synth1["y"], synth1_dup["y"])

        ok_different = treat_differ > 0 and y_differ > 0
        ok_reproducible = treat_same and y_same
        ok = ok_different and ok_reproducible
        if not ok:
            passed = False

        status = "PASS" if ok else "FAIL"
        print(f"\n  Surface: {complexity}")
        print(f"  Seed 1 vs Seed 2: {treat_differ} treatment differences, {y_differ} outcome differences")
        print(f"  Seed 1 vs Seed 1 (rerun): treatments match={treat_same}, outcomes match={y_same}")
        print(f"  [{status}] Different seeds differ AND same seed reproduces")

    print()
    return passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("\n" + "=" * 60)
    print("  PRE-MC VALIDATION CHECKS")
    print("=" * 60 + "\n")

    base_df = load_base_data()

    results = {}
    results["ATE calibration"] = check_ate_calibration(base_df)
    results["Propensity distributions"] = check_propensity_distributions(base_df)
    results["Seed independence"] = check_seed_independence(base_df)
    # Run coverage last since it's the slowest
    results["Baseline coverage"] = check_baseline_coverage(base_df)

    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    all_passed = True
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_passed = False
        print(f"  [{status}] {name}")

    print()
    if all_passed:
        print("All checks passed. Safe to proceed with the full MC run.")
    else:
        print("Some checks FAILED. Investigate before running the full MC grid.")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()

"""Analyze the joint (R_D, R_Y) diagnostic for DML reliability.

Produces:
  - Figure 1: 2D scatter of (R_D, R_Y) colored by coverage
  - Figure 2: Coverage heatmap in (R_D, R_Y) space
  - Figure 3: R_Y explains coverage gap at fixed R_D
  - Figure 4: Predictive comparison (R_D alone vs R_Y alone vs joint)
  - Figure 5: Practical decision boundaries
  - Table 1: Summary by (learner, overlap_strength, surface)
  - Table 2: Logistic regression predictive comparison
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LEARNER_STYLES = {
    "lasso": {"color": "#d62728", "marker": "o", "label": "Lasso"},
    "xgboost": {"color": "#1f77b4", "marker": "s", "label": "XGBoost"},
    "lasso_logistic": {"color": "#ff7f0e", "marker": "^", "label": "Lasso+Logistic"},
    "ridge": {"color": "#2ca02c", "marker": "D", "label": "Ridge"},
    "rf": {"color": "#9467bd", "marker": "v", "label": "Random Forest"},
}
SURFACE_TITLES = {
    "linear": "Surface A (linear)",
    "nonlinear": "Surface B (nonlinear)",
}


def _setup_style():
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 200
    plt.rcParams["savefig.bbox"] = "tight"


def load_results(*paths: str) -> pd.DataFrame:
    dfs = []
    for p in paths:
        df = pd.read_csv(p)
        df["bias"] = df["estimate"] - 4.0
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)

    # Unify outcome diagnostic column across PLR and IRM.
    # PLR: r_y. IRM: r_y0, r_y1 — use r_y0 (control arm) as the canonical
    # outcome diagnostic for comparison purposes (arm-specific and
    # contamination-free).
    if "r_y" not in combined.columns:
        if "r_y0" in combined.columns:
            combined["r_y"] = combined["r_y0"]
        else:
            raise ValueError(
                "Input CSV must contain either 'r_y' (PLR) or 'r_y0' (IRM)."
            )
    else:
        # If both r_y and r_y0 are present (mixed input), prefer r_y for PLR rows
        # and r_y0 for IRM rows based on framework column.
        if "r_y0" in combined.columns and "framework" in combined.columns:
            irm_mask = combined["framework"] == "IRM"
            combined.loc[irm_mask, "r_y"] = combined.loc[irm_mask, "r_y0"]

    # Fill missing outcome_rmse (IRM does not compute it) with NaN
    if "outcome_rmse" not in combined.columns:
        combined["outcome_rmse"] = np.nan

    return combined


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["overlap_strength", "complexity", "learner"]
    if "framework" in df.columns:
        group_cols.append("framework")

    agg_dict = {
        "mean_est": ("estimate", "mean"),
        "bias": ("bias", "mean"),
        "abs_bias": ("bias", lambda x: x.abs().mean()),
        "rmse": ("estimate", lambda x: np.sqrt(((x - 4.0) ** 2).mean())),
        "mean_se": ("se", "mean"),
        "coverage": ("covers", "mean"),
        "mean_r_d": ("r_d", "mean"),
        "mean_r_y": ("r_y", "mean"),
        "mean_outcome_rmse": ("outcome_rmse", "mean"),
        "n_reps": ("estimate", "count"),
    }
    # Add IRM-specific columns if present
    if "r_y0" in df.columns:
        agg_dict["mean_r_y0"] = ("r_y0", "mean")
    if "r_y1" in df.columns:
        agg_dict["mean_r_y1"] = ("r_y1", "mean")

    summary = df.groupby(group_cols).agg(**agg_dict).reset_index()
    return summary


# ---------------------------------------------------------------------------
# Figure 1: 2D scatter (R_D, R_Y) colored by coverage (cell-level)
# ---------------------------------------------------------------------------
def plot_2d_scatter(summary: pd.DataFrame, fig_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, complexity in zip(axes, ["linear", "nonlinear"]):
        sub = summary[summary["complexity"] == complexity]
        for _, row in sub.iterrows():
            style = LEARNER_STYLES.get(row["learner"], {"color": "gray", "marker": "x"})
            cov = row["coverage"]
            # Color by coverage: red (bad) to green (good)
            rgba = plt.cm.RdYlGn(cov)
            ax.scatter(
                row["mean_r_d"], row["mean_r_y"],
                c=[rgba], marker=style["marker"], s=80,
                edgecolors="black", linewidths=0.5, zorder=3,
            )

        # Legend for learners (marker shape)
        for learner, style in LEARNER_STYLES.items():
            if learner in sub["learner"].values:
                ax.scatter([], [], marker=style["marker"], c="gray",
                           edgecolors="black", s=60, label=style["label"])

        ax.set_xlabel("$R_D$ (propensity-side)")
        ax.set_ylabel("$R_Y$ (outcome-side)")
        ax.set_title(SURFACE_TITLES[complexity])
        ax.legend(loc="lower right", fontsize=9)

    # Shared colorbar
    sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=mcolors.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label("Coverage")

    fig.suptitle("Joint Diagnostic: ($R_D$, $R_Y$) Colored by Coverage  "
                 "— PLR: $R_Y$ contaminated; IRM: $R_{Y0}$",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 0.92, 0.95])
    fig.savefig(fig_dir / "joint_2d_scatter.png")
    plt.close(fig)
    print(f"  Saved {fig_dir / 'joint_2d_scatter.png'}")


# ---------------------------------------------------------------------------
# Figure 2: Coverage heatmap in (R_D, R_Y) space (rep-level)
# ---------------------------------------------------------------------------
def plot_coverage_heatmap(df: pd.DataFrame, fig_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, complexity in zip(axes, ["linear", "nonlinear"]):
        sub = df[df["complexity"] == complexity].copy()

        # Bin into 2D grid
        rd_bins = np.arange(0, 1.15, 0.10)
        ry_bins = np.arange(-0.5, 1.05, 0.10)
        sub["rd_bin"] = pd.cut(sub["r_d"], rd_bins)
        sub["ry_bin"] = pd.cut(sub["r_y"], ry_bins)

        heatmap_data = sub.groupby(["ry_bin", "rd_bin"], observed=False)["covers"].mean()
        heatmap_data = heatmap_data.unstack(level="rd_bin")

        # Plot
        sns.heatmap(
            heatmap_data, ax=ax, vmin=0, vmax=1,
            cmap="RdYlGn", annot=False,
            cbar_kws={"label": "Coverage"},
            xticklabels=2, yticklabels=3,
        )
        ax.invert_yaxis()
        ax.set_xlabel("$R_D$ bin")
        ax.set_ylabel("$R_Y$ bin")
        ax.set_title(SURFACE_TITLES[complexity])

    fig.suptitle("Coverage Heatmap in ($R_D$, $R_Y$) Space",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(fig_dir / "joint_coverage_heatmap.png")
    plt.close(fig)
    print(f"  Saved {fig_dir / 'joint_coverage_heatmap.png'}")


# ---------------------------------------------------------------------------
# Figure 3: R_Y explains coverage at fixed R_D
# ---------------------------------------------------------------------------
def plot_ry_marginal_value(df: pd.DataFrame, fig_dir: Path):
    """At similar R_D values, show that R_Y predicts coverage."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Use nonlinear surface (where outcome matters most)
    sub = df[df["complexity"] == "nonlinear"].copy()

    # Panel 1: Coverage vs R_D alone (showing the problem)
    ax = axes[0]
    for learner, style in LEARNER_STYLES.items():
        lsub = sub[sub["learner"] == learner]
        if len(lsub) == 0:
            continue
        cell = lsub.groupby("overlap_strength").agg(
            r_d=("r_d", "mean"), coverage=("covers", "mean"),
        ).reset_index()
        ax.plot(cell["r_d"], cell["coverage"],
                color=style["color"], marker=style["marker"],
                label=style["label"], linewidth=2, markersize=7)
    ax.axhline(0.95, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("$R_D$ (propensity-side)")
    ax.set_ylabel("Coverage")
    ax.set_title("Coverage vs $R_D$ alone\n(different learners at similar $R_D$ → different coverage)")
    ax.legend(fontsize=9)

    # Panel 2: Coverage vs R_Y at fixed R_D bands
    ax = axes[1]
    rd_bands = [(0.0, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 1.10)]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(rd_bands)))
    for (lo, hi), color in zip(rd_bands, colors):
        band = sub[(sub["r_d"] >= lo) & (sub["r_d"] < hi)]
        if len(band) < 20:
            continue
        # Bin by R_Y within this R_D band
        ry_bins = pd.qcut(band["r_y"], q=5, duplicates="drop")
        band_agg = band.groupby(ry_bins, observed=True).agg(
            r_y=("r_y", "mean"), coverage=("covers", "mean"), n=("covers", "count"),
        ).reset_index(drop=True)
        ax.plot(band_agg["r_y"], band_agg["coverage"],
                color=color, marker="o", linewidth=2, markersize=6,
                label=f"$R_D$ in [{lo:.2f}, {hi:.2f})")
    ax.axhline(0.95, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("$R_Y$ (outcome-side)")
    ax.set_ylabel("Coverage")
    ax.set_title("Coverage vs $R_Y$ at fixed $R_D$ bands\n($R_Y$ explains residual variation)")
    ax.legend(fontsize=9)

    fig.suptitle("Does $R_Y$ Add Predictive Value Beyond $R_D$?  "
                 "(AUC comparison in predictive_comparison table)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(fig_dir / "joint_ry_marginal_value.png")
    plt.close(fig)
    print(f"  Saved {fig_dir / 'joint_ry_marginal_value.png'}")


# ---------------------------------------------------------------------------
# Figure 4: Predictive comparison (logistic regression AUC)
# ---------------------------------------------------------------------------
def predictive_comparison(df: pd.DataFrame, table_dir: Path, fig_dir: Path):
    """Compare R_D alone, R_Y alone, and joint for predicting coverage."""
    results = []

    for complexity in ["linear", "nonlinear"]:
        sub = df[df["complexity"] == complexity].dropna(subset=["r_d", "r_y", "covers"])
        if len(sub) < 50:
            continue

        y = sub["covers"].values
        X_rd = sub[["r_d"]].values
        X_ry = sub[["r_y"]].values
        X_joint = sub[["r_d", "r_y"]].values

        # Skip if no variance in outcome
        if y.sum() == 0 or y.sum() == len(y):
            continue

        for name, X in [("R_D alone", X_rd), ("R_Y alone", X_ry), ("R_D + R_Y", X_joint)]:
            model = LogisticRegression(max_iter=5000, random_state=0)
            model.fit(X, y)
            proba = model.predict_proba(X)[:, 1]
            auc = roc_auc_score(y, proba)
            # McFadden pseudo-R²
            ll_model = np.mean(y * np.log(proba + 1e-10) + (1 - y) * np.log(1 - proba + 1e-10))
            p_bar = y.mean()
            ll_null = p_bar * np.log(p_bar + 1e-10) + (1 - p_bar) * np.log(1 - p_bar + 1e-10)
            pseudo_r2 = 1 - ll_model / ll_null if ll_null != 0 else np.nan

            results.append({
                "surface": complexity,
                "predictors": name,
                "auc": auc,
                "pseudo_r2": pseudo_r2,
                "n": len(y),
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(table_dir / "predictive_comparison.csv", index=False)
    print(f"  Saved {table_dir / 'predictive_comparison.csv'}")

    # Bar chart
    if len(results_df) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, metric, label in zip(axes, ["auc", "pseudo_r2"], ["AUC", "Pseudo-$R^2$"]):
            pivot = results_df.pivot(index="surface", columns="predictors", values=metric)
            pivot = pivot[["R_D alone", "R_Y alone", "R_D + R_Y"]]
            pivot.plot(kind="bar", ax=ax, rot=0, color=["#d62728", "#1f77b4", "#2ca02c"])
            ax.set_ylabel(label)
            ax.set_xlabel("")
            ax.set_title(f"Predicting Coverage: {label}")
            ax.legend(fontsize=9)
            if metric == "auc":
                ax.set_ylim(0.5, 1.0)
                ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5)
        fig.suptitle("Predicting Coverage from ($R_D$, $R_Y$)  "
                     "— PLR finding: $R_Y$ does not help; IRM: $R_{Y0}$ also fails due to averaging",
                     fontsize=12, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(fig_dir / "joint_predictive_comparison.png")
        plt.close(fig)
        print(f"  Saved {fig_dir / 'joint_predictive_comparison.png'}")

    return results_df


# ---------------------------------------------------------------------------
# Figure 5: Decision boundary in (R_D, R_Y) space
# ---------------------------------------------------------------------------
def plot_decision_boundary(df: pd.DataFrame, fig_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, complexity in zip(axes, ["linear", "nonlinear"]):
        sub = df[df["complexity"] == complexity].dropna(subset=["r_d", "r_y", "covers"])
        if len(sub) < 50:
            continue

        X = sub[["r_d", "r_y"]].values
        y = sub["covers"].values

        if y.sum() == 0 or y.sum() == len(y):
            continue

        model = LogisticRegression(max_iter=5000, random_state=0)
        model.fit(X, y)

        # Decision surface
        rd_range = np.linspace(X[:, 0].min() - 0.05, X[:, 0].max() + 0.05, 200)
        ry_range = np.linspace(X[:, 1].min() - 0.05, X[:, 1].max() + 0.05, 200)
        rd_grid, ry_grid = np.meshgrid(rd_range, ry_range)
        grid_X = np.c_[rd_grid.ravel(), ry_grid.ravel()]
        proba = model.predict_proba(grid_X)[:, 1].reshape(rd_grid.shape)

        # Filled contour for P(covers)
        contour = ax.contourf(rd_grid, ry_grid, proba,
                              levels=np.arange(0, 1.05, 0.05),
                              cmap="RdYlGn", alpha=0.6)
        ax.contour(rd_grid, ry_grid, proba,
                   levels=[0.90, 0.95], colors=["black", "black"],
                   linewidths=[1.5, 2.0], linestyles=["--", "-"])

        # Overlay cell-level means
        cell = sub.groupby(["learner", "overlap_strength"]).agg(
            r_d=("r_d", "mean"), r_y=("r_y", "mean"), coverage=("covers", "mean"),
        ).reset_index()
        for _, row in cell.iterrows():
            style = LEARNER_STYLES.get(row["learner"], {"marker": "x"})
            ax.scatter(row["r_d"], row["r_y"],
                       marker=style["marker"], s=60,
                       c=[plt.cm.RdYlGn(row["coverage"])],
                       edgecolors="black", linewidths=0.7, zorder=4)

        ax.set_xlabel("$R_D$ (propensity-side)")
        ax.set_ylabel("$R_Y$ (outcome-side)")
        ax.set_title(SURFACE_TITLES[complexity])

    # Legend for learners
    for learner, style in LEARNER_STYLES.items():
        if learner in df["learner"].values:
            axes[0].scatter([], [], marker=style["marker"], c="gray",
                            edgecolors="black", s=50, label=style["label"])
    axes[0].legend(loc="lower right", fontsize=8)

    fig.colorbar(contour, ax=axes, shrink=0.8, pad=0.02, label="P(coverage)")
    fig.suptitle("Decision Boundary: Safe vs Unreliable Regions",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 0.92, 0.95])
    fig.savefig(fig_dir / "joint_decision_boundary.png")
    plt.close(fig)
    print(f"  Saved {fig_dir / 'joint_decision_boundary.png'}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
def save_joint_summary(summary: pd.DataFrame, table_dir: Path):
    cols = [
        "overlap_strength", "complexity", "learner", "framework",
        "bias", "rmse", "mean_se", "coverage",
        "mean_r_d", "mean_r_y", "mean_r_y0", "mean_r_y1", "mean_outcome_rmse",
    ]
    available = [c for c in cols if c in summary.columns]
    table = summary[available].round(4)
    table.to_csv(table_dir / "joint_summary.csv", index=False)
    print(f"  Saved {table_dir / 'joint_summary.csv'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze joint (R_D, R_Y) diagnostic for DML reliability."
    )
    parser.add_argument(
        "--input", nargs="+", required=True,
        help="Path(s) to MC results CSV(s) containing r_d and r_y columns.",
    )
    parser.add_argument(
        "--output-dir", default="output",
        help="Directory for figures/ and tables/ subdirectories.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    fig_dir = output_dir / "figures" / "joint"
    table_dir = output_dir / "tables" / "joint"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {args.input}...")
    df = load_results(*args.input)
    print(f"  Loaded {len(df)} rows")

    if "framework" in df.columns:
        frameworks = df["framework"].dropna().unique().tolist()
        print(f"  Frameworks present: {frameworks}")

    summary = build_summary(df)

    _setup_style()

    print("\nGenerating figures...")
    plot_2d_scatter(summary, fig_dir)
    plot_coverage_heatmap(df, fig_dir)
    plot_ry_marginal_value(df, fig_dir)
    pred_df = predictive_comparison(df, table_dir, fig_dir)
    plot_decision_boundary(df, fig_dir)

    print("\nGenerating tables...")
    save_joint_summary(summary, table_dir)

    # Print key results
    print("\n" + "=" * 70)
    print("KEY RESULTS")
    print("=" * 70)
    if len(pred_df) > 0:
        print("\nPredictive comparison (AUC for coverage prediction):")
        for _, row in pred_df.iterrows():
            print(f"  {row['surface']:>10} | {row['predictors']:<12} | AUC={row['auc']:.3f} | Pseudo-R²={row['pseudo_r2']:.3f}")

    print("\nSummary at maximum overlap (strength=5.0, nonlinear):")
    extreme = summary[(summary["overlap_strength"] == 5.0) & (summary["complexity"] == "nonlinear")]
    if len(extreme) > 0:
        for _, row in extreme.iterrows():
            print(f"  {row['learner']:>15}: R_D={row['mean_r_d']:.3f}  R_Y={row['mean_r_y']:.3f}  "
                  f"Coverage={row['coverage']:.3f}  |Bias|={row['abs_bias']:.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()

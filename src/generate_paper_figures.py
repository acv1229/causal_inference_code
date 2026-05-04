"""
Generate publication-quality figures for paper_draft.tex.
Run from project root: python src/generate_paper_figures.py
Outputs: output/figures/paper/figN_*.{png,pdf}  (300 dpi)
All plots include 95% CI error bars from MC replications or bootstrap.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
os.makedirs("output/figures/paper", exist_ok=True)

# ---------------------------------------------------------------------------
# Publication style
# ---------------------------------------------------------------------------
matplotlib.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["DejaVu Serif", "Times New Roman", "Times", "serif"],
    "font.size":          11,
    "axes.titlesize":     12,
    "axes.labelsize":     11,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "legend.fontsize":    9,
    "axes.linewidth":     0.8,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.25,
    "grid.linewidth":     0.5,
    "grid.color":         "#999999",
    "lines.linewidth":    2.0,
    "lines.markersize":   6,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "figure.facecolor":   "white",
})

# Colorblind-safe palette (Wong 2011)
C = {
    "lasso":          "#E69F00",
    "xgboost":        "#0072B2",
    "lasso_logistic": "#009E73",
    "ridge":          "#CC79A7",
    "rf":             "#D55E00",
}
LS = {
    "lasso":          "-",
    "xgboost":        "--",
    "lasso_logistic": "-.",
    "ridge":          ":",
    "rf":             (0, (3, 1, 1, 1)),
}
MK = {
    "lasso":          "o",
    "xgboost":        "s",
    "lasso_logistic": "^",
    "ridge":          "D",
    "rf":             "v",
}
LABEL = {
    "lasso":          "Lasso",
    "xgboost":        "XGBoost",
    "lasso_logistic": "Lasso+Logistic",
    "ridge":          "Ridge",
    "rf":             "Random Forest",
}
SURF = {
    "linear":    "Surface A (Linear)",
    "nonlinear": "Surface B (Nonlinear)",
}
TRUE_ATE = 4.0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def save(name):
    for ext in ("png", "pdf"):
        plt.savefig(f"output/figures/paper/{name}.{ext}")
    plt.close()
    print(f"  saved {name}")


def plabel(ax, letter, dx=-0.13, dy=1.04):
    ax.text(dx, dy, letter, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="bottom")


def cov_se(p, n, z=1.96):
    """95 % CI half-width for a proportion."""
    return z * np.sqrt(p * (1 - p) / max(n, 1))


def mean_se(s, z=1.96):
    """95 % CI half-width for a mean."""
    return z * s.std(ddof=1) / np.sqrt(max(len(s), 1))


def rmse_boot_se(sq_err, n_boot=500, seed=0):
    rng = np.random.RandomState(seed)
    boots = [np.sqrt(sq_err.sample(len(sq_err), replace=True,
                                   random_state=rng).mean())
             for _ in range(n_boot)]
    return 1.96 * np.std(boots)


def auc_bootstrap(df, preds, n_boot=1000, seed=42):
    rng = np.random.RandomState(seed)
    df2 = df[preds + ["covers"]].dropna()
    X, y = df2[preds].values, df2["covers"].values
    if len(np.unique(y)) < 2 or len(y) < 10:
        return np.nan, np.nan, np.nan
    lr = LogisticRegression(max_iter=2000)
    lr.fit(X, y)
    base = roc_auc_score(y, lr.predict_proba(X)[:, 1])
    boots = []
    for _ in range(n_boot):
        idx = rng.choice(len(y), len(y), replace=True)
        Xb, yb = X[idx], y[idx]
        if len(np.unique(yb)) < 2:
            continue
        lrb = LogisticRegression(max_iter=2000)
        lrb.fit(Xb, yb)
        boots.append(roc_auc_score(yb, lrb.predict_proba(Xb)[:, 1]))
    boots = np.array(boots)
    return base, np.percentile(boots, 2.5), np.percentile(boots, 97.5)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading data...")
v7  = pd.read_csv("output/mc_results_plr_full.csv")
v10 = pd.read_csv("output/mc_results_v10_merged.csv")
v7["sq_error"]  = (v7["estimate"]  - TRUE_ATE) ** 2
v10["sq_error"] = (v10["estimate"] - TRUE_ATE) ** 2

S7  = sorted(v7["overlap_strength"].unique())
S10 = sorted(v10["overlap_strength"].unique())

pct = mtick.PercentFormatter(xmax=1, decimals=0)


# ===========================================================================
# FIG 1  Coverage collapse  —  line plots, 2 panels
# ===========================================================================
print("Fig 1: coverage collapse...")
L1 = ["lasso", "xgboost", "lasso_logistic"]

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
fig.subplots_adjust(wspace=0.06)

for ax, surf in zip(axes, ["linear", "nonlinear"]):
    sub = v7[v7["complexity"] == surf]
    for lrn in L1:
        xs, ys, es = [], [], []
        for s in S7:
            c = sub[(sub["learner"] == lrn) & (sub["overlap_strength"] == s)]["covers"]
            p = c.mean()
            xs.append(s); ys.append(p); es.append(cov_se(p, len(c)))
        ax.errorbar(xs, ys, yerr=es,
                    color=C[lrn], linestyle=LS[lrn], marker=MK[lrn],
                    linewidth=2, markersize=7, capsize=4, capthick=1.5,
                    label=LABEL[lrn])
    ax.axhline(0.95, color="#555", linewidth=1.2, linestyle=":", alpha=0.8,
               label="Nominal 95%")
    ax.set_xlabel(r"Overlap strength $\alpha$", labelpad=4)
    ax.set_title(SURF[surf])
    ax.set_xticks(S7)
    ax.set_ylim(0.08, 1.05)
    ax.yaxis.set_major_formatter(pct)

axes[0].set_ylabel("Empirical 95% CI coverage")
axes[0].legend(loc="lower left", frameon=True, framealpha=0.9,
               edgecolor="#cccccc")
plabel(axes[0], "(a)"); plabel(axes[1], "(b)")
save("fig1_coverage_vs_overlap")


# ===========================================================================
# FIG 2  Outcome-side blindness  —  scatter, IRM α = 5
# ===========================================================================
print("Fig 2: outcome-side blindness...")
L2 = ["lasso_logistic", "xgboost", "ridge", "rf"]

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
fig.subplots_adjust(wspace=0.32)

for ax, surf in zip(axes, ["linear", "nonlinear"]):
    sub = v10[(v10["overlap_strength"] == 5) & (v10["complexity"] == surf)]
    for lrn in L2:
        cell = sub[sub["learner"] == lrn]
        if cell.empty:
            continue
        rd_m  = cell["r_d"].mean();    rd_e  = mean_se(cell["r_d"])
        cv_m  = cell["covers"].mean(); cv_e  = cov_se(cv_m, len(cell))
        ax.errorbar(rd_m, cv_m, xerr=rd_e, yerr=cv_e,
                    color=C[lrn], marker=MK[lrn], markersize=11,
                    linewidth=0, elinewidth=2, capsize=5, capthick=1.5,
                    label=LABEL[lrn], zorder=3)
    ax.set_xlabel(r"$R_D$", labelpad=4)
    ax.set_title(SURF[surf])
    ax.yaxis.set_major_formatter(pct)
    ax.set_ylim(0.15, 1.0)

axes[0].set_ylabel("Empirical 95% CI coverage")
axes[0].legend(loc="upper left", frameon=True, framealpha=0.9,
               edgecolor="#cccccc")
plabel(axes[0], "(a)", dx=-0.15); plabel(axes[1], "(b)", dx=-0.15)
save("fig2_outcome_side_blindness")


# ===========================================================================
# FIG 3  PLR contamination  —  dual-axis, XGBoost nonlinear
# ===========================================================================
print("Fig 3: PLR contamination...")
sub3 = v7[(v7["learner"] == "xgboost") & (v7["complexity"] == "nonlinear")]

fig, ax1 = plt.subplots(figsize=(6, 4.2))
ax2 = ax1.twinx()
for sp in ("top",):
    ax1.spines[sp].set_visible(False)
    ax2.spines[sp].set_visible(False)

col_ry  = "#D55E00"
col_cov = "#0072B2"

ry_m, ry_e, cv_m, cv_e = [], [], [], []
for s in S7:
    c = sub3[sub3["overlap_strength"] == s]
    ry_m.append(c["r_y"].mean());    ry_e.append(mean_se(c["r_y"]))
    p = c["covers"].mean()
    cv_m.append(p);                  cv_e.append(cov_se(p, len(c)))

ax1.errorbar(S7, ry_m, yerr=ry_e, color=col_ry,
             linestyle="-", marker="s", linewidth=2, markersize=7,
             capsize=4, capthick=1.5, label=r"PLR $R_Y$ (left)")
ax2.errorbar(S7, cv_m, yerr=cv_e, color=col_cov,
             linestyle="--", marker="o", linewidth=2, markersize=7,
             capsize=4, capthick=1.5, label="Coverage (right)")
ax2.axhline(0.95, color="#555", linewidth=1.1, linestyle=":", alpha=0.7)

ax1.set_xlabel(r"Overlap strength $\alpha$", labelpad=4)
ax1.set_ylabel(r"Cross-fitted $R_Y$ (PLR)", color=col_ry)
ax2.set_ylabel("Empirical 95% CI coverage", color=col_cov)
ax1.tick_params(axis="y", labelcolor=col_ry)
ax2.tick_params(axis="y", labelcolor=col_cov)
ax2.yaxis.set_major_formatter(pct)
ax1.set_xticks(S7)
ax1.set_title("XGBoost, Surface B (Nonlinear)")

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="center right",
           frameon=True, framealpha=0.9, edgecolor="#cccccc")
save("fig3_plr_contamination")


# ===========================================================================
# FIG 4  IRM removes contamination  —  two lines
# ===========================================================================
print("Fig 4: IRM vs PLR contamination...")
sub4p = v7[(v7["learner"] == "xgboost") & (v7["complexity"] == "nonlinear")]
sub4i = v10[(v10["learner"] == "xgboost") & (v10["complexity"] == "nonlinear")]

fig, ax = plt.subplots(figsize=(6, 4.2))

pm, pe, im, ie = [], [], [], []
for s in S7:
    c = sub4p[sub4p["overlap_strength"] == s]["r_y"]
    pm.append(c.mean()); pe.append(mean_se(c))
for s in S10:
    c = sub4i[sub4i["overlap_strength"] == s]["r_y0"]
    im.append(c.mean()); ie.append(mean_se(c))

ax.errorbar(S7,  pm, yerr=pe, color="#D55E00",
            linestyle="-", marker="s", linewidth=2, markersize=7,
            capsize=4, capthick=1.5, label=r"PLR $R_Y$ (contaminated)")
ax.errorbar(S10, im, yerr=ie, color="#0072B2",
            linestyle="--", marker="o", linewidth=2, markersize=7,
            capsize=4, capthick=1.5, label=r"IRM $R_{Y0}$ (clean)")

ax.set_xlabel(r"Overlap strength $\alpha$", labelpad=4)
ax.set_ylabel(r"Outcome $R^2$")
ax.set_xticks(S7); ax.set_ylim(0.1, 0.95)
ax.set_title("XGBoost, Surface B (Nonlinear)")
ax.legend(frameon=True, framealpha=0.9, edgecolor="#cccccc")
save("fig4_irm_vs_plr")


# ===========================================================================
# FIG 5  Diagnostic staircase  —  grouped bars with bootstrap CI whiskers
# ===========================================================================
print("Fig 5: diagnostic staircase (bootstrap AUCs -- ~1 min)...")

plr_auc, irm_auc = {}, {}
for surf in ["linear", "nonlinear"]:
    sp = v7[v7["complexity"] == surf]
    plr_auc[surf] = {
        "rd":    auc_bootstrap(sp, ["r_d"]),
        "rd_ry": auc_bootstrap(sp, ["r_d", "r_y"]),
    }
    si = v10[v10["complexity"] == surf]
    irm_auc[surf] = {
        "rd":       auc_bootstrap(si, ["r_d"]),
        "rd_ry0":   auc_bootstrap(si, ["r_d", "r_y0"]),
        "rd_local": auc_bootstrap(si, ["r_d", "r_y0_local_05", "r_y1_local"]),
    }

glabels = [
    "PLR\n" + r"$R_D$",
    "PLR\n" + r"$R_D{+}R_Y$",
    "IRM\n" + r"$R_D$",
    "IRM\n" + r"$R_D{+}R_{Y0}$",
    "IRM\n" + r"$R_D{+}$local",
]
bcols   = ["#E69F00", "#E69F00", "#0072B2", "#0072B2", "#009E73"]
balphas = [0.50,       0.95,      0.50,       0.85,      0.95]
bhatch  = ["///",      "",        "///",       "",        ""]

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
fig.subplots_adjust(wspace=0.08)

for ax, surf in zip(axes, ["linear", "nonlinear"]):
    vals = [
        plr_auc[surf]["rd"],
        plr_auc[surf]["rd_ry"],
        irm_auc[surf]["rd"],
        irm_auc[surf]["rd_ry0"],
        irm_auc[surf]["rd_local"],
    ]
    means  = [v[0] for v in vals]
    err_lo = [v[0] - v[1] for v in vals]
    err_hi = [v[2] - v[0] for v in vals]
    x = np.arange(len(means))

    for i, (m, col, a, h) in enumerate(zip(means, bcols, balphas, bhatch)):
        ax.bar(x[i], m, color=col, alpha=a, width=0.65,
               hatch=h, edgecolor="white", linewidth=0.8, zorder=3)
    ax.errorbar(x, means, yerr=[err_lo, err_hi],
                fmt="none", color="#111", linewidth=1.8,
                capsize=5, capthick=1.8, zorder=4)
    ax.axhline(0.5, color="#888", linewidth=1.0, linestyle=":", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(glabels, fontsize=9)
    ax.set_ylim(0.40, 1.0)
    ax.set_title(SURF[surf])

axes[0].set_ylabel("AUC (predicting coverage failure)")

legend_els = [
    mpatches.Patch(facecolor="#E69F00", alpha=0.95, label="PLR"),
    mpatches.Patch(facecolor="#0072B2", alpha=0.85, label="IRM (global $R^2$)"),
    mpatches.Patch(facecolor="#009E73", alpha=0.95, label="IRM (local $R^2$)"),
    mpatches.Patch(facecolor="gray",    alpha=0.35,
                   hatch="///", label=r"$R_D$ alone"),
]
axes[1].legend(handles=legend_els, loc="lower right",
               frameon=True, framealpha=0.9, edgecolor="#cccccc", fontsize=8)
plabel(axes[0], "(a)"); plabel(axes[1], "(b)")
save("fig5_diagnostic_staircase")


# ===========================================================================
# FIG 6  Averaging barrier  —  grouped bars + gap-vs-coverage scatter
# ===========================================================================
print("Fig 6: averaging barrier...")
sub6 = v10[(v10["complexity"] == "nonlinear") & (v10["overlap_strength"] == 5)]
L6   = ["lasso_logistic", "xgboost", "ridge", "rf"]

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
fig.subplots_adjust(wspace=0.38)

# Panel (a): global vs local R_Y0 grouped bars
ax = axes[0]
x, w = np.arange(len(L6)), 0.35
gm = [sub6[sub6["learner"] == l]["r_y0"].mean()          for l in L6]
ge = [mean_se(sub6[sub6["learner"] == l]["r_y0"])         for l in L6]
lm = [sub6[sub6["learner"] == l]["r_y0_local_05"].mean()  for l in L6]
le = [mean_se(sub6[sub6["learner"] == l]["r_y0_local_05"])for l in L6]

ax.bar(x - w/2, gm, w, color="#0072B2", alpha=0.85,
       label=r"Global $R_{Y0}$", edgecolor="white")
ax.bar(x + w/2, lm, w, color="#CC79A7", alpha=0.85,
       label=r"Local $R_{Y0}\ (\hat{m}>0.5)$", edgecolor="white")
ax.errorbar(x - w/2, gm, yerr=ge, fmt="none",
            color="#111", linewidth=1.5, capsize=4, capthick=1.5)
ax.errorbar(x + w/2, lm, yerr=le, fmt="none",
            color="#111", linewidth=1.5, capsize=4, capthick=1.5)
ax.set_xticks(x)
ax.set_xticklabels([LABEL[l] for l in L6], fontsize=9, rotation=12)
ax.set_ylabel(r"Control arm $R^2$")
ax.set_ylim(0, 1)
ax.legend(frameon=True, framealpha=0.9, edgecolor="#cccccc", fontsize=9)
ax.set_title(r"Global vs.\ Local $R_{Y0}$"
             "\n" r"(Nonlinear, $\alpha=5$)")

# Panel (b): gap vs coverage scatter
ax = axes[1]
for lrn in L6:
    cell = sub6[sub6["learner"] == lrn]
    gap  = cell["r_y0"] - cell["r_y0_local_05"]
    gm2  = gap.mean();            ge2 = mean_se(gap)
    cm   = cell["covers"].mean(); ce  = cov_se(cm, len(cell))
    ax.errorbar(gm2, cm, xerr=ge2, yerr=ce,
                color=C[lrn], marker=MK[lrn], markersize=11,
                linewidth=0, elinewidth=2, capsize=5, capthick=1.5,
                label=LABEL[lrn], zorder=3)

ax.set_xlabel(r"Gap: global $-$ local $R_{Y0}$", labelpad=4)
ax.set_ylabel("Empirical 95% CI coverage")
ax.yaxis.set_major_formatter(pct)
ax.set_title("Coverage vs.\ Outcome Model Gap"
             "\n" r"(Nonlinear, $\alpha=5$)")
ax.legend(frameon=True, framealpha=0.9, edgecolor="#cccccc",
          fontsize=8, loc="upper right")

plabel(axes[0], "(a)", dx=-0.15)
plabel(axes[1], "(b)", dx=-0.15)
save("fig6_averaging_barrier")


# ===========================================================================
# FIG 7  Bias, RMSE, Coverage  —  3 × 2 panel
# ===========================================================================
print("Fig 7: bias / RMSE / coverage panel...")
L7 = ["lasso", "xgboost", "lasso_logistic"]

fig, axes = plt.subplots(3, 2, figsize=(10, 10))
fig.subplots_adjust(hspace=0.44, wspace=0.30)

row_titles = ["Bias", "RMSE", "Coverage"]

for col, surf in enumerate(["linear", "nonlinear"]):
    sub = v7[v7["complexity"] == surf]

    for lrn in L7:
        cell = sub[sub["learner"] == lrn]

        # --- Row 0: Bias ---
        ax = axes[0][col]
        xs, ys, es = [], [], []
        for s in S7:
            c = cell[cell["overlap_strength"] == s]["estimate"]
            xs.append(s); ys.append(c.mean() - TRUE_ATE); es.append(mean_se(c))
        ax.errorbar(xs, ys, yerr=es,
                    color=C[lrn], linestyle=LS[lrn], marker=MK[lrn],
                    linewidth=2, markersize=6, capsize=3, capthick=1.2,
                    label=LABEL[lrn])

        # --- Row 1: RMSE ---
        ax = axes[1][col]
        xs, ys, es = [], [], []
        for s in S7:
            c = cell[cell["overlap_strength"] == s]
            rmse = np.sqrt(c["sq_error"].mean())
            xs.append(s); ys.append(rmse); es.append(rmse_boot_se(c["sq_error"]))
        ax.errorbar(xs, ys, yerr=es,
                    color=C[lrn], linestyle=LS[lrn], marker=MK[lrn],
                    linewidth=2, markersize=6, capsize=3, capthick=1.2,
                    label=LABEL[lrn])

        # --- Row 2: Coverage ---
        ax = axes[2][col]
        xs, ys, es = [], [], []
        for s in S7:
            c = cell[cell["overlap_strength"] == s]["covers"]
            p = c.mean()
            xs.append(s); ys.append(p); es.append(cov_se(p, len(c)))
        ax.errorbar(xs, ys, yerr=es,
                    color=C[lrn], linestyle=LS[lrn], marker=MK[lrn],
                    linewidth=2, markersize=6, capsize=3, capthick=1.2,
                    label=LABEL[lrn])

    # Decoration
    axes[0][col].axhline(0, color="#555", linewidth=1.1, linestyle=":", alpha=0.7)
    axes[2][col].axhline(0.95, color="#555", linewidth=1.1, linestyle=":", alpha=0.7)
    axes[2][col].yaxis.set_major_formatter(pct)
    for row in range(3):
        axes[row][col].set_xticks(S7)
    axes[0][col].set_title(SURF[surf])
    axes[2][col].set_xlabel(r"Overlap strength $\alpha$", labelpad=4)

axes[0][0].set_ylabel("Bias")
axes[1][0].set_ylabel("RMSE")
axes[2][0].set_ylabel("Coverage")
axes[0][0].legend(loc="lower left", frameon=True, framealpha=0.9,
                  edgecolor="#cccccc", fontsize=9)

panels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
flat   = [axes[r][c] for r in range(3) for c in range(2)]
for ax, ltr in zip(flat, panels):
    plabel(ax, ltr, dx=-0.15)

save("fig7_bias_rmse_coverage")


# ===========================================================================
# FIG 8  PLR--IRM AUC decomposition  —  3-bar grouped plot
# ===========================================================================
print("Fig 8: PLR--IRM AUC decomposition...")

SHARED_ALPHA = [0.0, 2.0, 5.0]

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
fig.subplots_adjust(wspace=0.08)

bar_labels = [
    "PLR\n(incl. Lasso)",
    "PLR\n(excl. Lasso)",
    "IRM",
]
bar_colors  = ["#E69F00", "#E69F00", "#0072B2"]
bar_alphas  = [0.45,       0.95,      0.95]
bar_hatches = ["///",      "",        ""]

for ax, surf in zip(axes, ["linear", "nonlinear"]):
    plr_all  = v7[v7["complexity"] == surf]
    plr_cls  = v7[(v7["complexity"] == surf) &
                  (v7["learner"].isin(["lasso_logistic", "xgboost"])) &
                  (v7["overlap_strength"].isin(SHARED_ALPHA))]
    irm_grid = v10[(v10["complexity"] == surf) &
                   (v10["overlap_strength"].isin(SHARED_ALPHA))]

    aucs = [
        auc_bootstrap(plr_all,  ["r_d", "r_y"]),
        auc_bootstrap(plr_cls,  ["r_d", "r_y"]),
        auc_bootstrap(irm_grid, ["r_d", "r_y0"]),
    ]
    means  = [a[0] for a in aucs]
    err_lo = [a[0] - a[1] for a in aucs]
    err_hi = [a[2] - a[0] for a in aucs]
    x = np.arange(len(means))

    for i, (m, col, al, h) in enumerate(zip(means, bar_colors, bar_alphas, bar_hatches)):
        ax.bar(x[i], m, color=col, alpha=al, width=0.55,
               hatch=h, edgecolor="white", linewidth=0.8, zorder=3)
    ax.errorbar(x, means, yerr=[err_lo, err_hi],
                fmt="none", color="#111", linewidth=1.8,
                capsize=5, capthick=1.8, zorder=4)

    # Bracket annotation: residual IRM advantage (bar 1 → bar 2)
    y_top = max(means[1], means[2]) + max(err_hi[1], err_hi[2]) + 0.02
    ax.annotate("", xy=(x[2], y_top), xytext=(x[1], y_top),
                arrowprops=dict(arrowstyle="<->", color="#333", lw=1.5))
    diff = means[2] - means[1]
    ax.text((x[1] + x[2]) / 2, y_top + 0.012,
            f"residual IRM\n+{diff:+.3f}", ha="center", va="bottom",
            fontsize=8, color="#333")

    ax.axhline(0.5, color="#888", linewidth=1.0, linestyle=":", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, fontsize=9)
    ax.set_ylim(0.40, 1.05)
    ax.set_title(SURF[surf])

axes[0].set_ylabel("AUC (predicting coverage failure)")

legend_els = [
    mpatches.Patch(facecolor="#E69F00", alpha=0.45, hatch="///",
                   label="PLR (all learners, all $\\alpha$)"),
    mpatches.Patch(facecolor="#E69F00", alpha=0.95,
                   label="PLR (classifier learners, $\\alpha\\in\\{0,2,5\\}$)"),
    mpatches.Patch(facecolor="#0072B2", alpha=0.95,
                   label="IRM ($\\alpha\\in\\{0,2,5\\}$)"),
]
axes[1].legend(handles=legend_els, loc="lower right",
               frameon=True, framealpha=0.9, edgecolor="#cccccc", fontsize=8)
plabel(axes[0], "(a)"); plabel(axes[1], "(b)")
save("fig8_composition_decomposition")

print("\nDone. All figures in output/figures/paper/")

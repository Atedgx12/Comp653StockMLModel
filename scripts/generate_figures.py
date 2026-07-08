"""Generate result figures for the COMP653 presentation."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = r"B:\Rice\Comp653(Summer2026)\Module3\homework\stock_model\figures"
os.makedirs(OUT, exist_ok=True)

NAVY = "#1F2840"
BLUE = "#1B4F9C"
TEAL = "#2E8B8B"
GREY = "#9AA3B2"
RED  = "#B03A2E"

plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 19,
    "axes.titleweight": "bold",
    "axes.labelsize": 15,
    "axes.edgecolor": NAVY,
    "axes.labelcolor": NAVY,
    "xtick.color": NAVY,
    "ytick.color": NAVY,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

W, H = 11.0, 6.2   # inches, 16:9 friendly


def save(fig, name):
    fig.tight_layout()
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("saved", p)


# 1. Volatility term structure: per-horizon AUC (coupled model).
horizons = [1, 5, 10, 30, 90, 180]
auc_coupled = [0.7045, 0.8843, 0.9503, 0.9754, 0.9696, 0.9694]
fig, ax = plt.subplots(figsize=(W, H))
ax.plot(range(len(horizons)), auc_coupled, "-o", color=BLUE, lw=3, ms=11,
        markerfacecolor="white", markeredgewidth=2.5, markeredgecolor=BLUE)
for x, y in zip(range(len(horizons)), auc_coupled):
    ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                xytext=(0, 12), ha="center", fontsize=13, color=NAVY,
                fontweight="bold")
ax.axhline(0.5, color=GREY, ls="--", lw=1.5)
ax.text(0.05, 0.505, "coin flip", color=GREY, fontsize=12, va="bottom")
ax.set_xticks(range(len(horizons)))
ax.set_xticklabels([f"{h}d" for h in horizons])
ax.set_ylim(0.45, 1.02)
ax.set_xlabel("Forecast horizon")
ax.set_ylabel("Held-out AUC")
ax.set_title("Volatility Term Structure: Predictability Rises With Horizon")
ax.grid(axis="y", alpha=0.25)
save(fig, "fig_term_structure.png")


# 2. Direction vs Volatility: honest leak-free AUC.
fig, ax = plt.subplots(figsize=(W, H))
groups = ["Return direction", "Volatility"]
cpcv = [0.4721, 0.9888]
test = [0.5763, 0.9921]
x = np.arange(len(groups)); w = 0.34
ax.bar(x - w/2, cpcv, w, label="Purged CPCV mean", color=GREY)
ax.bar(x + w/2, test, w, label="Purged held-out test", color=BLUE)
for xi, (a, b) in enumerate(zip(cpcv, test)):
    ax.text(xi - w/2, a + 0.01, f"{a:.2f}", ha="center", fontweight="bold", color=NAVY)
    ax.text(xi + w/2, b + 0.01, f"{b:.2f}", ha="center", fontweight="bold", color=NAVY)
ax.axhline(0.5, color=RED, ls="--", lw=1.5)
ax.set_xticks(x); ax.set_xticklabels(groups)
ax.set_ylim(0.0, 1.08)
ax.set_ylabel("AUC")
ax.set_title("Same Pipeline, Two Targets: Direction Is Efficient, Volatility Is Not")
ax.legend(loc="upper left", frameon=False)
save(fig, "fig_direction_vs_vol.png")


# 3. CPCV path distribution for volatility.
paths = [0.9863, 0.9954, 0.9981, 0.9921, 0.9793,
         0.9867, 0.9740, 0.9937, 0.9903, 0.9924]
fig, ax = plt.subplots(figsize=(W, H))
ax.bar(range(1, 11), paths, color=TEAL)
ax.axhline(np.mean(paths), color=NAVY, lw=2,
           label=f"mean {np.mean(paths):.4f}")
ax.set_ylim(0.9, 1.0)
ax.set_xticks(range(1, 11))
ax.set_xlabel("Combinatorial purged CV path")
ax.set_ylabel("AUC")
ax.set_title("Volatility Model Is Stable Across All 10 Purged CV Paths")
ax.legend(loc="lower right", frameon=False)
save(fig, "fig_cpcv_paths.png")


# 4. Mutual information: volatility vs direction top features.
fig, ax = plt.subplots(figsize=(W, H))
feats = ["vol252", "vol120", "vol60", "vol20",
         "sector_vs_market\n(direction)"]
mis = [0.7686, 0.7598, 0.7177, 0.5768, 0.0165]
colors = [BLUE, BLUE, BLUE, BLUE, RED]
ax.barh(range(len(feats)), mis, color=colors)
for i, m in enumerate(mis):
    ax.text(m + 0.01, i, f"{m:.3f}", va="center", fontweight="bold", color=NAVY)
ax.set_yticks(range(len(feats)))
ax.set_yticklabels(feats)
ax.invert_yaxis()
ax.set_xlim(0, 0.85)
ax.set_xlabel("Mutual information with label (bits)")
ax.set_title("Volatility Carries ~70x the Signal of Direction")
save(fig, "fig_mutual_information.png")

print("all figures generated")

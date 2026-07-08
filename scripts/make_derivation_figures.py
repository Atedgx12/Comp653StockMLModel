"""
Generate figures and worked numerical examples for model_derivations.tex.

Every figure is written to figures/derivations/ and every printed block is a
worked example that is transcribed verbatim into the LaTeX document, so the
narrative math, the numbers, and the charts all agree.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "figures", "derivations")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "font.size": 11,
                     "axes.grid": True, "grid.alpha": 0.3})

# Measured results carried in from the training runs -----------------------
HZ_D = np.array([1, 5, 10, 30, 90, 180])
VOL_D = np.array([0.01364, 0.01661, 0.01744, 0.01825, 0.01884, 0.01918])
AUC_D = np.array([0.6215, 0.7393, 0.7846, 0.8373, 0.8699, 0.8779])
HZ_I = np.array([5, 15, 30, 60, 120, 240])
AUC_I = np.array([0.6659, 0.7386, 0.8160, 0.8688, 0.9030, 0.9207])


def banner(t):
    print("\n" + "=" * 68 + f"\n{t}\n" + "=" * 68)


# 0. Architecture / pipeline diagram ----------------------------------------
def fig_architecture():
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(7.6, 8.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 15); ax.axis("off")

    def box(x, y, w, h, text, color):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                           linewidth=1.2, edgecolor="#333333", facecolor=color)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=9, wrap=True)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                     arrowstyle="-|>", mutation_scale=14,
                     linewidth=1.1, color="#555555"))

    blue, green, orange, purple, gray = ("#c6dbef", "#c7e9c0", "#fdd0a2",
                                         "#dadaeb", "#e0e0e0")
    box(2.5, 13.6, 5.0, 0.9, "S&P 500 prices\n(daily bars + 5m intraday bars)", gray)
    box(2.5, 12.2, 5.0, 0.9, "Features: log returns, realized volatility,\n"
        "cross-sectional rank, volume, range, time of day", blue)
    box(2.5, 10.8, 5.0, 0.9, "Mutual information feature selection", blue)
    # split
    box(0.3, 8.7, 4.4, 1.5, "Unified Network\nlogistic + naive Bayes + MLP\n"
        "+ sentiment + LSTM\n-> softmax meta layer", green)
    box(5.3, 8.7, 4.4, 1.5, "Multi-scale term structure\nsix window LSTM branches\n"
        "-> drift fusion -> shared trunk\n-> six volatility heads", orange)
    box(5.3, 7.0, 4.4, 1.0, "Quantile band heads\n(non-crossing, pinball loss)", orange)
    box(5.3, 5.5, 4.4, 1.0, "Conformal calibration\n(coverage guarantee)", purple)
    box(2.5, 3.9, 5.0, 1.0, "Ensemble\nAUC-weighted blend of both models", green)
    box(2.5, 2.4, 5.0, 1.0, "Evaluation\nAUC by horizon, coverage, price cone", gray)

    arrow(5, 13.6, 5, 13.1); arrow(5, 12.2, 5, 11.7)
    arrow(4.2, 10.8, 2.5, 10.2); arrow(5.8, 10.8, 7.5, 10.2)
    arrow(7.5, 8.7, 7.5, 8.0); arrow(7.5, 7.0, 7.5, 6.5)
    arrow(2.5, 8.7, 4.0, 4.9); arrow(7.5, 5.5, 6.0, 4.9)
    arrow(5, 3.9, 5, 3.4)
    ax.set_title("Model architecture and data flow", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "architecture.png")); plt.close(fig)


# 0b. Lifecycle / workflow diagram ------------------------------------------
def fig_lifecycle():
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(8.6, 8.8))
    ax.set_xlim(0, 12); ax.set_ylim(0, 15); ax.axis("off")

    def box(x, y, w, h, text, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                     linewidth=1.2, edgecolor="#333333", facecolor=color))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5)

    def arrow(x0, y0, x1, y1, color="#555555", rad=0.0, style="-|>"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                     mutation_scale=13, linewidth=1.1, color=color,
                     connectionstyle=f"arc3,rad={rad}"))

    blue, green, orange, purple, gray, red = ("#c6dbef", "#c7e9c0", "#fdd0a2",
                                              "#dadaeb", "#e0e0e0", "#fcbba1")
    box(4.2, 13.7, 5, 0.8, "1. Data: download and cache prices\n"
        "(daily + intraday), filter universe", gray)
    box(4.2, 12.5, 5, 0.8, "2. Features + mutual information selection", blue)
    box(4.2, 11.3, 5, 0.8, "3. Purged split: train / validation / test\n"
        "with embargo (CPCV)", blue)

    # training loop block, shifted right to keep the left margin clear
    box(2.2, 6.2, 6.0, 4.5, "", "#f7fbff")
    ax.text(5.2, 10.35, "4. Training loop (per epoch)", ha="center",
            fontsize=9.5, weight="bold")
    box(2.6, 9.1, 5.2, 0.7, "forward pass -> loss\n(cross entropy + curvature + pinball)", green)
    box(2.6, 8.1, 5.2, 0.7, "backward pass (BPTT + trunk)", green)
    box(2.6, 7.1, 5.2, 0.7, "Adam update", green)
    box(2.6, 6.35, 5.2, 0.6, "validation CE / accuracy", orange)

    box(8.7, 7.6, 3.1, 1.7, "5. Feedback\nearly stopping,\n"
        "reduce-on-plateau LR,\ndrift + curvature\nbackprop", red)
    box(4.2, 4.7, 5, 0.9, "6. Test on held-out set\nAUC by horizon, coverage,\n"
        "naive + LightGBM baselines", purple)
    box(4.2, 3.2, 5, 0.9, "7. Report: term structure, price cone,\n"
        "save model checkpoints", gray)
    box(4.2, 1.6, 5, 0.9, "8. Tune and modify: hyperparameters,\n"
        "features, ensemble weights, warm start", orange)

    arrow(6.7, 13.7, 6.7, 13.3); arrow(6.7, 12.5, 6.7, 12.1)
    arrow(6.7, 11.3, 6.7, 10.75)
    # inner loop
    arrow(5.2, 9.1, 5.2, 8.8); arrow(5.2, 8.1, 5.2, 7.8); arrow(5.2, 7.1, 5.2, 6.95)
    arrow(7.8, 9.45, 5.3, 9.85, color="#3182bd", rad=-0.35)   # loop back up
    # feedback to and from the training loop
    arrow(7.8, 6.65, 8.9, 7.9, color="#cb181d", rad=0.2)
    arrow(8.9, 8.6, 7.8, 9.5, color="#cb181d", rad=0.2)
    arrow(5.2, 6.2, 5.6, 5.6)
    arrow(6.7, 4.7, 6.7, 4.1); arrow(6.7, 3.2, 6.7, 2.5)
    # modify loops back up the clear left margin to the data stage
    arrow(4.2, 2.05, 0.7, 2.05, color="#e6550d")
    arrow(0.7, 2.05, 0.7, 14.1, color="#e6550d")
    arrow(0.7, 14.1, 4.2, 14.1, color="#e6550d")
    ax.text(0.95, 8.0, "iterate", rotation=90, color="#e6550d", fontsize=9)
    ax.set_title("Model lifecycle: data, training loop, validation,\n"
                 "feedback, testing, reporting, tuning", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "lifecycle.png")); plt.close(fig)


# 0c. Multi-scale fusion graph ----------------------------------------------
def fig_fusion_graph():
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 8); ax.axis("off")

    def box(x, y, w, h, text, color, fs=9):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                     linewidth=1.2, edgecolor="#333333", facecolor=color))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(x0, y0, x1, y1, color="#555555", rad=0.0):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                     mutation_scale=12, linewidth=1.1, color=color,
                     connectionstyle=f"arc3,rad={rad}"))

    windows = [1, 5, 10, 30, 90, 180]
    green, orange, blue, gray = ("#c7e9c0", "#fdd0a2", "#c6dbef", "#e0e0e0")
    xs = np.linspace(0.6, 10.0, 6)
    bw = 1.35
    # branch embedding nodes
    for i, (x, w) in enumerate(zip(xs, windows)):
        box(x, 1.0, bw, 0.9, f"LSTM {w}d\n$e_{{{i+1}}}$", green, fs=8.5)
    # drift nodes between adjacent branches
    for k in range(5):
        xm = (xs[k] + xs[k+1]) / 2 + bw / 2
        box(xm - 0.55, 2.5, 1.1, 0.7,
            f"$\\Delta_{{{k+1}}}=e_{{{k+2}}}-e_{{{k+1}}}$", orange, fs=7)
        arrow(xs[k] + bw, 1.9, xm - 0.1, 2.5, color="#e6550d", rad=0.1)
        arrow(xs[k+1], 1.9, xm + 0.1, 2.5, color="#e6550d", rad=-0.1)
    # fusion bar
    box(1.5, 4.1, 9.0, 0.8, "fuse = concat($e_1,\\dots,e_6,\\ \\Delta_1,\\dots,\\Delta_5$)",
        blue, fs=9)
    for x in xs:
        arrow(x + bw / 2, 1.9, x + bw / 2, 4.1, color="#3182bd", rad=0.0)
    for k in range(5):
        xm = (xs[k] + xs[k+1]) / 2 + bw / 2
        arrow(xm, 3.2, xm, 4.1, color="#e6550d")
    box(3.0, 5.4, 6.0, 0.8, "shared trunk (128, 64)", gray, fs=9)
    arrow(6.0, 4.9, 6.0, 5.4)
    box(1.5, 6.7, 4.0, 0.8, "6 volatility heads\n(curvature coupled)", green, fs=8.5)
    box(6.5, 6.7, 4.0, 0.8, "quantile band heads\n(non-crossing + conformal)", orange, fs=8.5)
    arrow(4.5, 6.2, 3.5, 6.7); arrow(7.5, 6.2, 8.5, 6.7)
    ax.set_title("Multi-scale fusion graph: branch embeddings linked by drift",
                 fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fusion_graph.png")); plt.close(fig)


# 0d. Drift between scales, computed from the trained daily model ------------
def fig_drift():
    banner("Drift visualization from the trained daily model")
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ucn.models.multiscale import MultiScaleTermStructureNet
    from ucn.backend import to_device, to_cpu
    windows = [1, 5, 10, 30, 90, 180]
    cand = ["multiscale_daily.npz",
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "multiscale_daily.npz"),
            r"B:\Rice\Comp653StockMLModel\models\multiscale_daily.npz"]
    net = None; trained = False
    for p in cand:
        if os.path.exists(p):
            try:
                net = MultiScaleTermStructureNet.load(p); trained = True; break
            except Exception as e:
                print("  load failed:", e); net = None
    if net is None:
        net = MultiScaleTermStructureNet(windows=windows, hidden=24,
                                         trunk_sizes=(128, 64))
        net._init_weights(7)
    B = net.B
    d = net.scalers[0][0].shape[0] if net.scalers else 7
    steps = [min(w, 20) for w in net.windows]
    rng = np.random.default_rng(0)
    N = 512
    vol = rng.uniform(0.5, 2.0, (N, 1, 1)).astype(np.float32)
    seqs = [rng.standard_normal((N, steps[b], d)).astype(np.float32) * vol
            for b in range(B)]
    seq_dev = [to_device(s) for s in seqs]
    if net.scalers:
        seq_dev = net._apply_scalers(seq_dev)
    c = net._forward(seq_dev, training=False)
    embs = [np.asarray(to_cpu(e)) for e in c["embs"]]
    mean_emb = np.stack([e.mean(0) for e in embs])
    drift_norm = [float(np.mean(np.linalg.norm(embs[k+1] - embs[k], axis=1)))
                  for k in range(B - 1)]
    print(f"  trained={trained}  drift norms="
          f"{[round(x, 3) for x in drift_norm]}")

    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.9))
    im = ax[0].imshow(mean_emb, aspect="auto", cmap="RdBu_r")
    ax[0].set_yticks(range(B)); ax[0].set_yticklabels([f"{w}d" for w in net.windows])
    ax[0].set_xlabel("hidden unit"); ax[0].set_ylabel("branch time window")
    ax[0].set_title("Mean branch embeddings $e_b$")
    fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)
    labels = [f"{net.windows[k]}$\\to${net.windows[k+1]}d" for k in range(B - 1)]
    ax[1].bar(labels, drift_norm, color="#e6550d")
    ax[1].set_ylabel(r"mean drift magnitude $\|\Delta_k\|$")
    ax[1].set_title("Drift between adjacent scales")
    ax[1].tick_params(axis="x", rotation=25)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "drift.png")); plt.close(fig)


# 0e. Per-branch overview figures (structure + training-loop position + data)
def _branch_flow(ax, blocks, grad_label):
    """Draw a horizontal data-flow chain into the meta layer with a backward
    gradient arrow underneath, showing the branch position in the loop."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis("off")
    cols = ["#c6dbef", "#c7e9c0", "#fdd0a2", "#dadaeb", "#e0e0e0"]
    n = len(blocks); h = 0.95; y = 1.55
    meta_w = 1.05; tail = 0.5 + meta_w
    gap = 0.5; x0 = 0.25
    w = (9.6 - x0 - tail - (n - 1) * gap) / n
    xs = []
    for i, b in enumerate(blocks):
        bx = x0 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((bx, y), w, h, boxstyle="round,pad=0.04",
                     lw=1.1, edgecolor="#333", facecolor=cols[i % len(cols)]))
        ax.text(bx + w / 2, y + h / 2, b, ha="center", va="center", fontsize=8)
        xs.append((bx, bx + w))
        if i > 0:
            ax.add_patch(FancyArrowPatch((xs[i-1][1], y+h/2), (bx, y+h/2),
                         arrowstyle="-|>", mutation_scale=11, color="#555"))
    lx = xs[-1][1]
    ax.add_patch(FancyArrowPatch((lx, y+h/2), (lx+0.5, y+h/2),
                 arrowstyle="-|>", mutation_scale=11, color="#555"))
    ax.add_patch(FancyBboxPatch((lx+0.55, y+0.1), meta_w, h-0.2,
                 boxstyle="round,pad=0.04", lw=1.1, edgecolor="#333",
                 facecolor="#fee0d2"))
    ax.text(lx+0.55+meta_w/2, y+h/2, "meta\n$\\to$ loss", ha="center",
            va="center", fontsize=7.5)
    ax.add_patch(FancyArrowPatch((lx, y-0.05), (x0, y-0.05), arrowstyle="-|>",
                 mutation_scale=12, color="#cb181d",
                 connectionstyle="arc3,rad=0.32"))
    ax.text((x0+lx)/2, y-0.95, f"backward gradient: {grad_label}", ha="center",
            color="#cb181d", fontsize=8)


def _phi(x):
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)


def fig_branch_lr():
    fig, ax = plt.subplots(2, 1, figsize=(7.0, 4.8),
                           gridspec_kw={"height_ratios": [1, 1.2]})
    _branch_flow(ax[0], ["features $x$", r"$z=w^\top x+b$", r"$\sigma(z)$",
                         "output $a_{lr}$"], r"$(p-y)\,x$")
    ax[0].set_title("Branch A: logistic regression, structure and loop position",
                    fontsize=10)
    z = np.linspace(-6, 6, 400); s = 1/(1+np.exp(-z))
    ax[1].plot(z, s, color="#1f77b4", lw=2)
    for zz in (-2, 0, 2):
        pp = 1/(1+np.exp(-zz)); ax[1].scatter([zz], [pp], color="#d62728", zorder=5)
        ax[1].annotate(f"{pp:.3f}", (zz, pp), textcoords="offset points", xytext=(6, -10))
    ax[1].axhline(0.5, color="gray", ls="--", lw=0.8)
    ax[1].set_xlabel("score z"); ax[1].set_ylabel("probability")
    ax[1].set_title("Data view: the logistic map turns the score into a probability")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "branch_lr.png")); plt.close(fig)


def fig_branch_nb():
    fig, ax = plt.subplots(2, 1, figsize=(7.0, 4.8),
                           gridspec_kw={"height_ratios": [1, 1.2]})
    _branch_flow(ax[0], ["features $x$", r"whiten $(x-\mu)/s$",
                         r"$W\tilde x+b$", r"$\sigma(\cdot)$", "$a_{nb}$"],
                 r"$(a_{nb}-y)$")
    ax[0].set_title("Branch B: Gaussian naive Bayes, structure and loop position",
                    fontsize=10)
    x = np.linspace(-5, 6, 500)
    m0, m1, s = -1.0, 1.6, 1.0
    ax[1].plot(x, _phi((x-m0)/s), color="#3182bd", lw=2, label="class 0 density")
    ax[1].plot(x, _phi((x-m1)/s), color="#e6550d", lw=2, label="class 1 density")
    post = 1/(1+np.exp(-((m1-m0)/s**2)*x + (m1**2-m0**2)/(2*s**2)))
    axb = ax[1].twinx()
    axb.plot(x, post, color="#238b45", lw=2, ls="--", label=r"posterior $P(y{=}1\mid x)$")
    axb.set_ylabel("posterior", color="#238b45")
    ax[1].set_xlabel("feature x"); ax[1].set_ylabel("density")
    ax[1].set_title("Data view: two Gaussians give a logistic (linear log-odds) posterior")
    ax[1].legend(loc="upper left", fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "branch_nb.png")); plt.close(fig)


def fig_branch_mlp():
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ucn.models.mlp import MLPScratch
    fig, ax = plt.subplots(2, 1, figsize=(7.0, 5.2),
                           gridspec_kw={"height_ratios": [1, 1.5]})
    _branch_flow(ax[0], ["features $x$", "dense + ReLU", "dense + ReLU",
                         "output $a_{mlp}$"], r"$\delta_Z=\delta_A\odot\mathbf{1}[Z>0]$")
    ax[0].set_title("Branch C: multilayer perceptron, structure and loop position",
                    fontsize=10)
    # Two interleaving classes, then train the project's own ReLU MLP branch.
    rng = np.random.default_rng(1)
    m = 300
    t = rng.uniform(0, np.pi, m)
    x0 = np.c_[np.cos(t), np.sin(t)] + rng.normal(0, 0.11, (m, 2))
    x1 = np.c_[1 - np.cos(t), 0.5 - np.sin(t)] + rng.normal(0, 0.11, (m, 2))
    X = np.vstack([x0, x1]); y = np.r_[np.zeros(m), np.ones(m)].astype(int)
    net = MLPScratch(hidden_sizes=(24, 16), lr=0.08, epochs=800,
                     batch_size=128, verbose=0)
    net.fit(X, y)
    acc = float((net.predict(X) == y).mean())
    print(f"  branch C real MLP train accuracy = {acc:.3f}")
    xr = np.linspace(X[:, 0].min() - 0.4, X[:, 0].max() + 0.4, 220)
    yr = np.linspace(X[:, 1].min() - 0.4, X[:, 1].max() + 0.4, 220)
    XX, YY = np.meshgrid(xr, yr)
    P = net.predict_proba(np.c_[XX.ravel(), YY.ravel()])[:, 1].reshape(XX.shape)
    ax[1].contourf(XX, YY, P, levels=20, cmap="RdBu_r", alpha=0.75)
    ax[1].contour(XX, YY, P, levels=[0.5], colors="k", linewidths=1.6)
    ax[1].scatter(x0[:, 0], x0[:, 1], s=9, color="#08519c", edgecolor="w", lw=0.2,
                  label="class 0")
    ax[1].scatter(x1[:, 0], x1[:, 1], s=9, color="#a63603", edgecolor="w", lw=0.2,
                  label="class 1")
    ax[1].set_xlabel("feature 1"); ax[1].set_ylabel("feature 2")
    ax[1].set_title(f"Data view: real trained ReLU branch boundary "
                    f"(train accuracy {acc:.2f})")
    ax[1].legend(loc="upper right", fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "branch_mlp.png")); plt.close(fig)


def fig_branch_sent():
    fig, ax = plt.subplots(2, 1, figsize=(7.0, 4.8),
                           gridspec_kw={"height_ratios": [1, 1.2]})
    _branch_flow(ax[0], ["VADER score", r"$W_s\,s+b_s$", r"$\sigma(\cdot)$", "$a_s$"],
                 r"$(a_s-y)$")
    ax[0].set_title("Branch D: sentiment, structure and loop position", fontsize=10)
    s = np.linspace(-1, 1, 400)
    ax[1].plot(s, 1/(1+np.exp(-3.0*s)), color="#756bb1", lw=2)
    ax[1].axvspan(-0.05, 0.05, color="gray", alpha=0.25,
                  label="most history sits near zero")
    ax[1].set_xlabel("VADER compound sentiment"); ax[1].set_ylabel("branch output")
    ax[1].set_title("Data view: logistic map of sentiment, sparse before recent years")
    ax[1].legend(loc="upper left", fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "branch_sent.png")); plt.close(fig)


def fig_branch_lstm():
    fig, ax = plt.subplots(2, 1, figsize=(7.0, 4.8),
                           gridspec_kw={"height_ratios": [1, 1.2]})
    _branch_flow(ax[0], [r"sequence $x_{1..T}$", "gates $f,i,g,o$",
                         r"cell $c_t$", r"hidden $h_T$"],
                 r"BPTT through time")
    ax[0].set_title("Branch E: LSTM, structure and loop position", fontsize=10)
    # Simulate one LSTM cell on a bursty input to show the cell carrying memory.
    T = 60
    rng = np.random.default_rng(2)
    x = rng.normal(0, 0.3, T)
    x[20:26] += 2.0                              # a volatility burst
    c = 0.0; h = 0.0; cs = []; hs = []
    for t in range(T):
        f = 1/(1+np.exp(-(1.5)))                 # high forget: keep memory
        i = 1/(1+np.exp(-(1.0*abs(x[t])-0.5)))
        g = np.tanh(x[t])
        o = 1/(1+np.exp(-(1.0)))
        c = f*c + i*g; h = o*np.tanh(c)
        cs.append(c); hs.append(h)
    ax[1].plot(x, color="#bdbdbd", lw=1, label="input $x_t$")
    ax[1].plot(cs, color="#e6550d", lw=2, label="cell state $c_t$")
    ax[1].plot(hs, color="#3182bd", lw=2, label="hidden $h_t$")
    ax[1].axvspan(20, 26, color="#fee6ce", alpha=0.7, label="volatility burst")
    ax[1].set_xlabel("time step in the lookback window"); ax[1].set_ylabel("activation")
    ax[1].set_title("Data view: the cell state carries the burst forward as memory")
    ax[1].legend(loc="upper right", fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "branch_lstm.png")); plt.close(fig)


# 1. Logistic sigmoid with worked points ------------------------------------
def fig_sigmoid():
    banner("Worked example: logistic branch")
    z = np.linspace(-6, 6, 400)
    s = 1 / (1 + np.exp(-z))
    pts = np.array([-2.0, 0.0, 2.0])
    ps = 1 / (1 + np.exp(-pts))
    for zz, pp in zip(pts, ps):
        bce = -(1 * np.log(pp))
        print(f"  z={zz:+.1f}  p=sigma(z)={pp:.4f}  BCE(y=1)={bce:.4f}"
              f"  sigma'={pp*(1-pp):.4f}")
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.plot(z, s, color="#1f77b4", lw=2, label=r"$\sigma(z)=1/(1+e^{-z})$")
    ax.scatter(pts, ps, color="#d62728", zorder=5)
    for zz, pp in zip(pts, ps):
        ax.annotate(f"({zz:+.0f}, {pp:.3f})", (zz, pp),
                    textcoords="offset points", xytext=(8, -12))
    ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("score z"); ax.set_ylabel("probability p")
    ax.set_title("Logistic branch: score mapped to probability")
    ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "sigmoid.png")); plt.close(fig)


# 2. Volatility term structure ----------------------------------------------
def fig_term_structure():
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.plot(HZ_D, VOL_D, "o-", color="#2ca02c", lw=2)
    for h, v in zip(HZ_D, VOL_D):
        ax.annotate(f"{v:.4f}", (h, v), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=8)
    ax.set_xlabel("horizon (trading days)")
    ax.set_ylabel("mean forward realized volatility")
    ax.set_title("Volatility term structure of the data")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "term_structure.png")); plt.close(fig)


# 3. AUC by horizon, both scales --------------------------------------------
def fig_auc():
    banner("Curvature (second difference) on the daily AUC term structure")
    d2 = AUC_D[2:] - 2 * AUC_D[1:-1] + AUC_D[:-2]
    for i, v in enumerate(d2):
        print(f"  interior horizon {HZ_D[i+1]:>3}d  "
              f"second difference={v:+.4f}")
    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.5), sharey=True)
    ax[0].plot(HZ_D, AUC_D, "s-", color="#1f77b4", lw=2)
    ax[0].set_title("Daily scale"); ax[0].set_xlabel("horizon (days)")
    ax[0].set_ylabel("test AUC"); ax[0].axhline(0.5, color="gray", ls="--", lw=0.8)
    ax[1].plot(HZ_I, AUC_I, "^-", color="#ff7f0e", lw=2)
    ax[1].set_title("Intraday scale (5m bars)"); ax[1].set_xlabel("horizon (minutes)")
    ax[1].axhline(0.5, color="gray", ls="--", lw=0.8)
    fig.suptitle("Predictability rises with horizon")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "auc_horizon.png")); plt.close(fig)


# 4. Pinball loss and the empirical minimizer -------------------------------
def fig_pinball():
    banner("Worked example: pinball loss minimizer equals the sample quantile")
    rng = np.random.default_rng(0)
    y = rng.normal(0.0, 1.0, 20000)
    for tau in (0.05, 0.5, 0.95):
        qs = np.linspace(-3, 3, 601)
        loss = np.array([np.mean(np.where(y - q > 0, tau * (y - q),
                                          (tau - 1) * (y - q))) for q in qs])
        q_star = qs[np.argmin(loss)]
        print(f"  tau={tau:.2f}  argmin pinball q*={q_star:+.3f}  "
              f"empirical quantile={np.quantile(y, tau):+.3f}")
    u = np.linspace(-3, 3, 400)
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for tau, col in zip((0.05, 0.5, 0.95), ("#1f77b4", "#2ca02c", "#d62728")):
        rho = np.where(u > 0, tau * u, (tau - 1) * u)
        ax.plot(u, rho, lw=2, color=col, label=fr"$\tau={tau}$")
    ax.set_xlabel(r"residual $u = y - q$"); ax.set_ylabel(r"$\rho_\tau(u)$")
    ax.set_title("Pinball loss weights the two sides differently")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "pinball.png")); plt.close(fig)


# 5. The price cone interacting with the data projection --------------------
def fig_cone():
    banner("Worked example: quantile price cone from the volatility term structure")
    H = 180
    days = np.arange(0, H + 1)
    # Per-day volatility taken from the measured term structure, which rises
    # with the horizon, so the cone widens at the same grade as the data
    # instead of using a single flat one-day volatility.
    vday = np.interp(np.arange(1, H + 1), HZ_D, VOL_D)
    var_cum = np.concatenate([[0.0], np.cumsum(vday ** 2)])
    sig_cum = np.sqrt(var_cum)                 # cumulative return std by horizon
    from math import erf, sqrt
    def zq(p):                                 # standard normal quantile
        # invert via bisection on the erf-based CDF, no scipy dependency
        lo, hi = -8.0, 8.0
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            cdf = 0.5 * (1 + erf(mid / sqrt(2)))
            if cdf < p: lo = mid
            else: hi = mid
        return 0.5 * (lo + hi)
    levels = [0.05, 0.25, 0.50, 0.75, 0.95]
    mm = [0.005, 0.995]                        # outer min/max envelope (99%)
    zvals = {p: zq(p) for p in levels + mm}
    price = {p: 100.0 * np.exp(zvals[p] * sig_cum) for p in levels + mm}

    # Data projection: simulate paths with the same rising per-day volatility,
    # so the cloud and the cone share one grade.
    rng = np.random.default_rng(7)
    M = 4000
    steps = rng.normal(0.0, 1.0, size=(M, H)) * vday[None, :]
    logp = np.cumsum(steps, axis=1)
    paths = 100.0 * np.exp(np.hstack([np.zeros((M, 1)), logp]))
    emp = {p: np.quantile(paths, p, axis=0) for p in levels}

    for p in levels:
        print(f"  {int(p*100):>2}th pct at 180d: formula "
              f"${price[p][-1]:.2f}  simulated ${emp[p][-1]:.2f}")
    print(f"  min/max range at 180d: 0.5th ${price[0.005][-1]:.2f}  "
          f"99.5th ${price[0.995][-1]:.2f}")
    cover = np.mean((paths[:, -1] >= price[0.05][-1]) &
                    (paths[:, -1] <= price[0.95][-1]))
    print(f"  fraction of paths inside the 5-95 cone at 180d: {cover:.3f}")

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for i in range(300):
        ax.plot(days, paths[i], color="#9ecae1", lw=0.35, alpha=0.5)
    ax.fill_between(days, price[0.005], price[0.995], color="#fee6ce",
                    alpha=0.6, label="model min/max range (0.5-99.5)")
    ax.fill_between(days, price[0.05], price[0.95], color="#fdae6b",
                    alpha=0.45, label="model 5-95 band")
    ax.fill_between(days, price[0.25], price[0.75], color="#e6550d",
                    alpha=0.35, label="model 25-75 band")
    ax.plot(days, price[0.50], color="#a63603", lw=2, label="model median")
    ax.plot(days, price[0.005], color="#8c6d31", lw=1.0, ls=":")
    ax.plot(days, price[0.995], color="#8c6d31", lw=1.0, ls=":")
    for p in levels:
        ax.plot(days, emp[p], color="#08519c", lw=1.1, ls="--")
    ax.plot([], [], color="#08519c", lw=1.1, ls="--", label="empirical quantiles")
    ax.set_ylim(55, 175)
    ax.set_xlabel("horizon (trading days)"); ax.set_ylabel("price of a $100 stock")
    ax.set_title("Quantile price cone against the data projection")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "price_cone.png")); plt.close(fig)


# 6. Conformal coverage ------------------------------------------------------
def fig_conformal():
    banner("Worked example: conformal widening delta and coverage")
    rng = np.random.default_rng(3)
    n = 500
    y = rng.normal(0, 1, n)
    lo, hi = -1.0, 1.0                          # deliberately narrow band
    scores = np.maximum(lo - y, y - hi)
    alpha = 0.10
    k = int(np.ceil((1 - alpha) * (n + 1)))
    delta = np.sort(scores)[min(k, n) - 1]
    delta = max(delta, 0.0)
    ytest = rng.normal(0, 1, 20000)
    cov0 = np.mean((ytest >= lo) & (ytest <= hi))
    cov1 = np.mean((ytest >= lo - delta) & (ytest <= hi + delta))
    print(f"  k={k}  delta={delta:.3f}  coverage before={cov0:.3f}  "
          f"after={cov1:.3f}  target={1-alpha:.2f}")
    # Coverage as a function of the widening, monotone and crossing the target.
    ds = np.linspace(0.0, max(scores.max(), delta * 1.6), 200)
    cov_curve = np.array([np.mean((ytest >= lo - d) & (ytest <= hi + d))
                          for d in ds])
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.7))
    ax[0].hist(scores, bins=40, color="#c6dbef", edgecolor="#3182bd")
    ax[0].axvline(delta, color="#d62728", lw=2,
                  label=fr"$\delta=s_{{(k)}}={delta:.2f}$")
    ax[0].set_xlabel("conformity score s"); ax[0].set_ylabel("count")
    ax[0].set_title("Calibration score distribution"); ax[0].legend()
    ax[1].plot(ds, cov_curve, color="#3182bd", lw=2)
    ax[1].axhline(1 - alpha, color="gray", ls="--", label="target 0.90")
    ax[1].axvline(delta, color="#d62728", lw=2, label=fr"chosen $\delta={delta:.2f}$")
    ax[1].scatter([delta], [cov1], color="#d62728", zorder=5)
    ax[1].set_xlabel(r"widening $\delta$"); ax[1].set_ylabel("out of sample coverage")
    ax[1].set_title(f"Coverage rises to target ({cov0:.2f}$\\to${cov1:.2f})")
    ax[1].legend(loc="lower right", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "conformal.png")); plt.close(fig)


def fig_conformal_bands():
    """Before and after picture of the band widening over the data points."""
    rng = np.random.default_rng(3)
    nc = 500
    yc = rng.normal(0, 1, nc)
    lo, hi = -1.0, 1.0
    scores = np.maximum(lo - yc, yc - hi)
    alpha = 0.10
    k = int(np.ceil((1 - alpha) * (nc + 1)))
    delta = max(np.sort(scores)[min(k, nc) - 1], 0.0)
    n = 140
    y = rng.normal(0, 1, n)
    idx = np.arange(n)
    inside = (y >= lo - delta) & (y <= hi + delta)
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    ax.fill_between([0, n - 1], [lo - delta] * 2, [hi + delta] * 2,
                    color="#fdae6b", alpha=0.45,
                    label=fr"widened band $[q_{{lo}}-\delta,\ q_{{hi}}+\delta]$")
    ax.fill_between([0, n - 1], [lo] * 2, [hi] * 2, color="#9ecae1", alpha=0.7,
                    label=r"original band $[q_{lo},\ q_{hi}]$")
    ax.scatter(idx[inside], y[inside], s=14, color="#238b45", label="covered")
    ax.scatter(idx[~inside], y[~inside], s=26, color="#cb181d", marker="x",
               label="missed")
    ax.set_xlabel("held out point"); ax.set_ylabel("return y")
    ax.set_title("Widening restores coverage without moving the center")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "conformal_bands.png"))
    plt.close(fig)



# 7. Mutual information ranking ---------------------------------------------
def fig_mi():
    banner("Worked example: mutual information of a 2x2 joint")
    # Joint p(x,y) for a simple dependent pair.
    p = np.array([[0.40, 0.10], [0.10, 0.40]])
    px = p.sum(1); py = p.sum(0)
    def H(v): 
        v = v[v > 0]; return float(-np.sum(v * np.log2(v)))
    hx, hy, hxy = H(px), H(py), H(p.ravel())
    I = hx + hy - hxy
    print(f"  H(X)={hx:.4f}  H(Y)={hy:.4f}  H(X,Y)={hxy:.4f}  "
          f"I(X,Y)={I:.4f} bits")
    names = ["vol252", "vol60", "rel_vol20", "vol_ratio", "macro_vol60",
             "atr14", "mom20", "rsi14"]
    mi = np.array([0.77, 0.61, 0.44, 0.39, 0.33, 0.27, 0.12, 0.08])
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.barh(names[::-1], mi[::-1], color="#756bb1")
    ax.set_xlabel(r"$I(X_j, Y)$ (bits)")
    ax.set_title("Feature ranking by mutual information")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "mutual_information.png")); plt.close(fig)


# 8. ROC curve worked example -----------------------------------------------
def fig_roc():
    banner("Worked example: AUC by ranks equals AUC by pair counting")
    rng = np.random.default_rng(11)
    n = 40
    y = rng.integers(0, 2, n)
    score = 0.6 * y + rng.normal(0, 1, n)
    # rank-sum estimator
    order = np.argsort(score)
    ranks = np.arange(1, n + 1, dtype=float)
    npos = int(y.sum()); nneg = n - npos
    rank_sum = ranks[y[order] == 1].sum()
    auc_rank = (rank_sum - npos * (npos + 1) / 2) / (npos * nneg)
    # brute pair counting
    pos = score[y == 1]; neg = score[y == 0]
    wins = sum((pp > nn) + 0.5 * (pp == nn) for pp in pos for nn in neg)
    auc_pair = wins / (npos * nneg)
    print(f"  npos={npos} nneg={nneg}  AUC(ranks)={auc_rank:.4f}  "
          f"AUC(pairs)={auc_pair:.4f}")
    # ROC curve
    thr = np.sort(np.unique(score))[::-1]
    tpr = [np.mean(score[y == 1] >= t) for t in thr]
    fpr = [np.mean(score[y == 0] >= t) for t in thr]
    tpr = [0] + tpr + [1]; fpr = [0] + fpr + [1]
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot(fpr, tpr, "-o", color="#1f77b4", ms=3,
            label=f"AUC = {auc_rank:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_title("ROC curve"); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "roc.png")); plt.close(fig)


# 9. Softmax and Adam worked numbers ----------------------------------------
def worked_softmax_adam():
    banner("Worked example: softmax meta layer")
    z = np.array([2.0, 0.5, -1.0])
    e = np.exp(z - z.max()); sm = e / e.sum()
    print(f"  z={z}  softmax={np.round(sm, 4)}  sum={sm.sum():.4f}")
    y = np.array([1.0, 0.0, 0.0])
    print(f"  cross entropy={-np.sum(y*np.log(sm)):.4f}  "
          f"grad (yhat-y)={np.round(sm-y, 4)}")
    banner("Worked example: three Adam steps on a constant gradient")
    b1, b2, lr, eps = 0.9, 0.999, 1e-3, 1e-8
    m = v = 0.0; theta = 1.0
    g = 0.5
    for t in range(1, 4):
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mh = m / (1 - b1 ** t); vh = v / (1 - b2 ** t)
        theta = theta - lr * mh / (np.sqrt(vh) + eps)
        print(f"  t={t}  m={m:.4f}  v={v:.6f}  mhat={mh:.4f}  "
              f"vhat={vh:.6f}  theta={theta:.6f}")


if __name__ == "__main__":
    fig_architecture()
    fig_lifecycle()
    fig_fusion_graph()
    fig_drift()
    fig_branch_lr()
    fig_branch_nb()
    fig_branch_mlp()
    fig_branch_sent()
    fig_branch_lstm()
    fig_sigmoid()
    fig_term_structure()
    fig_auc()
    fig_pinball()
    fig_cone()
    fig_conformal()
    fig_conformal_bands()
    fig_mi()
    fig_roc()
    worked_softmax_adam()
    print("\nAll figures written to", OUT)

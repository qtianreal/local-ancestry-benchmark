"""Publication figures.

Colours are the Okabe-Ito qualitative palette (Okabe & Ito 2008), constructed
to stay separable under deuteranopia, protanopia and tritanopia. Series are
grouped by family so the encoding carries meaning beyond identity: our two
likelihood baselines in blues, the released tools in warm hues, the learned
network in green. Every series is directly labelled, so identity is never
carried by colour alone.
"""

import json
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Only canonical POP_POP files are population pairs; tuning and ablation
# runs also write results/real_*.json and must not be plotted as pairs.
PAIR_RE = re.compile(r"^real_[A-Z]{3}_[A-Z]{3}\.json$")

OUT = Path("results")
FIG = Path("figures")
FIG.mkdir(exist_ok=True)

COL = {
    "naive_bayes": "#56B4E9",  # light blue   - windowed likelihood (ours)
    "hmm": "#0072B2",          # blue         - likelihood + HMM (ours)
    "rfmix": "#D55E00",        # vermillion   - RFMix v2
    "flare": "#E69F00",        # orange       - FLARE
    "cnn": "#009E73",          # bluish green - dilated CNN, frequency input
    "cnn_haplo": "#00503A",    # dark green   - same network, + haplotype input
}
LAB = {
    "naive_bayes": "Window likelihood", "hmm": "Likelihood + HMM",
    "rfmix": "RFMix v2", "flare": "FLARE", "cnn": "Dilated CNN",
    "cnn_haplo": "Dilated CNN + haplotype",
}
MARK = {"cnn": "o", "cnn_haplo": "s"}

mpl.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "figure.dpi": 300,
})


def load(name):
    p = OUT / name
    return json.loads(p.read_text()) if p.exists() else None


def load_npz(name):
    p = OUT / name
    return np.load(p) if p.exists() else None


def style_axis(ax, fst):
    ax.axhline(0.5, color="#BBBBBB", lw=0.7, ls="--", zorder=0)
    ax.set_xscale("log")
    ax.grid(axis="y", color="#EEEEEE", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(min(fst) * 0.7, max(fst) * 1.35)


def tag(ax, text, x, y, ha="right"):
    """Corner label saying which kind of data a panel is drawn from."""
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va="bottom",
            fontsize=5.4, color="#8A8A8A", style="italic")


def fig0():
    """Overview: what the task is, why divergence is the binding constraint,
    and how per-site accuracy hides the tract failure.

    Panel A is a diagram. Panels B and C are not: B counts diagnostic sites in
    simulated panels at the two ends of the sweep, and C shows a real ancestry
    track from a cached trained network on a held-out replicate, with its
    measured accuracies and tract counts. A schematic that invents its own
    numbers would be worse than no schematic, so the only hand-drawn element
    is the construction itself.
    """
    d = load_npz("overview_trace.npz")
    if d is None:
        return

    fig = plt.figure(figsize=(7.2, 5.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.40, 1.12, 0.95],
                          width_ratios=[1.0, 1.02], hspace=0.62, wspace=0.85)

    # ---- A: how an admixed chromosome is built -------------------------
    ax = fig.add_subplot(gs[0, :])
    ax.set_xlim(0, 100); ax.set_ylim(0.2, 13.6); ax.axis("off")
    ca, cb = "#0072B2", "#D55E00"
    rng = np.random.default_rng(7)

    for k, (col, y0, name) in enumerate(((ca, 7.2, "Source A"),
                                         (cb, 1.2, "Source B"))):
        for r in range(4):
            y = y0 + r * 0.72
            ax.add_patch(plt.Rectangle((6, y), 26, 0.5, color=col, alpha=0.30,
                                       lw=0))
            sites = rng.choice(np.arange(7, 31), size=9, replace=False)
            ax.scatter(sites, np.full(9, y + 0.25), s=9, marker="|",
                       linewidths=0.7, color=col, zorder=3)
        ax.text(19, y0 + 3.15, f"{name} reference panel", ha="center",
                fontsize=6.6, color=col)

    ax.annotate("", xy=(46, 5.1), xytext=(34, 5.1),
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#666666"))
    ax.text(40, 5.7, "recombine", ha="center", fontsize=6, color="#666666")

    # Mosaic target: alternating ancestry blocks with exact boundaries.
    bounds = [48, 58, 71, 79, 94]
    cols = [ca, cb, ca, cb]
    for i, (s, e) in enumerate(zip(bounds[:-1], bounds[1:])):
        ax.add_patch(plt.Rectangle((s, 4.6), e - s - 0.35, 1.0, color=cols[i],
                                   alpha=0.75, lw=0))
    ax.text(71, 6.0, "Admixed chromosome", ha="center", fontsize=6.6,
            color="#333333")
    ax.annotate("switch points are set, not inferred", xy=(71, 4.45),
                ha="center", va="top", fontsize=6, color="#666666")
    ax.text(6, 6.4, "ticks mark variant sites", fontsize=5.8, color="#888888",
            ha="left", va="top")

    # Every experiment in the paper scores these six on identical data; the
    # released tools in particular were not visible anywhere in this figure.
    ax.text(71, 2.65, "scored on every experiment", ha="center", fontsize=5.8,
            color="#555555")
    # Named, not colour-coded. In this panel blue and vermillion already mean
    # ancestry A and B; reusing the series palette here would give the same two
    # colours a second meaning inches away from the first. Six further hues
    # separable from the series palette under deuteranopia do not exist -- the
    # closest candidates we tested collapse onto "Likelihood + HMM" at dE 5-8
    # -- so the methods are listed by name and carry their colours in Figs 2,
    # 4 and 5, where each is actually plotted.
    meth = ["Window likelihood", "Likelihood + HMM", "RFMix v2",
            "FLARE", "Dilated CNN", "+ haplotype channels"]
    for i, lab in enumerate(meth):
        ax.text(50 + (i % 3) * 15.5, 1.5 - (i // 3) * 1.1, lab,
                fontsize=5.6, va="center", ha="left", color="#444444")

    ax.text(0, 13.5, "A", fontsize=9, fontweight="bold", va="top")
    ax.text(4.0, 13.45, "Exact ground truth by construction, used in both tracks",
            fontsize=7.2, va="top", color="#333333")

    # ---- B: the signal that distinguishes the sources ------------------
    ax = fig.add_subplot(gs[1, 0])
    ts = np.linspace(0.02, 0.6, 60)
    for arr, fst, col, ls in ((d["hi_diag"], float(d["hi_fst"]), "#333333", "-"),
                              (d["lo_diag"], float(d["lo_fst"]), "#009E73", "-")):
        ax.plot(ts, [100 * (arr > t).mean() for t in ts], color=col, lw=1.8,
                ls=ls, label=rf"$F_{{ST}}={fst:.3f}$")
    ax.set_yscale("log")
    ax.set_xlabel(r"Allele-frequency separation $|p_A-p_B|$")
    ax.set_ylabel("Sites exceeding (%)")
    ax.set_ylim(0.05, 90)
    ax.legend(frameon=False, fontsize=6.2, loc="upper right")
    ax.grid(axis="y", color="#EEEEEE", lw=0.6)
    ax.set_axisbelow(True)
    ax.text(-0.28, 1.22, "B", transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="top")
    # Last threshold at which any site in the close pair still separates.
    frac = np.array([(d["lo_diag"] > x).mean() for x in ts])
    lo_end = float(ts[int(np.flatnonzero(frac > 0).max())])
    ax.annotate("no site separates the\nclosely related pair\nby more than %.2f" % lo_end,
                xy=(lo_end, 0.085), xytext=(lo_end + 0.045, 0.14),
                fontsize=5.6, color="#4A7A68", linespacing=1.35,
                arrowprops=dict(arrowstyle="-", lw=0.5, color="#9ABFB1"))
    ax.set_title("A feasibility floor: the evidence runs out (simulated)",
                 fontsize=7.2)

    # ---- D: what moved accuracy and what did not -----------------------
    ax = fig.add_subplot(gs[1, 1])
    iv = load("interventions.json") or []
    SHORT = {"attn": "Self-attention", "ssm": "State-space layer",
             "cap": "Capacity, 15$\\times$ params",
             "lda": "Discriminant loss",
             "ssl": "Self-sup. pretrain", "more": "More labelled data",
             "simhaplo": "Simulation pretrain",
             "input": "Haplotype input"}
    # The frequency-only pretraining arm is excluded here and reported in the
    # supplement: it measures a configuration the recommended one supersedes,
    # and on this axis it would read as an architectural win that it is not.
    keep = [r for r in iv if r["tag"] in SHORT][::-1]
    for i, r in enumerate(keep):
        is_input = r["kind"] == "input"
        col = "#009E73" if is_input else "#9A9A9A"
        ax.scatter(r["deltas"], np.full(len(r["deltas"]), i), s=7,
                   color=col, alpha=0.45, zorder=2, linewidths=0)
        ax.scatter([r["mean"]], [i], s=26, color=col, zorder=3,
                   marker="D" if is_input else "o",
                   edgecolor="white", linewidth=0.6)
        ax.text(-0.092, i, SHORT[r["tag"]], ha="right", va="center",
                fontsize=5.6, color="#111111" if is_input else "#555555")
    ax.axvspan(-0.02, 0.02, color="#DDDDDD", alpha=0.45, lw=0, zorder=0)
    ax.axvline(0, color="#999999", lw=0.7, zorder=1)
    ax.set_ylim(-0.7, len(keep) - 0.3)
    ax.set_xlim(-0.089, 0.075)
    ax.set_yticks([])
    ax.set_xticks([-0.05, 0, 0.05])
    ax.set_xlabel("Change in per-site accuracy", fontsize=6.4)
    ax.tick_params(labelsize=6)
    ax.spines["left"].set_visible(False)
    ax.set_title("Only the input change moved accuracy (real haplotypes)",
                 fontsize=7.2)
    ax.text(-0.42, 1.24, "D", transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="top")

    # ---- C: accuracy hides the tract failure ---------------------------
    ax = fig.add_subplot(gs[2, :])
    tracks = (("True ancestry", d["truth"], 2.0),
              ("Network, per-site call", d["raw"], 1.0),
              ("Same logits, Viterbi-decoded", d["crf"], 0.0))
    n = d["truth"].size
    for name, seq, y in tracks:
        seq = np.asarray(seq)
        edges = np.flatnonzero(np.diff(seq) != 0) + 1
        for s, e in zip(np.r_[0, edges], np.r_[edges, n]):
            ax.add_patch(plt.Rectangle((s, y), e - s, 0.62,
                                       color=ca if seq[s] == 0 else cb,
                                       alpha=0.8, lw=0))
        ax.text(-0.012 * n, y + 0.31, name, ha="right", va="center",
                fontsize=6.4, color="#333333")
        ntr = 1 + int((np.diff(seq) != 0).sum())
        ax.text(1.012 * n, y + 0.31, f"{ntr} tracts", ha="left", va="center",
                fontsize=6.4, color="#666666")
    ax.set_xlim(0, n); ax.set_ylim(-0.45, 3.45)
    ax.axis("off")
    ax.text(0, 3.36, "C", fontsize=9, fontweight="bold", va="bottom",
            transform=ax.transData)
    ax.text(0.030 * n, 3.36, "Per-site accuracy misses tract structure (simulated)",
            fontsize=7.2, va="bottom", color="#333333")
    ax.annotate(
        "Per-site accuracy %.3f versus %.3f: the standard metric is nearly\n"
        "blind to the difference between these two tracks, and to the fix."
        % (float(d["acc_raw"]), float(d["acc_crf"])),
        xy=(n / 2, -0.30), ha="center", va="top", fontsize=6.3,
        color="#444444", linespacing=1.5)

    for e_ in ("pdf", "png"):
        fig.savefig(FIG / f"fig0_overview.{e_}", bbox_inches="tight")
    plt.close(fig)


def fig1():
    """Accuracy against divergence for all five methods."""
    agg = load("aggregate_results.json") or []
    ext = load("external_aggregate.json") or load("external_results.json") or []
    if not agg:
        return
    agg = sorted(agg, key=lambda r: r["fst"])
    ext = sorted(ext, key=lambda r: r["fst"])

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    fst_all = [r["fst"] for r in agg] + [r["fst"] for r in ext]
    style_axis(ax, fst_all)

    for key, rows in (("naive_bayes", agg), ("hmm", agg), ("cnn", agg),
                      ("rfmix", ext), ("flare", ext)):
        if not rows:
            continue
        x = np.array([r["fst"] for r in rows])
        y = np.array([r.get(key) if isinstance(r.get(key), float) else np.nan
                      for r in rows], dtype=float)
        e = np.array([r.get(f"{key}_sd", 0.0) or 0.0 for r in rows], dtype=float)
        ok = ~np.isnan(y)
        ax.errorbar(x[ok], y[ok], yerr=e[ok], color=COL[key], lw=1.8, marker="o",
                    ms=4.0, capsize=2, elinewidth=0.8, markeredgecolor="white",
                    markeredgewidth=0.5, zorder=3, label=LAB[key])

    # Same network with haplotype channels added, from the powered 5-seed
    # replication. Plotted at the main sweep's F_ST, matched by split time
    # rather than by measured F_ST: both sweeps ran the same split times, and
    # F_ST is a per-replicate estimate of that shared scenario. Using this
    # sweep's own estimates instead shifts the curve left by up to a factor of
    # two at the lowest level, which reads as a large low-divergence advantage
    # where the paired difference is in fact -0.006.
    sim = load("tuning/inputs_sim.json")
    if sim:
        levels = sorted({r["split_time"] for r in sim})
        anchor = [r["fst"] for r in agg]
        xs, ys, es = [], [], []
        for i, T in enumerate(levels):
            if i >= len(anchor):
                break
            h = [r["acc"] for r in sim
                 if r["split_time"] == T and r["features"] == "haplo"]
            xs.append(anchor[i])
            ys.append(np.mean(h))
            es.append(np.std(h, ddof=1))
        ax.errorbar(xs, ys, yerr=es, color=COL["cnn_haplo"], lw=1.5,
                    ls=(0, (4, 1.6)), marker=MARK["cnn_haplo"], ms=3.6,
                    capsize=2, elinewidth=0.8, markeredgecolor="white",
                    markeredgewidth=0.5, zorder=4, label=LAB["cnn_haplo"])

    # Series cannot be direct-labelled at the right edge without collision, so
    # identity is carried by a legend placed in the empty lower-right region
    # rather than by colour alone.
    # Legend order is set explicitly rather than by plot order, so the two
    # configurations of the network sit together and the grouping reads by
    # family: our likelihood baselines, the released tools, then the network.
    order = ["naive_bayes", "hmm", "rfmix", "flare", "cnn", "cnn_haplo"]
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    keep = [LAB[k] for k in order if LAB[k] in by_label]
    ax.legend([by_label[l] for l in keep], keep, fontsize=6.5, frameon=False,
              loc="lower right", handlelength=1.6, labelspacing=0.35,
              borderaxespad=0.6)

    ax.text(min(fst_all) * 0.78, 0.507, "chance", fontsize=6, color="#888888", va="bottom")
    ax.set_xlabel(r"Source-population divergence, Hudson $F_{ST}$")
    ax.set_ylabel("Per-site local-ancestry accuracy")
    ax.set_ylim(0.44, 1.03)
    fig.tight_layout()
    for ext_ in ("pdf", "png"):
        fig.savefig(FIG / f"fig1_accuracy_vs_fst.{ext_}", bbox_inches="tight")
    plt.close(fig)


def fig2():
    d = load("transfer_results.json")
    if not d:
        return
    m = np.array(d["matrix"])
    ticks = [f"{v:.4f}" for v in d["fst"]]
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(m, cmap="Blues", vmin=0.5, vmax=1.0)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if m[i, j] > 0.8 else "#333333")
    ax.set_xticks(range(len(ticks)), ticks, rotation=45, ha="right")
    ax.set_yticks(range(len(ticks)), ticks)
    ax.set_xlabel(r"Evaluated at $F_{ST}$")
    ax.set_ylabel(r"Trained at $F_{ST}$")
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Per-site accuracy", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_visible(False)
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(FIG / f"fig2_transfer_matrix.{e}", bbox_inches="tight")
    plt.close(fig)


def fig3():
    """Long-range reliance (ablation) and discriminant magnitude (LDA).

    The retention ratio divides by above-chance accuracy, so where the network
    is itself near chance the denominator approaches zero and the ratio is not
    usefully estimable. Those points are drawn open and joined by a dashed
    segment rather than being given the same visual weight as the rest, so the
    figure asserts only what the text does.

    The band marks where the network's advantage over the best likelihood
    baseline is within 80% of its maximum. That the retention minimum falls
    inside it is the point of the panel, and without the band the reader has
    to align two x-axes across figures by eye to see it.
    """
    pr = load("pruning_summary.json")
    lda = load("lda_summary.json")
    if not pr:
        return
    pr = sorted(pr, key=lambda r: r["fst"])
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9))

    ax = axes[0]
    x = np.array([r["fst"] for r in pr])
    y = np.array([r["retention"] for r in pr])
    e = np.array([r.get("retention_sd", 0.0) for r in pr])
    # Denominator of the ratio: how much above-chance accuracy there is to lose.
    head = np.array([r["full_acc"] - 0.5 for r in pr])
    ok = head >= 0.20

    agg = sorted(load("aggregate_results.json") or [], key=lambda r: r["fst"])
    if agg:
        adv = np.array([r["cnn"] - max(r["hmm"], r["naive_bayes"]) for r in agg])
        inside = np.array([r["fst"] for r in agg])[adv >= 0.8 * adv.max()]
        ax.axvspan(inside.min(), inside.max(), color="#009E73", alpha=0.09,
                   lw=0, zorder=0)
        ax.annotate("band of maximum\nadvantage over baselines",
                    xy=(float(np.sqrt(inside.min() * inside.max())), 1.40),
                    ha="center", va="top", fontsize=5.4, color="#4A7A68",
                    linespacing=1.3)

    ax.plot(x[~ok], y[~ok], color="#009E73", lw=1.0, ls=":", zorder=2)
    if not ok[0] and ok.any():
        # Bridge the last non-estimable point to the first solid one, dotted,
        # so the series reads as continuous without implying equal confidence.
        j = int(ok.argmax())
        ax.plot(x[j - 1:j + 1], y[j - 1:j + 1], color="#009E73", lw=1.0,
                ls=":", zorder=2)
    ax.errorbar(x[ok], y[ok], yerr=e[ok], color="#009E73", lw=1.8, marker="o",
                ms=4, capsize=2, elinewidth=0.8, markeredgecolor="white",
                markeredgewidth=0.5, zorder=3)
    ax.errorbar(x[~ok], y[~ok], yerr=e[~ok], color="#009E73", lw=0,
                marker="o", ms=4, capsize=2, elinewidth=0.8, alpha=0.55,
                markerfacecolor="white", markeredgecolor="#009E73",
                markeredgewidth=0.9, zorder=3)
    ax.set_ylim(-0.22, 1.45)
    if (~ok).any():
        # Below the spine, centred under the open points: inside the axes it
        # either collides with their error bars or runs off the left edge.
        import matplotlib.transforms as mtransforms
        tr = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(float(np.sqrt(x[~ok].min() * x[~ok].max())), -0.115,
                "not estimable", transform=tr, ha="center", va="top",
                fontsize=5.4, color="#8A8A8A", clip_on=False)

    ax.axhline(1.0, color="#BBBBBB", lw=0.7, ls="--", zorder=1)
    ax.text(x.max(), 1.02, "no reliance on long-range blocks", ha="right",
            va="bottom", fontsize=5.4, color="#999999")
    ax.set_xscale("log")
    ax.set_xlabel(r"$F_{ST}$")
    ax.set_ylabel("Above-chance accuracy retained")
    ax.set_title("Removing dilations $\\geq 8$", fontsize=8)
    ax.grid(axis="y", color="#EEEEEE", lw=0.6)
    ax.set_axisbelow(True)

    if lda:
        lda = sorted(lda, key=lambda r: r["fst"])
        ax = axes[1]
        x = [r["fst"] for r in lda]
        y = [r["J_final"] for r in lda]
        e = [r.get("J_final_sd", 0.0) for r in lda]
        ax.errorbar(x, y, yerr=e, color="#0072B2", lw=1.8, marker="o", ms=4,
                    capsize=2, elinewidth=0.8, markeredgecolor="white", markeredgewidth=0.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$F_{ST}$")
        ax.set_ylabel(r"LDA criterion $J$ (final block)")
        ax.set_title("Discriminant information", fontsize=8)
        ax.grid(axis="y", color="#EEEEEE", lw=0.6)
        ax.set_axisbelow(True)

    for ax_, letter in zip(axes, "AB"):
        ax_.text(-0.20, 1.06, letter, transform=ax_.transAxes, fontsize=9,
                 fontweight="bold", va="bottom", ha="left")

    fig.tight_layout()
    for e_ in ("pdf", "png"):
        fig.savefig(FIG / f"fig3_mechanism.{e_}", bbox_inches="tight")
    plt.close(fig)


def fig4():
    """Real 1000 Genomes pairs, on the same log-Fst axis as Fig 1.

    With enough pairs to span the divergence range, a scatter against measured
    Fst is far more informative than grouped bars: it can be read directly
    against the simulated curve, which is the comparison the paper is about.
    """
    rows = []
    for p in sorted(OUT.glob("real_*.json")):
        if not PAIR_RE.match(p.name):
            continue
        r = json.loads(p.read_text())
        tag = "_".join(r["pops"])
        e = OUT / f"realext_{tag}.json"
        if e.exists():
            r.update({k: v for k, v in json.loads(e.read_text()).items()
                      if k in ("rfmix", "flare")})
        rows.append(r)
    if not rows:
        return
    rows.sort(key=lambda r: r["fst"])
    fst = np.array([r["fst"] for r in rows])

    # Haplotype-aware accuracies, matched to the same pairs by population tag.
    ip = {tuple(r["pops"]): r for r in (load("inputs_summary.json") or [])}
    hap = np.array([ip[tuple(r["pops"])]["acc"]["haplo"]
                    if tuple(r["pops"]) in ip else np.nan for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4),
                             gridspec_kw={"width_ratios": [2.3, 1]})

    # Grouped bars, drawn from chance rather than from zero. Chance is the
    # meaningful floor for a two-way call, so anchoring there is a reference
    # baseline rather than a truncated axis, and it lets a below-chance result
    # (CHB/CHS) read as a bar below the line instead of vanishing.
    ax = axes[0]
    keys = ("naive_bayes", "hmm", "rfmix", "flare", "cnn", "cnn_haplo")
    n = len(keys)
    width = 0.78 / n
    xs = np.arange(len(rows))
    for i, k in enumerate(keys):
        if k == "cnn_haplo":
            y = hap
        else:
            y = np.array([r.get(k) if isinstance(r.get(k), float) else np.nan
                          for r in rows], dtype=float)
        off = (i - (n - 1) / 2) * width
        ax.bar(xs + off, y - 0.5, bottom=0.5, width=width * 0.86,
               color=COL[k], label=LAB[k], zorder=3, linewidth=0)
    ax.axhline(0.5, color="#888888", lw=0.8, zorder=4)
    ax.set_xticks(xs)
    # F_ST is the axis the whole paper is organised around, so each pair
    # carries its own value rather than only its rank. Rotating lets the
    # two-part label fit where stacked lines collided at eleven pairs.
    ax.set_xticklabels([f"{'/'.join(r['pops'])}  {r['fst']:.4f}" for r in rows],
                       fontsize=6.2, rotation=45, ha="right",
                       rotation_mode="anchor")
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.set_ylim(0.44, 1.02)
    ax.set_xlabel(r"Population pair, ordered by measured $F_{ST}$")
    ax.set_ylabel("Per-site accuracy")
    ax.grid(axis="y", color="#EEEEEE", lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7.2, frameon=False, ncol=3, loc="upper left",
              labelspacing=0.3, columnspacing=1.1, handlelength=1.1,
              handletextpad=0.4)

    # Right panel: the quantity the argument turns on.
    ax = axes[1]
    gap = np.array([r["cnn"] - max(r["rfmix"], r["flare"])
                    if isinstance(r.get("rfmix"), float) else np.nan for r in rows])
    gap_h = np.array([h - max(r["rfmix"], r["flare"])
                      if isinstance(r.get("rfmix"), float) and not np.isnan(h)
                      else np.nan for r, h in zip(rows, hap)])
    ok = ~np.isnan(gap)
    ax.axhline(0, color="#888888", lw=0.9, zorder=2)
    # Join each pair's two configurations so the shift is readable per pair.
    for f, a, b in zip(fst, gap, gap_h):
        if not (np.isnan(a) or np.isnan(b)):
            ax.plot([f, f], [a, b], color="#CCCCCC", lw=0.8, zorder=1)
    ax.plot(fst[ok], gap[ok], color=COL["cnn"], lw=0, marker="o", ms=5,
            markeredgecolor="white", markeredgewidth=0.6,
            label=LAB["cnn"], zorder=3)
    okh = ~np.isnan(gap_h)
    ax.plot(fst[okh], gap_h[okh], color=COL["cnn_haplo"], lw=0,
            marker=MARK["cnn_haplo"], ms=5, markeredgecolor="white",
            markeredgewidth=0.6, label=LAB["cnn_haplo"], zorder=4)
    ax.legend(fontsize=6.9, frameon=False, loc="lower right", labelspacing=0.3)
    gap = np.concatenate([gap, gap_h])
    lo, hi = np.nanmin(gap), np.nanmax(gap)
    pad = 0.35 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)
    # Per-pair labels are omitted here: with two configurations per pair they
    # collide, and Fig 5 carries the labelled per-pair view.
    ax.set_xscale("log")
    ax.set_xlabel(r"Measured $F_{ST}$")
    ax.set_ylabel("Network $-$ best released tool")
    ax.grid(axis="y", color="#EEEEEE", lw=0.6)
    ax.set_axisbelow(True)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig4_real_data.{ext}", bbox_inches="tight")
    plt.close(fig)


def fig5():
    """The headline result: what haplotype features buy, against divergence.

    Two panels because the effect is not one quantity. Accuracy changes sign
    with divergence; tract structure improves almost everywhere. Plotting only
    accuracy would hide half the result, which is the same mistake the paper
    documents elsewhere.

    Label offsets are set per pair: several points sit close enough that
    automatic placement collides, and a publication figure should not make the
    reader disentangle overlapping text.
    """
    import json as _json
    recs = load("inputs_summary.json")
    if not recs:
        return
    raw = OUT / "tuning"
    recs = sorted(recs, key=lambda r: r["fst"])
    fst = np.array([r["fst"] for r in recs])
    d = np.array([r["acc"]["delta"] for r in recs])
    tags = ["/".join(r["pops"]) for r in recs]

    sem = []
    for r in recs:
        f = raw / f"inputs_{'_'.join(r['pops'])}.json"
        rows = _json.loads(f.read_text()) if f.exists() else []
        base = {x["seed"]: x["acc"] for x in rows if x["features"] == "freq"}
        dd = [x["acc"] - base[x["seed"]] for x in rows
              if x["features"] == "haplo" and x["seed"] in base]
        sem.append(np.std(dd, ddof=1) / np.sqrt(len(dd)) if len(dd) > 1 else np.nan)
    sem = np.array(sem)

    # (dx pt, dy pt, ha) per pair; tuned against the rendered figure.
    OFF = {
        "CHB/CHS": (0, 9, "center"),   "CEU/TSI": (0, 10, "center"),
        "FIN/GBR": (-4, 9, "right"),   "CHB/CDX": (-5, -12, "right"),
        "CHB/JPT": (0, -13, "center"), "FIN/TSI": (6, 6, "left"),
        "TSI/PJL": (-3, 9, "right"),   "CEU/GIH": (4, -12, "left"),
        "CHB/BEB": (-4, 8, "right"),   "CHB/GIH": (0, 9, "center"),
        "CHB/CEU": (2, -13, "left"),
    }
    OFF["CEU/GIH"] = (6, -2, "left")   # was colliding with CHB/GIH near zero
    # The right panel has a different point layout and needs its own offsets.
    OFF2 = {
        "CHB/CHS": (0, 9, "center"),   "CEU/TSI": (0, 9, "center"),
        "FIN/GBR": (-6, -2, "right"),  "CHB/CDX": (0, -12, "center"),
        "CHB/JPT": (-5, 5, "right"),   "FIN/TSI": (5, 5, "left"),
        "TSI/PJL": (0, -12, "center"), "CEU/GIH": (0, -12, "center"),
        "CHB/BEB": (0, 9, "center"),   "CHB/GIH": (6, -3, "left"),
        "CHB/CEU": (-5, 5, "right"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.3))

    ax = axes[0]
    ax.axvspan(0.042, fst.max() * 1.9, color="#F4F4F4", zorder=0)
    ax.axhline(0, color="#888888", lw=0.9, zorder=2)
    pos = d > 0
    ax.errorbar(fst[pos], d[pos], yerr=sem[pos], fmt="o", ms=5,
                color=COL["cnn_haplo"], ecolor="#AAAAAA", elinewidth=0.8,
                capsize=2, markeredgecolor="white", markeredgewidth=0.6,
                label="improves accuracy", zorder=3)
    ax.errorbar(fst[~pos], d[~pos], yerr=sem[~pos], fmt="v", ms=5,
                color="#D55E00", ecolor="#AAAAAA", elinewidth=0.8, capsize=2,
                markeredgecolor="white", markeredgewidth=0.6,
                label="degrades accuracy", zorder=3)
    for tag, g, f in zip(tags, d, fst):
        dx, dy, ha = OFF.get(tag, (0, 8, "center"))
        ax.annotate(tag, xy=(f, g), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, fontsize=6.4, color="#555555", zorder=5)
    ax.set_xscale("log")
    ax.set_xlim(fst.min() * 0.55, fst.max() * 1.9)
    ax.set_ylim(-0.037, 0.093)
    ax.set_xlabel(r"Measured $F_{ST}$ between source populations")
    ax.set_ylabel("Accuracy change from haplotype features")
    ax.grid(axis="y", color="#EEEEEE", lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7.4, frameon=False, loc="upper left", labelspacing=0.3,
              handletextpad=0.4)
    ax.annotate("frequency alone suffices", xy=(0.05, -0.033), fontsize=6.9,
                color="#999999", ha="left")

    ax = axes[1]
    tf = np.array([r["tracts"]["freq"] for r in recs])
    ratio = np.array([r["tracts"]["haplo"] for r in recs]) / tf
    ax.axhline(1, color="#888888", lw=0.9, zorder=2)
    ax.plot(fst, ratio, color=COL["cnn_haplo"], lw=0, marker="s", ms=5,
            markeredgecolor="white", markeredgewidth=0.6, zorder=3)
    for tag, g, f in zip(tags, ratio, fst):
        dx, dy, ha = OFF2.get(tag, (0, 8, "center"))
        ax.annotate(tag, xy=(f, g), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, fontsize=6.4, color="#555555", zorder=5)
    ax.set_xscale("log")
    ax.set_xlim(fst.min() * 0.55, fst.max() * 1.9)
    ax.set_ylim(0.44, 1.30)
    ax.set_xlabel(r"Measured $F_{ST}$")
    ax.set_ylabel("Tract count, haplotype $/$ frequency")
    ax.annotate("fewer spurious tracts", xy=(fst.min() * 0.65, 0.47),
                fontsize=6.9, color="#999999")
    ax.grid(axis="y", color="#EEEEEE", lw=0.6)
    ax.set_axisbelow(True)

    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(FIG / f"fig5_haplotype.{e}", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    import sys
    made = []
    for name, fn in (("fig0", fig0), ("fig1", fig1), ("fig2", fig2),
                     ("fig3", fig3), ("fig4", fig4), ("fig5", fig5)):
        try:
            fn(); made.append(name)
        except Exception as exc:                       # noqa: BLE001
            print(f"FAILED {name}: {type(exc).__name__}: {exc}", file=sys.stderr)
    print("wrote:", ", ".join(sorted(p.name for p in FIG.glob("*.pdf"))))
    if len(made) != 6:
        sys.exit(f"only {len(made)}/6 figures generated: {made}")


def fig6():
    """The identifiability index: does one constant predict accuracy?

    Left: accuracy against the index, both arms, with each arm's fitted curve.
    The vertical offset between the two curves is the simulator's flattery,
    and is the reason the arms are not pooled. Right: leave-one-out predicted
    against observed on the real pairs, which is the practitioner's case.
    """
    d = load("identifiability.json")
    if d is None:
        raise FileNotFoundError("results/identifiability.json -- run run_identifiability.py")

    import math
    phi = lambda z: 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    real = [p for p in d["pairs"] if p["arm"] == "real"]
    sim = [p for p in d["pairs"] if p["arm"] == "sim"]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

    ax = axes[0]
    xs = np.logspace(np.log10(2), np.log10(2500), 400)
    for pts, k, col, lab, mk in (
        (sim, d["kappa_sim"], "#8A8A8A", "Simulated", "^"),
        (real, d["kappa_real"], COL["cnn"], "Real 1000G pairs", "o"),
    ):
        ax.plot(xs, [phi(math.sqrt(k * x)) for x in xs], color=col, lw=1.2, zorder=2)
        ax.scatter([p["x"] for p in pts], [p["best"] for p in pts], s=18, color=col,
                   marker=mk, label=lab, zorder=3, edgecolor="white", linewidth=0.4)
    ax.axhline(0.5, color="#BBBBBB", lw=0.7, ls="--", zorder=0)
    ax.set_xscale("log")
    ax.set_xlabel(r"identifiability index  $x=\delta F_{ST}/g$  (sites per tract $\times$ $F_{ST}$)")
    ax.set_ylabel("best per-site accuracy")
    ax.set_ylim(0.45, 1.02)
    ax.grid(axis="y", color="#EEEEEE", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=6.5, loc="lower right", bbox_to_anchor=(1.0, 0.02))
    tag(ax, "simulated + real", 0.98, 0.30)

    ax = axes[1]
    obs = np.array([p["best"] for p in real])
    err = np.array(d["loo_index"]["errors"])
    pred = obs + err
    ax.plot([0.5, 1.0], [0.5, 1.0], color="#BBBBBB", lw=0.7, ls="--", zorder=1)
    ax.scatter(obs, pred, s=20, color=COL["cnn"], zorder=3,
               edgecolor="white", linewidth=0.4)
    # Several pairs sit on top of each other near 0.72, and the three most
    # accurate sit against the right edge. Place each label at the first
    # candidate offset that neither collides with an already-placed label nor
    # leaves the axes, working in display coordinates so the test is in the
    # units collisions actually happen in.
    fig.canvas.draw()
    CAND = [(5, -6), (5, 5), (-6, -6), (-6, 5), (5, -15), (5, 14), (-6, -15), (-6, 14)]
    boxes = []
    x0, x1 = ax.get_xlim()
    for p, o, pr in sorted(zip(real, obs, pred), key=lambda t: -t[1]):
        px, py = ax.transData.transform((o, pr))
        s = fig.dpi / 72.0                          # points -> pixels
        w, h = 3.1 * len(p["label"]) * s, 7.0 * s   # rough label extent, pixels
        for dx, dy in CAND:
            left = px + dx * s if dx > 0 else px + dx * s - w
            box = (left, py + dy * s - h / 2, left + w, py + dy * s + h / 2)
            inside = (ax.transData.transform((x0, 0))[0] <= box[0]
                      and box[2] <= ax.transData.transform((x1, 0))[0])
            clash = any(box[0] < b[2] and b[0] < box[2] and box[1] < b[3] and b[1] < box[3]
                        for b in boxes)
            if inside and not clash:
                break
        boxes.append(box)
        ax.annotate(p["label"], (o, pr), fontsize=5.0, color="#666666",
                    ha="left" if dx > 0 else "right",
                    xytext=(dx, dy), textcoords="offset points")
    ax.set_xlabel("observed accuracy")
    ax.set_ylabel("predicted, leave-one-out")
    ax.set_xlim(0.5, 1.02)
    ax.set_ylim(0.5, 1.02)
    ax.grid(color="#EEEEEE", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.text(0.04, 0.93, f"RMSE {d['loo_index']['rmse']:.3f}", transform=ax.transAxes,
            fontsize=6.5, color="#444444")
    tag(ax, "real pairs only", 0.98, 0.04)

    # Same lettering as fig3, since the caption refers to panels A and B.
    for ax_, letter in zip(axes, "AB"):
        ax_.text(-0.20, 1.06, letter, transform=ax_.transAxes, fontsize=9,
                 fontweight="bold", va="bottom", ha="left")

    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(FIG / f"fig6_identifiability.{e}", bbox_inches="tight")
    plt.close(fig)

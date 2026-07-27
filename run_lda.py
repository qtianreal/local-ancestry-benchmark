"""Discriminant structure of the trained networks as a function of task difficulty.

Block-group ablation answers "how much accuracy is lost if this is removed",
which is confounded by compensation between residual blocks. A discriminant
criterion answers the prior question directly: how much class-separating
information does each block's representation actually carry?

For each block output h (C channels x S sites) and the per-site ancestry label
y, we form the between- and within-class scatter matrices over channels,

    S_B = sum_k n_k (mu_k - mu)(mu_k - mu)^T
    S_W = sum_k sum_{i in k} (h_i - mu_k)(h_i - mu_k)^T

and report three quantities:

  J          trace((S_W + lambda I)^-1 S_B), the multivariate LDA criterion.
             Unlike per-channel Fisher ratios this accounts for redundancy
             between correlated channels, which is precisely the failure mode
             that made single-block ablation uninformative here.

  d_eff      participation ratio of the per-channel Fisher ratios,
             (sum f)^2 / sum f^2: how many channels effectively carry
             discriminant signal. Note the eigen-spectrum of
             (S_W + lambda I)^-1 S_B cannot be used for this, since with two
             classes S_B has rank one and its participation ratio is
             identically 1 regardless of the representation.

  fisher_max largest single-channel Fisher ratio, for comparability with
             channel-pruning criteria that ignore redundancy.

The divergence grid supplies something an ordinary vision benchmark cannot: a
continuous, ground-truth task-difficulty axis, so discriminant structure can be
tracked as the task moves from trivially separable to information-theoretically
impossible.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lai.methods import DilatedCNN

OUT = Path("results")
CACHE = OUT / "cache"
DILATIONS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
RIDGE = 1e-3  # relative ridge on S_W; the scatter is rank-deficient otherwise


@torch.no_grad()
def block_activations(model, x, device, batch=16, max_sites=2048):
    """Collect stem and per-block outputs, subsampled over sites."""
    acts = [[] for _ in range(len(model.blocks) + 1)]
    for i in range(0, x.shape[0], batch):
        xb = torch.from_numpy(x[i : i + batch].astype(np.float32)).to(device)
        h = model.stem(xb)
        acts[0].append(h[:, :, :max_sites].cpu().numpy())
        for j, blk in enumerate(model.blocks):
            h = h + blk(h)
            acts[j + 1].append(h[:, :, :max_sites].cpu().numpy())
    return [np.concatenate(a) for a in acts]


def discriminant_stats(act, labels):
    """LDA criterion, effective discriminant dimensionality, max Fisher ratio.

    act    : (N, C, S) activations
    labels : (N, S) binary ancestry labels
    """
    n, c, s = act.shape
    X = act.transpose(0, 2, 1).reshape(-1, c).astype(np.float64)
    y = labels[:, :s].reshape(-1).astype(np.int8)

    # Residual accumulation over nine blocks leaves activations on very
    # different per-channel scales, which overflows the scatter accumulation.
    # trace(S_W^-1 S_B) is invariant to per-channel rescaling, so standardising
    # is numerically necessary and analytically free.
    X -= X.mean(axis=0)
    X /= np.maximum(X.std(axis=0), 1e-8)

    mu = X.mean(axis=0)
    S_B = np.zeros((c, c))
    S_W = np.zeros((c, c))
    per_channel = np.zeros(c)
    means, varis, counts = [], [], []

    for k in (0, 1):
        m = y == k
        nk = int(m.sum())
        if nk < 2:
            return dict(J=float("nan"), d_eff=float("nan"), fisher_max=float("nan"))
        Xk = X[m]
        muk = Xk.mean(axis=0)
        d = (muk - mu)[:, None]
        S_B += nk * (d @ d.T)
        Xc = Xk - muk
        S_W += Xc.T @ Xc
        means.append(muk)
        varis.append(Xk.var(axis=0))
        counts.append(nk)

    S_B /= len(y)
    S_W /= len(y)
    per_channel = (means[0] - means[1]) ** 2 / (varis[0] + varis[1] + 1e-12)

    ridge = RIDGE * np.trace(S_W) / c
    M = np.linalg.solve(S_W + ridge * np.eye(c), S_B)
    w = np.clip(np.linalg.eigvals(M).real, 0, None)

    f = per_channel
    d_eff = float(f.sum() ** 2 / (f ** 2).sum()) if (f ** 2).sum() > 0 else float("nan")
    # Channels needed to reach 90% of the summed per-channel Fisher ratio.
    order = np.sort(f)[::-1]
    csum = np.cumsum(order) / max(order.sum(), 1e-12)
    n90 = int(np.searchsorted(csum, 0.90) + 1)
    return dict(J=float(w.sum()), d_eff=d_eff, n90=n90,
                fisher_max=float(f.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    fst_by_T = {}
    for s in seeds:
        f = OUT / f"main_results_s{s}.json"
        if f.exists():
            for r in json.loads(f.read_text()):
                fst_by_T.setdefault(r["split_time"], []).append(r["fst"])

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    records = []

    for T in sorted(fst_by_T):
        for s in seeds:
            mp, tp = CACHE / f"cnn_T{T}_s{s}.pt", CACHE / f"test_T{T}_s{s}.npz"
            if not (mp.exists() and tp.exists()):
                continue
            model = DilatedCNN().to(device)
            model.load_state_dict(torch.load(mp, map_location=device))
            model.eval()
            d = np.load(tp)
            x, y = d["x"][:64], d["y"][:64]

            acts = block_activations(model, x, device)
            stats = [discriminant_stats(a, y) for a in acts]
            records.append({
                "split_time": T, "seed": s, "fst": float(np.mean(fst_by_T[T])),
                "J": [st["J"] for st in stats],
                "d_eff": [st["d_eff"] for st in stats],
                "fisher_max": [st["fisher_max"] for st in stats],
                "n90": [st["n90"] for st in stats],
            })
            print(f"Fst={records[-1]['fst']:.5f} s{s}  "
                  f"J(stem->last)={stats[0]['J']:.2f}->{stats[-1]['J']:.2f}  "
                  f"d_eff last={stats[-1]['d_eff']:.2f}", flush=True)

    (OUT / "lda_results.json").write_text(json.dumps(records, indent=2))

    # Summarise across seeds.
    by_T = {}
    for r in records:
        by_T.setdefault(r["split_time"], []).append(r)
    summary = []
    for T in sorted(by_T):
        g = by_T[T]
        Jl = np.array([r["J"][-1] for r in g])
        de = np.array([r["d_eff"][-1] for r in g])
        n90 = np.array([r["n90"][-1] for r in g])
        # Which block contributes the largest increment in J.
        inc = np.array([np.diff(r["J"]) for r in g]).mean(axis=0)
        summary.append({
            "split_time": T, "fst": g[0]["fst"], "n": len(g),
            "J_final": float(Jl.mean()), "J_final_sd": float(Jl.std(ddof=1)) if len(g) > 1 else 0.0,
            "d_eff_final": float(de.mean()), "d_eff_final_sd": float(de.std(ddof=1)) if len(g) > 1 else 0.0,
            "n90_final": float(n90.mean()),
            "peak_block": int(inc.argmax()), "peak_dilation": DILATIONS[min(int(inc.argmax()), 8)],
        })
    (OUT / "lda_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{'Fst':>9}{'n':>3}{'J_final':>16}{'d_eff':>14}{'n90':>7}{'peak dil':>10}")
    for r in summary:
        print(f"{r['fst']:>9.5f}{r['n']:>3}{r['J_final']:>10.2f}+/-{r['J_final_sd']:<5.2f}"
              f"{r['d_eff_final']:>8.2f}+/-{r['d_eff_final_sd']:<5.2f}"
              f"{r['n90_final']:>7.1f}{r['peak_dilation']:>10}")


if __name__ == "__main__":
    main()

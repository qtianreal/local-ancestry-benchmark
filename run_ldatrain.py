"""Does a discriminant-shaping training objective close the gap to FLARE?

Everything that changed the network's capacity or context in this study moved
accuracy by under 0.02 with inconsistent sign; the only intervention that
mattered was changing what the network is shown. An auxiliary LDA objective is
a different axis again -- it changes neither capacity nor input, but shapes the
representation the head reads from -- and it is the one axis left untested.

The auxiliary term is the two-class Fisher criterion on the penultimate
features,

    J = (m_1 - m_0)^T (S_W + lambda I)^-1 (m_1 - m_0),

maximised alongside binary cross-entropy. With two classes the between-class
scatter has rank one, so J is a scalar rather than a spectrum -- the
eigen-decomposition used for effective dimensionality is unavailable here, but
is not needed for a loss. We optimise log(1 + J) rather than J so that
gradients stay bounded when the classes separate cleanly.

Run on the pairs where the haplotype-aware network still trails FLARE, since
those are the only ones where closing a gap would change a conclusion. Paired
on seed against the same 8-channel configuration trained without the auxiliary
term, so the comparison isolates the objective.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from lai.methods import DilatedCNN
from lai.real import load_region, split_panel
from lai.sim import hudson_fst
from run_inputs import CHANNELS, evaluate, windows_from
from run_real import build_segment

OUT = Path("results/tuning")


class Probed(DilatedCNN):
    """DilatedCNN that also returns the features the head reads from."""

    def forward(self, x, return_feat=False):
        h = self.stem(x)
        for block in self.blocks:
            h = h + block(h)
        out = self.head(h).squeeze(1)
        return (out, h) if return_feat else out


def fisher_j(feat, labels, n_sites=512, ridge=1e-2, rng=None):
    """Two-class Fisher criterion on a subsample of per-site features.

    feat   (N, C, L) penultimate activations
    labels (N, L)    binary ancestry
    """
    n, c, L = feat.shape
    idx = torch.randint(0, L, (min(n_sites, L),), device=feat.device)
    f = feat[:, :, idx].permute(0, 2, 1).reshape(-1, c)      # (M, C)
    y = labels[:, idx].reshape(-1)
    m0, m1 = y < 0.5, y >= 0.5
    if m0.sum() < 2 or m1.sum() < 2:
        return feat.new_zeros(())
    f0, f1 = f[m0], f[m1]
    d = f1.mean(0) - f0.mean(0)                              # (C,)
    f0c, f1c = f0 - f0.mean(0), f1 - f1.mean(0)
    sw = (f0c.T @ f0c + f1c.T @ f1c) / f.shape[0]
    sw = sw + ridge * torch.eye(c, device=f.device, dtype=f.dtype)
    # torch.linalg.solve has no MPS backward implementation, and the scatter
    # matrix is only C x C, so the solve is done on CPU and moved back. The
    # cost is negligible against the convolutions and autograd handles the
    # device round trip.
    # Kept in float32: a float64 round trip cannot send gradients back to MPS.
    # The ridge above keeps the 64x64 system well conditioned at this precision.
    dev = d.device
    r = torch.linalg.solve(sw.cpu(), d.cpu()).to(dev)
    return d @ r


def train(model, x, y, device, epochs, lr, batch, alpha):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lf = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(x.shape[0])
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            xb, yb = x[idx].to(device), y[idx].to(device)
            opt.zero_grad()
            if alpha > 0:
                out, feat = model(xb, return_feat=True)
                loss = lf(out, yb) - alpha * torch.log1p(
                    torch.clamp(fisher_j(feat, yb), min=0))
            else:
                loss = lf(model(xb), yb)
            loss.backward()
            opt.step()
        sch.step()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pops", required=True)
    ap.add_argument("--vcf", default="data/chr22.vcf.gz")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=15)
    args = ap.parse_args()

    pops = args.pops.split(",")
    haps, positions = load_region(args.vcf, "chr22", 16_000_000, 51_000_000, pops)
    fst = hudson_fst(haps[pops[0]], haps[pops[1]])
    n = positions.size
    cut, buf = int(n * 0.60), int(n * 0.05)
    tr_sl, te_sl = slice(0, cut), slice(cut + buf, n)
    print(f"{'/'.join(pops)}: Fst={fst:.5f}  alpha={args.alpha}", flush=True)

    segs = []
    for rep in range(args.reps):
        rng = np.random.default_rng(4242 + rep)
        ra, da = split_panel(haps[pops[0]], 80, 80, rng)
        rb, db = split_panel(haps[pops[1]], 80, 80, rng)
        s_tr, _ = build_segment(ra[tr_sl], rb[tr_sl], da[tr_sl], db[tr_sl],
                                positions[tr_sl], 64, rng)
        s_te, _ = build_segment(ra[te_sl], rb[te_sl], da[te_sl], db[te_sl],
                                positions[te_sl], 64, rng)
        segs.append((rep, s_tr, s_te))

    xs, ys, xe, ye = [], [], [], []
    for rep, s_tr, s_te in segs:
        rng = np.random.default_rng(4242 + rep)
        a, b = windows_from(s_tr, 3, rng, "haplo"); xs.append(a); ys.append(b)
        a, b = windows_from(s_te, 1, np.random.default_rng(999 + rep), "haplo")
        xe.append(a); ye.append(b)
    xtr, ytr = torch.cat(xs), torch.cat(ys)
    xte, yte = torch.cat(xe), torch.cat(ye)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"  train {tuple(xtr.shape)} test {tuple(xte.shape)}", flush=True)

    out_path = OUT / f"ldatrain_{'_'.join(pops)}.json"
    rows = json.loads(out_path.read_text()) if out_path.exists() else []
    for tag, alpha in (("haplo", 0.0), ("haplo+lda", args.alpha)):
        rows = [r for r in rows if r["config"] != tag]
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            model = Probed(in_ch=CHANNELS["haplo"]).to(device)
            model = train(model, xtr, ytr, device, args.epochs, 1e-3, 32, alpha)
            acc, ts = evaluate(model, xte, yte, device)
            rows.append({"pops": pops, "fst": fst, "config": tag, "seed": seed,
                         "alpha": alpha, "acc": acc, **ts})
            print(f"    {tag:<10} seed {seed}: acc={acc:.4f}  "
                  f"tracts={ts['n_tract_ratio']:.1f}x", flush=True)
            OUT.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(rows, indent=2))

    base = {r["seed"]: r["acc"] for r in rows if r["config"] == "haplo"}
    d = [r["acc"] - base[r["seed"]] for r in rows
         if r["config"] == "haplo+lda" and r["seed"] in base]
    print(f"\n  {'/'.join(pops)} paired delta {np.mean(d):+.4f} "
          f"(positive on {sum(1 for x in d if x > 0)}/{len(d)})", flush=True)


if __name__ == "__main__":
    main()

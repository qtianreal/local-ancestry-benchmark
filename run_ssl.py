"""Self-supervised pretraining of the encoder, on real haplotypes.

Pretraining on simulation helps the frequency-only network and does nothing for
the haplotype-aware one, because the haplotype-match channels are computed
against reference panels and simulated panels carry more sharing than real
ones. Self-supervised pretraining avoids that entirely: the encoder is
pretrained on real haplotype windows with no labels and no simulation, so
there is no domain to shift between.

The encoder is the dilated stack; a small convolutional decoder reconstructs
the masked input from its output. Contiguous spans are masked rather than
isolated sites, since a network with a 2045-site receptive field can
interpolate a single missing site from its neighbours and learn nothing.

Three arms, paired on seed:

    base        the published training budget, supervised only
    ssl+base    encoder pretrained by masked reconstruction on unlabelled
                windows, then fine-tuned on the same supervised budget
    more        the same *supervised* budget as ssl+base saw windows

The third arm is the control that matters. Without it, any gain from the
second could simply be more exposure to the training segment rather than
anything self-supervision contributes.
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


class Encoder(DilatedCNN):
    """DilatedCNN exposing its pre-head features."""

    def features(self, x):
        h = self.stem(x)
        for block in self.blocks:
            h = h + block(h)
        return h


def mask_spans(x, rng, frac=0.25, span=64):
    """Blank contiguous spans; return the corrupted copy and the mask."""
    n, c, L = x.shape
    m = torch.zeros(n, 1, L, dtype=torch.bool)
    n_span = max(1, int(frac * L / span))
    for i in range(n):
        for s in rng.integers(0, L - span, size=n_span):
            m[i, 0, s:s + span] = True
    xc = x.clone()
    xc = xc.masked_fill(m.expand(-1, c, -1), 0.0)
    return xc, m


def pretrain(model, x, device, epochs, rng, lr=1e-3, batch=16):
    """Masked reconstruction. The decoder is discarded afterwards."""
    dec = nn.Sequential(nn.GroupNorm(8, 64), nn.GELU(),
                        nn.Conv1d(64, x.shape[1], 1)).to(device)
    opt = torch.optim.AdamW(list(model.parameters()) + list(dec.parameters()),
                            lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    for _ in range(epochs):
        model.train(); dec.train()
        perm = torch.randperm(x.shape[0])
        for i in range(0, len(perm), batch):
            xb = x[perm[i:i + batch]]
            xc, m = mask_spans(xb, rng)
            xb, xc, m = xb.to(device), xc.to(device), m.to(device)
            opt.zero_grad()
            rec = dec(model.features(xc))
            # Score only the masked positions: reconstructing visible input is
            # trivial and would dominate the loss.
            mm = m.expand(-1, xb.shape[1], -1)
            loss = ((rec - xb) ** 2)[mm].mean()
            loss.backward()
            opt.step()
        sch.step()
    return model


def fit(model, x, y, device, epochs, lr, batch=32):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    lf = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(x.shape[0])
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            lf(model(x[idx].to(device)), y[idx].to(device)).backward()
            opt.step()
        sch.step()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pops", default="CHB,CDX")
    ap.add_argument("--vcf", default="data/chr22.vcf.gz")
    ap.add_argument("--features", default="haplo")
    ap.add_argument("--windows", type=int, default=3, help="labelled windows per partition")
    ap.add_argument("--ssl-windows", type=int, default=12,
                    help="unlabelled windows per partition for pretraining")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--ssl-epochs", type=int, default=15)
    ap.add_argument("--reps", type=int, default=6)
    args = ap.parse_args()

    pops = args.pops.split(",")
    mode = args.features
    haps, positions = load_region(args.vcf, "chr22", 16_000_000, 51_000_000, pops)
    fst = hudson_fst(haps[pops[0]], haps[pops[1]])
    n = positions.size
    cut, buf = int(n * 0.60), int(n * 0.05)
    tr_sl, te_sl = slice(0, cut), slice(cut + buf, n)
    print(f"{'/'.join(pops)}: Fst={fst:.5f}  features={mode}", flush=True)

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

    def build(per_rep, seed_off):
        xs, ys = [], []
        for rep, s_tr, _ in segs:
            a, b = windows_from(s_tr, per_rep, np.random.default_rng(seed_off + rep), mode)
            xs.append(a); ys.append(b)
        return torch.cat(xs), torch.cat(ys)

    xtr, ytr = build(args.windows, 7)                 # published budget
    xmore, ymore = build(args.ssl_windows, 300)       # supervised control
    xssl, _ = build(args.ssl_windows, 300)            # same windows, unlabelled
    xe, ye = [], []
    for rep, _, s_te in segs:
        a, b = windows_from(s_te, 1, np.random.default_rng(999 + rep), mode)
        xe.append(a); ye.append(b)
    xte, yte = torch.cat(xe), torch.cat(ye)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"  supervised {tuple(xtr.shape)}  control {tuple(xmore.shape)}  "
          f"test {tuple(xte.shape)}", flush=True)

    ch = CHANNELS[mode]
    rows = []
    for tag in ("base", "ssl+base", "more"):
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            model = Encoder(in_ch=ch).to(device)
            if tag == "ssl+base":
                model = pretrain(model, xssl, device, args.ssl_epochs,
                                 np.random.default_rng(seed))
                model = fit(model, xtr, ytr, device, args.epochs, 3e-4)
            elif tag == "more":
                model = fit(model, xmore, ymore, device, args.epochs, 1e-3)
            else:
                model = fit(model, xtr, ytr, device, args.epochs, 1e-3)
            acc, ts = evaluate(model, xte, yte, device)
            rows.append({"pops": pops, "fst": fst, "features": mode, "config": tag,
                         "seed": seed, "acc": acc, **ts})
            print(f"    {tag:<9} seed {seed}: acc={acc:.4f}  "
                  f"tracts={ts['n_tract_ratio']:.1f}x", flush=True)
            OUT.mkdir(parents=True, exist_ok=True)
            (OUT / f"ssl_{'_'.join(pops)}_{mode}.json").write_text(
                json.dumps(rows, indent=2))

    print("\n--- paired on seed, vs base ---", flush=True)
    base = {r["seed"]: r["acc"] for r in rows if r["config"] == "base"}
    for tag in ("base", "ssl+base", "more"):
        sel = [r for r in rows if r["config"] == tag]
        d = [r["acc"] - base[r["seed"]] for r in sel]
        print(f"  {tag:<9} {np.mean([r['acc'] for r in sel]):.4f}   "
              f"paired delta {np.mean(d):+.4f} "
              f"(positive on {sum(1 for x in d if x > 0)}/{len(d)})", flush=True)


if __name__ == "__main__":
    main()

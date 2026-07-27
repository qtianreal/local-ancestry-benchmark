"""How much of the real-data deficit is architecture, and how much is the method?

We report a single architecture throughout, which leaves open whether the
network's loss to RFMix and FLARE on real haplotypes reflects the approach or
merely an unlucky configuration. This sweeps width, depth, window size,
optimisation and capacity on a real pair and asks how far the deficit closes.

The sweep selects on the same data the deficit was measured on, so the best
configuration here is an optimistic bound rather than an honest estimate of
real-data performance. It answers "could tuning plausibly close the gap?", not
"what does the tuned method achieve?".
"""

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from lai.methods import DilatedCNN, build_features, panel_frequencies
from lai.real import load_region, split_panel
from lai.sim import hudson_fst, make_admixed
from run_pilot import eval_cnn
from run_real import RealConfig

OUT = Path("results")

CONFIGS = [
    # name,            width, dilations,                        window, epochs, lr
    ("baseline",          64, (1,2,4,8,16,32,64,128,256),         4096, 15, 1e-3),
    ("wide",            128, (1,2,4,8,16,32,64,128,256),          4096, 15, 1e-3),
    ("narrow",           32, (1,2,4,8,16,32,64,128,256),          4096, 15, 1e-3),
    ("deeper",           64, (1,2,4,8,16,32,64,128,256,512),      4096, 15, 1e-3),
    ("shallow",          64, (1,2,4,8,16,32),                     4096, 15, 1e-3),
    ("long-window",      64, (1,2,4,8,16,32,64,128,256),          8192, 15, 1e-3),
    ("more-epochs",      64, (1,2,4,8,16,32,64,128,256),          4096, 40, 1e-3),
    ("lower-lr",         64, (1,2,4,8,16,32,64,128,256),          4096, 30, 3e-4),
]


def windows(seg, n, rng, width):
    f, l = [], []
    ns = seg["admixed"].shape[0]
    for _ in range(n):
        s = int(rng.integers(0, ns - width)); sl = slice(s, s + width)
        pa, pb = panel_frequencies(seg["ref_a"][sl], seg["ref_b"][sl])
        f.append(build_features(seg["admixed"][sl], pa, pb))
        l.append(seg["labels"][sl].T.astype(np.float32))
    return (torch.from_numpy(np.concatenate(f)),
            torch.from_numpy(np.concatenate(l)))


def train(x, y, device, epochs, lr, width, dils, batch=32, seed=0):
    torch.manual_seed(seed)
    m = DilatedCNN(in_ch=4, width=width, dilations=dils).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lf = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        m.train()
        perm = torch.randperm(x.shape[0])
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            lf(m(x[idx].to(device)), y[idx].to(device)).backward()
            opt.step()
        sch.step()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pops", default="CHB,CDX")
    ap.add_argument("--vcf", default="data/chr22.vcf.gz")
    ap.add_argument("--start", type=int, default=16_000_000)
    ap.add_argument("--end", type=int, default=51_000_000)
    args = ap.parse_args()

    pops = args.pops.split(",")
    haps, positions = load_region(args.vcf, "chr22", args.start, args.end, pops)
    fst = hudson_fst(haps[pops[0]], haps[pops[1]])
    print(f"{'/'.join(pops)}: {positions.size} sites, Fst={fst:.5f}", flush=True)

    rng = np.random.default_rng(4242)
    ra, da = split_panel(haps[pops[0]], 80, 80, rng)
    rb, db = split_panel(haps[pops[1]], 80, 80, rng)

    n = positions.size
    cut, buf = int(n * 0.60), int(n * 0.05)
    tr_sl, te_sl = slice(0, cut), slice(cut + buf, n)

    def seg(sl, seed):
        pos = positions[sl] - positions[sl][0]
        cfg = RealConfig(seq_length=float(pos[-1]), n_admixed=64)
        adm, lab = make_admixed(cfg, da[sl], db[sl], pos, np.random.default_rng(seed))
        return {"ref_a": ra[sl], "ref_b": rb[sl], "admixed": adm,
                "labels": lab, "positions": pos}

    seg_tr, seg_te = seg(tr_sl, 1), seg(te_sl, 2)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    rows = []

    for name, width, dils, win, epochs, lr in CONFIGS:
        xtr, ytr = windows(seg_tr, 18, np.random.default_rng(7), win)
        xte, yte = windows(seg_te, 6, np.random.default_rng(8), win)
        m = train(xtr, ytr, device, epochs, lr, width, dils)
        acc = eval_cnn(m, xte, yte, device)
        params = sum(p.numel() for p in m.parameters())
        rows.append({"config": name, "acc": acc, "params": params,
                     "width": width, "n_dilations": len(dils),
                     "window": win, "epochs": epochs, "lr": lr, "fst": fst})
        print(f"  {name:<13} acc={acc:.4f}  params={params/1000:.0f}k", flush=True)
        (OUT / f"tune_{'_'.join(pops)}.json").write_text(json.dumps(rows, indent=2))

    best = max(rows, key=lambda r: r["acc"])
    print(f"\nbest: {best['config']} at {best['acc']:.4f}")


if __name__ == "__main__":
    main()

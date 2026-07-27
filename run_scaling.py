"""Does more simulated training data improve real-data performance?

RFMix and FLARE consume reference haplotypes but have no training phase: they
cannot use labelled admixed examples at all. A learned method can, and
simulation supplies them without limit. This asks what that buys, by measuring
real-data accuracy as a function of simulated training budget, against the two
released tools as fixed reference points.

Three regimes per budget, all evaluated on the same held-out real segment:

  sim-only    trained on simulated data alone, applied to real haplotypes
              with no real training data at all
  real-only   trained on real-derived mosaics alone (the paper's setting),
              shown once since it does not depend on the simulated budget
  sim+real    pretrained on simulation, then fine-tuned on the real training
              segment

The simulated divergence is matched to the real pair's measured Fst. Selection
happens on a pair already evaluated elsewhere in this study, so these are
optimistic bounds rather than reportable accuracies.
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
from lai.sim import SimConfig, hudson_fst, make_admixed, simulate_replicate
from run_pilot import WINDOW, assemble, eval_cnn
from run_real import RealConfig, windows_from

OUT = Path("results")


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


def match_split_time(target_fst, seed=99):
    """Pick the split time whose Fst is closest to the real pair's."""
    best, best_d = None, 1e9
    for T in (25, 50, 75, 100, 150, 200):
        f = simulate_replicate(SimConfig(split_time=T), seed=seed)["fst"]
        if abs(f - target_fst) < best_d:
            best, best_d = (T, f), abs(f - target_fst)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pops", default="CHB,CDX")
    ap.add_argument("--vcf", default="data/chr22.vcf.gz")
    ap.add_argument("--budgets", default="2,5,10,20,40",
                    help="simulated replicates used for pretraining")
    ap.add_argument("--epochs", type=int, default=15)
    args = ap.parse_args()

    pops = args.pops.split(",")
    haps, positions = load_region(args.vcf, "chr22", 16_000_000, 51_000_000, pops)
    fst = hudson_fst(haps[pops[0]], haps[pops[1]])
    T, sim_fst = match_split_time(fst)
    print(f"{'/'.join(pops)}: Fst={fst:.5f}; matched simulated split T={T} "
          f"(Fst={sim_fst:.5f})", flush=True)

    n = positions.size
    cut, buf = int(n * 0.60), int(n * 0.05)
    tr_sl, te_sl = slice(0, cut), slice(cut + buf, n)
    rng = np.random.default_rng(4242)
    ra, da = split_panel(haps[pops[0]], 80, 80, rng)
    rb, db = split_panel(haps[pops[1]], 80, 80, rng)

    def seg(sl, seed):
        pos = positions[sl] - positions[sl][0]
        cfg = RealConfig(seq_length=float(pos[-1]), n_admixed=64)
        adm, lab = make_admixed(cfg, da[sl], db[sl], pos, np.random.default_rng(seed))
        return {"ref_a": ra[sl], "ref_b": rb[sl], "admixed": adm,
                "labels": lab, "positions": pos}

    seg_tr, seg_te = seg(tr_sl, 1), seg(te_sl, 2)
    xr, yr = windows_from(seg_tr, 18, np.random.default_rng(7))
    xte, yte = windows_from(seg_te, 6, np.random.default_rng(8))
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    torch.manual_seed(0)
    real_only = eval_cnn(fit(DilatedCNN().to(device), xr, yr, device,
                             args.epochs, 1e-3), xte, yte, device)
    print(f"  real-only baseline: {real_only:.4f}  ({xr.shape[0]} real examples)",
          flush=True)

    rows = []
    cfg = SimConfig(split_time=T)
    for budget in [int(b) for b in args.budgets.split(",")]:
        reps = [simulate_replicate(cfg, seed=606_000 + i) for i in range(budget)]
        xs, ys = assemble(reps, 3, np.random.default_rng(11))

        torch.manual_seed(0)
        m = fit(DilatedCNN().to(device), xs, ys, device, args.epochs, 1e-3)
        sim_only = eval_cnn(m, xte, yte, device)
        ft = eval_cnn(fit(m, xr, yr, device, max(args.epochs // 2, 3), 3e-4),
                      xte, yte, device)

        rows.append({"pops": pops, "fst": fst, "sim_split_time": T,
                     "budget_replicates": budget,
                     "sim_examples": int(xs.shape[0]),
                     "real_examples": int(xr.shape[0]),
                     "sim_only": sim_only, "sim_plus_real": ft,
                     "real_only": real_only})
        print(f"  budget {budget:>3} reps ({xs.shape[0]:>4} examples): "
              f"sim-only={sim_only:.4f}  sim+real={ft:.4f}", flush=True)
        (OUT / f"scaling_{'_'.join(pops)}.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

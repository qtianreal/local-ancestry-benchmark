"""Does pretraining on simulation help the haplotype-aware network on real data?

The frequency-only version of this question was asked in run_scaling.py. The
haplotype-aware configuration is not obviously the same case: its extra
channels are computed against reference panels, and simulated panels carry
measurably more exploitable haplotype sharing than real ones at matched
divergence. The input a network sees may therefore be more domain-shifted with
the channels than without, so pretraining could transfer worse rather than
better.

Both configurations are run side by side on the same pair and the same
evaluation segment, in three regimes:

    sim-only    trained on simulation alone, applied to real haplotypes
    real-only   trained on the real training segment alone
    sim+real    pretrained on simulation, fine-tuned on the real segment

The simulated divergence is matched to the pair's measured Fst. The comparison
that matters is sim+real against real-only within a configuration: whether the
simulated pretraining adds anything on top of the real data already available.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from lai.methods import DilatedCNN, build_features, panel_frequencies
from lai.real import load_region, split_panel
from lai.sim import SimConfig, hudson_fst, simulate_replicate
from run_inputs import CHANNELS, evaluate, windows_from
from run_pilot import WINDOW, crop
from run_real import build_segment
from run_scaling import match_split_time

OUT = Path("results/tuning")


def assemble(reps, per_rep, rng, with_haps):
    """Simulated training tensors, with or without haplotype channels."""
    feats, labels = [], []
    for rep in reps:
        n = rep["admixed"].shape[0]
        for _ in range(per_rep):
            win = crop(rep, int(rng.integers(0, n - WINDOW)))
            pa, pb = panel_frequencies(win["ref_a"], win["ref_b"])
            feats.append(build_features(win["admixed"], pa, pb,
                                        win["ref_a"] if with_haps else None,
                                        win["ref_b"] if with_haps else None))
            labels.append(win["labels"].T.astype(np.float32))
    return (torch.from_numpy(np.concatenate(feats)),
            torch.from_numpy(np.concatenate(labels)))


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
    ap.add_argument("--budget", type=int, default=20,
                    help="simulated replicates used for pretraining")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=15)
    args = ap.parse_args()

    pops = args.pops.split(",")
    haps, positions = load_region(args.vcf, "chr22", 16_000_000, 51_000_000, pops)
    fst = hudson_fst(haps[pops[0]], haps[pops[1]])
    T, sim_fst = match_split_time(fst)
    print(f"{'/'.join(pops)}: Fst={fst:.5f}; matched simulated split T={T} "
          f"(Fst={sim_fst:.5f}); budget {args.budget} replicates", flush=True)

    n = positions.size
    cut, buf = int(n * 0.60), int(n * 0.05)
    tr_sl, te_sl = slice(0, cut), slice(cut + buf, n)
    rng = np.random.default_rng(4242)
    ra, da = split_panel(haps[pops[0]], 80, 80, rng)
    rb, db = split_panel(haps[pops[1]], 80, 80, rng)
    seg_tr, _ = build_segment(ra[tr_sl], rb[tr_sl], da[tr_sl], db[tr_sl],
                              positions[tr_sl], 64, rng)
    seg_te, _ = build_segment(ra[te_sl], rb[te_sl], da[te_sl], db[te_sl],
                              positions[te_sl], 64, rng)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    reps = [simulate_replicate(SimConfig(split_time=T), seed=606_000 + i)
            for i in range(args.budget)]

    rows = []
    for mode, with_haps in (("freq", False), ("haplo", True)):
        ch = CHANNELS[mode]
        xr, yr = windows_from(seg_tr, 18, np.random.default_rng(7), mode)
        xte, yte = windows_from(seg_te, 6, np.random.default_rng(8), mode)
        xs, ys = assemble(reps, 3, np.random.default_rng(11), with_haps)
        print(f"  [{mode}] real train {tuple(xr.shape)}  "
              f"simulated {tuple(xs.shape)}", flush=True)

        for seed in range(args.seeds):
            torch.manual_seed(seed)
            real_only = evaluate(fit(DilatedCNN(in_ch=ch).to(device), xr, yr,
                                     device, args.epochs, 1e-3), xte, yte, device)[0]
            torch.manual_seed(seed)
            m = fit(DilatedCNN(in_ch=ch).to(device), xs, ys, device,
                    args.epochs, 1e-3)
            sim_only = evaluate(m, xte, yte, device)[0]
            # Fine-tune the pretrained network on the same real data the
            # real-only arm saw, at a reduced rate.
            sim_real = evaluate(fit(m, xr, yr, device, max(args.epochs // 2, 3),
                                    3e-4), xte, yte, device)[0]
            rows.append({"pops": pops, "fst": fst, "features": mode, "seed": seed,
                         "budget": args.budget, "sim_only": sim_only,
                         "real_only": real_only, "sim_plus_real": sim_real})
            print(f"    {mode:<6} seed {seed}: sim-only={sim_only:.4f}  "
                  f"real-only={real_only:.4f}  sim+real={sim_real:.4f}  "
                  f"({sim_real - real_only:+.4f})", flush=True)
            OUT.mkdir(parents=True, exist_ok=True)
            (OUT / f"scalinghap_{'_'.join(pops)}.json").write_text(
                json.dumps(rows, indent=2))

    print("\n--- pretraining benefit, paired on seed ---", flush=True)
    for mode in ("freq", "haplo"):
        sel = [r for r in rows if r["features"] == mode]
        d = [r["sim_plus_real"] - r["real_only"] for r in sel]
        so = [r["sim_only"] for r in sel]
        ro = [r["real_only"] for r in sel]
        print(f"  {mode:<6} real-only {np.mean(ro):.4f}   sim-only {np.mean(so):.4f}"
              f"   sim+real minus real-only {np.mean(d):+.4f} "
              f"(positive on {sum(1 for x in d if x > 0)}/{len(d)})", flush=True)


if __name__ == "__main__":
    main()

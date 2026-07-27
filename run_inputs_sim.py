"""The haplotype-feature comparison in simulation, properly powered.

run_haplo.py ran one training seed per divergence level and reported a mean
delta of +0.002 across six levels. The per-level deltas ranged from -0.013 to
+0.015, which we now know is entirely within seed noise, so that experiment
could not distinguish "no effect" from an effect the size of the one measured
on real haplotypes (+0.023 mean). It is under-powered rather than negative.

This repeats it with five seeds per level, paired: the two feature sets share
simulations, windows and initialisation, so the difference isolates the input
representation exactly as in the real-data sweep. Tract statistics are
recorded alongside accuracy for the same reason as there.

The comparison matters because the manuscript's explanation for the real-data
result rests on it. If haplotype features help on real chromosomes and not in
simulation, a clean split understates the haplotype sharing real data carries,
and simulation-only benchmarking flatters frequency-based methods. That
argument needs the simulation null to be trustworthy, not merely unmeasured.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lai.methods import DilatedCNN, build_features, panel_frequencies
from lai.sim import SimConfig, simulate_replicate
from run_attn import train as fit
from run_inputs import evaluate
from run_pilot import WINDOW

OUT = Path("results/tuning")


def windows(reps, n_per_rep, rng, with_haps):
    feats, labs = [], []
    for rep in reps:
        n_sites = rep["admixed"].shape[0]
        for _ in range(n_per_rep):
            s = int(rng.integers(0, n_sites - WINDOW))
            sl = slice(s, s + WINDOW)
            ra, rb = rep["ref_a"][sl], rep["ref_b"][sl]
            pa, pb = panel_frequencies(ra, rb)
            feats.append(build_features(rep["admixed"][sl], pa, pb,
                                        ra if with_haps else None,
                                        rb if with_haps else None))
            labs.append(rep["labels"][sl].T.astype(np.float32))
    return (torch.from_numpy(np.concatenate(feats)),
            torch.from_numpy(np.concatenate(labs)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-times", default="25,50,100,200,400,800,1600,3200")
    ap.add_argument("--n-train", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=15)
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    rows = []
    out = OUT / "inputs_sim.json"

    for T in [int(x) for x in args.split_times.split(",")]:
        cfg = SimConfig(split_time=T)
        # Same simulation seeds as run_haplo.py and run_external.py, so this is
        # the same data the published simulation numbers were computed on.
        test = [simulate_replicate(cfg, seed=777_000 + T + 31 * i) for i in range(3)]
        train_reps = [simulate_replicate(cfg, seed=778_000 + T * 7 + i)
                      for i in range(args.n_train)]
        fst = test[0]["fst"]
        print(f"T={T} Fst={fst:.5f}", flush=True)

        for tag, with_haps, in_ch in (("freq", False, 4), ("haplo", True, 8)):
            xtr, ytr = windows(train_reps, 3, np.random.default_rng(T), with_haps)
            xte, yte = windows(test, 4, np.random.default_rng(5000 + T), with_haps)
            for seed in range(args.seeds):
                torch.manual_seed(seed)
                model = DilatedCNN(in_ch=in_ch).to(device)
                model = fit(model, xtr, ytr, device, args.epochs, 1e-3, 32)
                acc, ts = evaluate(model, xte, yte, device)
                rows.append({"split_time": T, "fst": fst, "features": tag,
                             "seed": seed, "in_ch": in_ch, "acc": acc, **ts})
                print(f"    {tag:<6} seed {seed}: acc={acc:.4f}  "
                      f"tracts={ts['n_tract_ratio']:.1f}x", flush=True)
                OUT.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(rows, indent=2))

        base = {r["seed"]: r["acc"] for r in rows
                if r["split_time"] == T and r["features"] == "freq"}
        hap = [r for r in rows if r["split_time"] == T and r["features"] == "haplo"]
        d = [r["acc"] - base[r["seed"]] for r in hap]
        print(f"  T={T} paired delta {np.mean(d):+.4f} "
              f"(positive on {sum(1 for x in d if x > 0)}/{len(d)})", flush=True)

    print("\n--- simulation, paired on seed ---", flush=True)
    for T in sorted({r["split_time"] for r in rows}):
        base = {r["seed"]: r["acc"] for r in rows
                if r["split_time"] == T and r["features"] == "freq"}
        hap = [r for r in rows if r["split_time"] == T and r["features"] == "haplo"]
        d = np.array([r["acc"] - base[r["seed"]] for r in hap])
        f = [r["fst"] for r in hap][0]
        print(f"  T={T:<5} Fst={f:.5f}  delta {d.mean():+.4f} +/- {d.std(ddof=1):.4f}",
              flush=True)


if __name__ == "__main__":
    main()

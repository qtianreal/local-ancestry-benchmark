"""Haplotype-aware CNN sweep.

Identical protocol to the frequency-only sweep, except that build_features is
given the reference panels so the network additionally receives local
haplotype-matching summaries. Seeds are chosen to match run_external.py
exactly, so accuracies are directly comparable to the RFMix and FLARE numbers
computed on the same test replicates.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from lai.methods import DilatedCNN, build_features, panel_frequencies
from lai.sim import SimConfig, simulate_replicate
from run_pilot import WINDOW, eval_cnn

OUT = Path("results")
IN_CH = 8


def windows(reps, n_per_rep, rng, with_haps=True):
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


def train(x, y, device, epochs, in_ch, batch=32, seed=0):
    torch.manual_seed(seed)
    model = DilatedCNN(in_ch=in_ch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.BCEWithLogitsLoss()
    n = x.shape[0]
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            opt.zero_grad()
            loss = lossf(model(x[idx].to(device)), y[idx].to(device))
            loss.backward()
            opt.step()
        sched.step()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-times", default="25,50,100,200,400,800,1600,3200")
    ap.add_argument("--n-train", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=15)
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    rows = []

    for T in [int(x) for x in args.split_times.split(",")]:
        t0 = time.time()
        cfg = SimConfig(split_time=T)
        # Same seeds as run_external.py so the comparison is on identical data.
        test = [simulate_replicate(cfg, seed=777_000 + T + 31 * i)
                for i in range(3)]
        train_reps = [simulate_replicate(cfg, seed=778_000 + T * 7 + i)
                      for i in range(args.n_train)]

        # Matched ablation: identical simulations, windows, budget and seed;
        # the only difference is whether the reference panels are exposed.
        acc = {}
        for tag, with_haps, in_ch in (("freq", False, 4), ("haplo", True, 8)):
            xtr, ytr = windows(train_reps, 3, np.random.default_rng(T), with_haps)
            xte, yte = windows(test, 4, np.random.default_rng(5000 + T), with_haps)
            model = train(xtr, ytr, device, args.epochs, in_ch, seed=T)
            acc[tag] = eval_cnn(model, xte, yte, device)
            torch.save(model.state_dict(), OUT / "cache" / f"cnn_{tag}_T{T}.pt")
            n_train_ex = int(xtr.shape[0])

        rows.append({"split_time": T, "fst": test[0]["fst"],
                     "cnn_freq": acc["freq"], "cnn_haplo": acc["haplo"],
                     "delta": acc["haplo"] - acc["freq"],
                     "train_examples": n_train_ex,
                     "test_examples": int(xte.shape[0]),
                     "seconds": round(time.time() - t0, 1)})
        print(f"T={T:5d} Fst={test[0]['fst']:.5f}  freq={acc['freq']:.4f}  "
              f"haplo={acc['haplo']:.4f}  delta={acc['haplo']-acc['freq']:+.4f}  "
              f"({rows[-1]['seconds']}s)", flush=True)
        (OUT / "haplo_results.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

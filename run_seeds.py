"""Seed-replicated divergence sweep.

The original sweep is treated as seed 0. This script reruns the identical
protocol under additional independent seeds -- new coalescent simulations, new
admixture mosaics, new network initialisation -- so that the shape of the
accuracy curve can be separated from run-to-run noise.

Everything else (divergence grid, architecture, optimiser, epoch count,
evaluation windows) is held fixed.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from lai.sim import SimConfig, simulate_replicate
from run_pilot import (
    WINDOW,
    assemble,
    baseline_scores,
    crop,
    eval_cnn,
    train_cnn,
)

OUT = Path("results")
SPLIT_TIMES = [25, 50, 100, 200, 400, 800, 1600, 3200]
N_TRAIN, N_TEST, WIN_PER_REP, EPOCHS = 20, 8, 3, 15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True, help="replicate index (0 = original run)")
    args = ap.parse_args()
    s = args.seed
    off = s * 1_000_000  # keeps simulation seeds disjoint across replicates

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    (OUT / "cache").mkdir(parents=True, exist_ok=True)
    rows = []

    for T in SPLIT_TIMES:
        t0 = time.time()
        cfg = SimConfig(split_time=T)

        train_reps = [simulate_replicate(cfg, seed=off + 10_000 + T * 13 + i)
                      for i in range(N_TRAIN)]
        test_reps = [simulate_replicate(cfg, seed=off + 90_000 + T * 13 + i)
                     for i in range(N_TEST)]
        fst = float(np.mean([r["fst"] for r in train_reps]))

        nb_acc, hmm_acc = [], []
        for rep in test_reps:
            start = (rep["admixed"].shape[0] - WINDOW) // 2
            nb, hmm, _ = baseline_scores(crop(rep, start), cfg.admix_generations)
            nb_acc.append(nb)
            hmm_acc.append(hmm)

        xtr, ytr = assemble(train_reps, WIN_PER_REP, np.random.default_rng(off + 1000 + T))
        xte, yte = assemble(test_reps, 1, np.random.default_rng(off + 5000 + T))

        model, _ = train_cnn(xtr, ytr, device, EPOCHS, seed=off + T)
        cnn_acc = eval_cnn(model, xte, yte, device)

        torch.save(model.state_dict(), OUT / "cache" / f"cnn_T{T}_s{s}.pt")
        np.savez_compressed(OUT / "cache" / f"test_T{T}_s{s}.npz",
                            x=xte.numpy().astype(np.float16),
                            y=yte.numpy().astype(np.int8))

        rows.append({
            "seed": s, "split_time": T, "fst": fst,
            "naive_bayes": float(np.mean(nb_acc)),
            "naive_bayes_sd": float(np.std(nb_acc)),
            "hmm": float(np.mean(hmm_acc)),
            "hmm_sd": float(np.std(hmm_acc)),
            "cnn": cnn_acc,
            "seconds": round(time.time() - t0, 1),
        })
        print(f"[seed {s}] T={T:5d} Fst={fst:.5f}  NB={rows[-1]['naive_bayes']:.4f} "
              f"HMM={rows[-1]['hmm']:.4f} CNN={cnn_acc:.4f}  ({rows[-1]['seconds']}s)",
              flush=True)
        (OUT / f"main_results_s{s}.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

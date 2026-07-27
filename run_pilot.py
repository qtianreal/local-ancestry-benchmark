"""Main experiment: local-ancestry accuracy as a function of source divergence.

For each divergence level we simulate independent replicates, hold out a
disjoint set for testing, and evaluate three methods on identical windows.
Trained CNN weights and test tensors are cached so that the cross-divergence
transfer experiment can reuse them without re-simulating.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from lai.methods import (
    DilatedCNN,
    build_features,
    hmm_predict,
    naive_bayes_predict,
    panel_frequencies,
    site_log_ratio,
)
from lai.sim import RHO, SimConfig, simulate_replicate

WINDOW = 4096  # SNPs per evaluation window
NB_WINDOW = 128  # SNPs per naive-Bayes / HMM window
OUT = Path("results")


def crop(rep, start, width=WINDOW):
    sl = slice(start, start + width)
    return {
        "ref_a": rep["ref_a"][sl],
        "ref_b": rep["ref_b"][sl],
        "admixed": rep["admixed"][sl],
        "labels": rep["labels"][sl],
        "positions": rep["positions"][sl],
    }


def baseline_scores(win, admix_generations):
    """Per-site accuracy of the two likelihood baselines on one window."""
    p_a, p_b = panel_frequencies(win["ref_a"], win["ref_b"])
    lr = site_log_ratio(win["admixed"], p_a, p_b)

    nb_pred, win_score = naive_bayes_predict(lr, NB_WINDOW)
    n_used = nb_pred.shape[0]
    truth = win["labels"][:n_used]

    span = np.diff(win["positions"][[0, -1]])[0] / (len(win["positions"]) / NB_WINDOW)
    switch_prob = float(np.clip(admix_generations * RHO * span, 1e-6, 0.4))

    hmm_win = hmm_predict(win_score, switch_prob)
    hmm_pred = np.repeat(hmm_win, NB_WINDOW, axis=0)

    return (
        float((nb_pred == truth).mean()),
        float((hmm_pred == truth).mean()),
        switch_prob,
    )


def assemble(reps, windows_per_rep, rng):
    """Build CNN tensors from a list of replicates."""
    feats, labels = [], []
    for rep in reps:
        n_sites = rep["admixed"].shape[0]
        for _ in range(windows_per_rep):
            start = int(rng.integers(0, n_sites - WINDOW))
            win = crop(rep, start)
            p_a, p_b = panel_frequencies(win["ref_a"], win["ref_b"])
            feats.append(build_features(win["admixed"], p_a, p_b))
            labels.append(win["labels"].T.astype(np.float32))
    return (
        torch.from_numpy(np.concatenate(feats)),
        torch.from_numpy(np.concatenate(labels)),
    )


def train_cnn(x, y, device, epochs, batch=32, seed=0, width=64, dilations=None):
    torch.manual_seed(seed)
    model = (DilatedCNN(width=width, dilations=dilations) if dilations
             else DilatedCNN(width=width)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.BCEWithLogitsLoss()

    n = x.shape[0]
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            xb, yb = x[idx].to(device), y[idx].to(device)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        sched.step()
    return model, total / n


@torch.no_grad()
def eval_cnn(model, x, y, device, batch=64):
    model.eval()
    correct = tot = 0
    for i in range(0, x.shape[0], batch):
        xb = x[i : i + batch].to(device)
        pred = (torch.sigmoid(model(xb)) > 0.5).float().cpu()
        correct += (pred == y[i : i + batch]).sum().item()
        tot += y[i : i + batch].numel()
    return correct / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    split_times = [25, 50, 100, 200, 400, 800, 1600, 3200]
    n_train_reps, n_test_reps, win_per_rep, epochs = 20, 8, 3, 15
    if args.quick:
        split_times = [50, 800]
        n_train_reps, n_test_reps, win_per_rep, epochs = 4, 2, 2, 3

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    OUT.mkdir(exist_ok=True)
    (OUT / "cache").mkdir(exist_ok=True)
    rows = []

    for T in split_times:
        t0 = time.time()
        cfg = SimConfig(split_time=T)
        rng = np.random.default_rng(1000 + T)

        train_reps = [simulate_replicate(cfg, seed=10_000 + T * 13 + i)
                      for i in range(n_train_reps)]
        test_reps = [simulate_replicate(cfg, seed=90_000 + T * 13 + i)
                     for i in range(n_test_reps)]
        fst = float(np.mean([r["fst"] for r in train_reps]))

        # Baselines on a deterministic centred window of each test replicate.
        nb_acc, hmm_acc, switch = [], [], None
        for rep in test_reps:
            start = (rep["admixed"].shape[0] - WINDOW) // 2
            nb, hmm, switch = baseline_scores(crop(rep, start), cfg.admix_generations)
            nb_acc.append(nb)
            hmm_acc.append(hmm)

        xtr, ytr = assemble(train_reps, win_per_rep, rng)
        xte, yte = assemble(test_reps, 1, np.random.default_rng(5000 + T))

        model, final_loss = train_cnn(xtr, ytr, device, epochs, seed=T)
        cnn_acc = eval_cnn(model, xte, yte, device)

        torch.save(model.state_dict(), OUT / "cache" / f"cnn_T{T}.pt")
        np.savez_compressed(OUT / "cache" / f"test_T{T}.npz",
                            x=xte.numpy().astype(np.float16), y=yte.numpy().astype(np.int8))

        row = {
            "split_time": T,
            "fst": fst,
            "naive_bayes": float(np.mean(nb_acc)),
            "naive_bayes_sd": float(np.std(nb_acc)),
            "hmm": float(np.mean(hmm_acc)),
            "hmm_sd": float(np.std(hmm_acc)),
            "cnn": cnn_acc,
            "hmm_switch_prob": switch,
            "train_examples": int(xtr.shape[0]),
            "test_examples": int(xte.shape[0]),
            "train_loss": final_loss,
            "seconds": round(time.time() - t0, 1),
        }
        rows.append(row)
        print(f"T={T:5d} Fst={fst:.5f}  NB={row['naive_bayes']:.4f} "
              f"HMM={row['hmm']:.4f} CNN={cnn_acc:.4f}  ({row['seconds']}s)", flush=True)

        with open(OUT / "main_results.json", "w") as fh:
            json.dump(rows, fh, indent=2)


if __name__ == "__main__":
    main()

"""Does the Viterbi fix survive not knowing the admixture age?

The tract-fragmentation result decodes the network's per-site scores under a
transition prior set from the true pulse age. An applied analysis does not know
that age, so the question is how much of the fix survives misspecifying it. We
decode the same cached logits under assumed ages of 10, 30 and 100 generations
against a true 30, changing nothing else.

The same run answers a second question. Tract-count ratio says how many tracts
a method produces but not whether the breakpoints are in the right places, and
nearest-breakpoint distance flatters a method that predicts many spurious
breakpoints. We therefore also score breakpoints by precision, recall and F1
under a fixed tolerance, which penalises missed and spurious breakpoints
alike.

No retraining: the networks are loaded from results/cache and only the decoding
changes.

    python run_decode_robustness.py            # 8 divergence levels x 3 seeds
    python run_decode_robustness.py --quick    # one level, for a timing check

Writes results/decode_robustness.json.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lai.decode import viterbi_decode
from lai.export import _dedup_positions
from lai.methods import DilatedCNN
from lai.sim import RHO, SimConfig, simulate_replicate
from run_crf import site_logits
from run_tracts import tract_stats

OUT = Path("results")
SPLITS = [25, 50, 100, 200, 400, 800, 1600, 3200]
SEEDS = [0, 1, 2]
ASSUMED_G = [10, 30, 100]      # true value is 30
# Tolerance in physical distance, which with rho = 1e-8 per bp is also
# centimorgans: 100 kb = 0.1 cM and 1 Mb = 1 cM. A tract at g = 30 spans about
# 1/30 Morgan, so these are roughly 3% and 30% of a tract. Counting in
# segregating sites instead would make the tolerance depend on marker density,
# which is one of the quantities under study.
TOLERANCES_BP = [100_000, 1_000_000]


def breakpoint_prf(pred, truth, pos, tol_bp):
    """Precision, recall and F1 for breakpoint placement.

    A predicted breakpoint counts once if it falls within tol_bp base pairs
    of a true breakpoint, and each true breakpoint can be matched only
    once, so spurious breakpoints cost precision and missed ones cost recall.
    That is what nearest-breakpoint distance fails to do: a method predicting
    many breakpoints scores well on it by chance.
    """
    tp = fp = fn = 0
    for j in range(truth.shape[1]):
        p, t = pred[:, j], truth[:, j]
        if (p < 0).any():
            continue
        pb = pos[np.flatnonzero(np.diff(p) != 0)]
        tb = pos[np.flatnonzero(np.diff(t) != 0)]
        used = np.zeros(len(tb), dtype=bool)
        matched = 0
        for b in pb:
            if not len(tb):
                break
            d = np.abs(tb - b).astype(float)
            d[used] = np.inf
            k = int(np.argmin(d))
            if d[k] <= tol_bp:
                used[k] = True
                matched += 1
        tp += matched
        fp += len(pb) - matched
        fn += len(tb) - matched
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = (2 * prec * rec / (prec + rec)) if prec + rec else float("nan")
    return prec, rec, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    splits = SPLITS[2:3] if args.quick else SPLITS
    seeds = SEEDS[:1] if args.quick else SEEDS

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    rows = []
    for T in splits:
        cfg = SimConfig(split_time=T)
        for seed in seeds:
            ck = OUT / "cache" / f"cnn_T{T}_s{seed}.pt"
            if not ck.exists():
                continue
            rep = simulate_replicate(cfg, seed=777_000 + T)
            pos, keep = _dedup_positions(rep["positions"])
            ra, rb = rep["ref_a"][keep], rep["ref_b"][keep]
            adm, truth = rep["admixed"][keep], rep["labels"][keep]

            model = DilatedCNN().to(device)
            model.load_state_dict(torch.load(ck, map_location=device))
            model.eval()
            logits, last = site_logits(model, adm, ra, rb, device)
            p, tr = pos[:last], truth[:last]

            preds = {"raw": (logits > 0).astype(np.int8)}
            for g in ASSUMED_G:
                preds[f"viterbi_g{g}"] = viterbi_decode(logits, p, g, rho=RHO)

            for tag, pred in preds.items():
                st = tract_stats(pred, tr)
                rec = {"split_time": T, "seed": seed, "fst": rep["fst"],
                       "decoding": tag, "true_g": cfg.admix_generations,
                       "acc": float((pred == tr).mean()),
                       "n_tract_ratio": st["n_tract_ratio"]}
                for tol in TOLERANCES_BP:
                    pr, rc, f1 = breakpoint_prf(pred, tr, p, tol)
                    kb = tol // 1000
                    rec[f"prec_{kb}kb"] = pr
                    rec[f"rec_{kb}kb"] = rc
                    rec[f"f1_{kb}kb"] = f1
                rows.append(rec)
            print(f"  T={T:<5} seed={seed}  " + "  ".join(
                f"{k}: tract {r['n_tract_ratio']:.1f}x F1 {r['f1_1000kb']:.2f}"
                for k, r in ((x["decoding"], x) for x in rows[-len(preds):])),
                flush=True)

    OUT.mkdir(exist_ok=True)
    (OUT / "decode_robustness.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {OUT/'decode_robustness.json'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

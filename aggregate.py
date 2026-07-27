"""Combine seed replicates into the numbers the manuscript quotes.

Two sources of variation are distinguished:

  within-seed : spread across the 8 held-out test replicates of a single run
                (already reported per seed for the likelihood baselines)
  across-seed : spread across independent simulations + initialisations

Only the second speaks to whether the shape of the curve is real, so it is
what the manuscript reports as error.
"""

import json
from pathlib import Path

import numpy as np

OUT = Path("results")
METHODS = ["naive_bayes", "hmm", "cnn"]


def load_seeds():
    seeds, rows = [], {}
    for p in sorted(OUT.glob("main_results_s*.json")):
        s = int(p.stem.split("_s")[-1])
        seeds.append(s)
        for r in json.loads(p.read_text()):
            rows.setdefault(r["split_time"], []).append(r)
    return sorted(seeds), rows


def main():
    seeds, by_T = load_seeds()
    print(f"seeds found: {seeds}")

    agg = []
    for T in sorted(by_T):
        reps = by_T[T]
        rec = {"split_time": T, "n_seeds": len(reps),
               "fst": float(np.mean([r["fst"] for r in reps])),
               "fst_sd": float(np.std([r["fst"] for r in reps]))}
        for m in METHODS:
            v = np.array([r[m] for r in reps])
            rec[m] = float(v.mean())
            rec[f"{m}_sd"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
            rec[f"{m}_vals"] = [float(x) for x in v]
        gap = np.array([r["cnn"] - r["hmm"] for r in reps])
        rec["gap"] = float(gap.mean())
        rec["gap_sd"] = float(gap.std(ddof=1)) if len(gap) > 1 else 0.0
        agg.append(rec)

    (OUT / "aggregate_results.json").write_text(json.dumps(agg, indent=2))

    print(f"\n{'Fst':>8} {'NB':>14} {'HMM':>14} {'CNN':>14} {'CNN-HMM':>14}")
    for r in agg:
        print(f"{r['fst']:8.5f} "
              f"{r['naive_bayes']:.3f}+/-{r['naive_bayes_sd']:.3f} "
              f"{r['hmm']:.3f}+/-{r['hmm_sd']:.3f} "
              f"{r['cnn']:.3f}+/-{r['cnn_sd']:.3f} "
              f"{r['gap']:+.3f}+/-{r['gap_sd']:.3f}")

    # Is the peak separated from both tails by more than seed noise?
    gaps = np.array([r["gap"] for r in agg])
    sds = np.array([r["gap_sd"] for r in agg])
    k = int(gaps.argmax())
    print(f"\npeak gap at Fst={agg[k]['fst']:.5f}: {gaps[k]:+.3f} +/- {sds[k]:.3f}")
    for end, idx in (("low", 0), ("high", len(agg) - 1)):
        pooled = float(np.hypot(sds[k], sds[idx]))
        diff = gaps[k] - gaps[idx]
        z = diff / pooled if pooled > 0 else float("inf")
        print(f"  vs {end:>4} tail (Fst={agg[idx]['fst']:.5f}): "
              f"delta={diff:+.3f}  pooled_sd={pooled:.3f}  z={z:.1f}")


if __name__ == "__main__":
    main()

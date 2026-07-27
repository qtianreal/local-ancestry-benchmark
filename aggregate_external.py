"""Combine external-tool replicates and test whether the simulated gap is real.

The central quantity is the per-level difference between the learned network
and the better of RFMix and FLARE. Single-run values of this gap proved
unstable below Fst ~ 0.01, so what matters is its mean and spread across
independent seeds, and whether the mean is separated from zero by more than
that spread.
"""

import json
from pathlib import Path

import numpy as np

OUT = Path("results")
METHODS = ["naive_bayes", "hmm", "rfmix", "flare", "cnn"]


def main():
    by_T = {}
    seeds = []
    for p in sorted(OUT.glob("external_results_s*.json")):
        s = int(p.stem.split("_s")[-1])
        seeds.append(s)
        for r in json.loads(p.read_text()):
            by_T.setdefault(r["split_time"], []).append(r)
    if not by_T:
        print("no external replicates found")
        return
    print(f"seeds: {sorted(seeds)}")

    agg = []
    for T in sorted(by_T):
        reps = by_T[T]
        rec = {"split_time": T, "n_seeds": len(reps),
               "fst": float(np.mean([r["fst"] for r in reps]))}
        for m in METHODS:
            v = np.array([r[m] for r in reps if isinstance(r.get(m), float)])
            if v.size:
                rec[m] = float(v.mean())
                rec[f"{m}_sd"] = float(v.std(ddof=1)) if v.size > 1 else 0.0
        gaps = []
        for r in reps:
            ext = [r[k] for k in ("rfmix", "flare") if isinstance(r.get(k), float)]
            if ext and isinstance(r.get("cnn"), float):
                gaps.append(r["cnn"] - max(ext))
        if gaps:
            g = np.array(gaps)
            rec["gap"] = float(g.mean())
            rec["gap_sd"] = float(g.std(ddof=1)) if g.size > 1 else 0.0
            rec["gap_vals"] = [float(x) for x in g]
            rec["gap_t"] = (float(g.mean() / (g.std(ddof=1) / np.sqrt(g.size)))
                            if g.size > 1 and g.std(ddof=1) > 0 else float("nan"))
        agg.append(rec)

    (OUT / "external_aggregate.json").write_text(json.dumps(agg, indent=2))

    print(f"\n{'Fst':>9}{'n':>3}{'RFMix':>16}{'FLARE':>16}{'CNN':>16}"
          f"{'CNN-best':>18}{'t':>7}")
    for r in agg:
        f = lambda k: (f"{r[k]:.3f}+/-{r.get(k+'_sd', 0):.3f}" if k in r else "   --   ")
        gap = (f"{r['gap']:+.3f}+/-{r['gap_sd']:.3f}" if "gap" in r else "  --  ")
        tv = r.get("gap_t", float("nan"))
        print(f"{r['fst']:>9.5f}{r['n_seeds']:>3}{f('rfmix'):>16}{f('flare'):>16}"
              f"{f('cnn'):>16}{gap:>18}{tv:>7.1f}")

    lo = [r for r in agg if r["fst"] < 0.015 and "gap" in r]
    if lo:
        g = np.array([v for r in lo for v in r["gap_vals"]])
        print(f"\nlow-divergence levels (Fst < 0.015), pooled over {len(g)} runs:")
        print(f"  mean gap {g.mean():+.4f}  sd {g.std(ddof=1):.4f}  "
              f"se {g.std(ddof=1)/np.sqrt(g.size):.4f}")
        print(f"  runs favouring the network: {(g > 0).sum()}/{g.size}")


if __name__ == "__main__":
    main()

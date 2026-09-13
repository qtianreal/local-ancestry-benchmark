"""Replication of the released tools across reference/donor partitions.

Every released-tool number elsewhere in this study is a single run on one
partition of each population into reference and donor haplotypes. This script
repeats four of the low-divergence pairs over five partitions and asks how much
of the difference between methods survives the draw.

The comparison that matters is between two spreads: the range a single method
covers across partitions, and the gap between methods on any one partition. If
the first exceeds the second, a single run cannot rank the methods.
"""

import json
import re
from pathlib import Path

import numpy as np

RES = Path("results")
OUT = RES / "draws_summary.json"

PAIRS = ("CEU_TSI", "FIN_GBR", "CHB_JPT", "FIN_TSI")
SEEDS = ("", "_s1", "_s2", "_s3", "_s4")  # "" is seed 4242, the published draw
METHODS = ("rfmix", "flare", "recombmix")


def collect(tag):
    """Per-method accuracy across partitions; None where a run is missing."""
    out = {m: [] for m in METHODS}
    fst = None
    for s in SEEDS:
        ext = RES / f"realext_{tag}{s}.json"
        rm = RES / f"recombmix_{tag}{s}.json"
        e = json.loads(ext.read_text()) if ext.exists() else {}
        r = json.loads(rm.read_text()) if rm.exists() else {}
        fst = fst or e.get("fst") or r.get("fst")
        for m in METHODS:
            v = r.get("recombmix") if m == "recombmix" else e.get(m)
            out[m].append(v if isinstance(v, float) else None)
    return fst, out


def main():
    recs, all_spreads, all_gaps = [], [], []
    for tag in PAIRS:
        fst, by = collect(tag)
        rec = {"tag": tag, "pair": tag.replace("_", "/"), "fst": fst, "methods": {}}
        for m in METHODS:
            v = [x for x in by[m] if x is not None]
            if not v:
                continue
            rec["methods"][m] = {
                "n": len(v), "mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))
                if len(v) > 1 else None, "min": min(v), "max": max(v),
                "spread": max(v) - min(v), "values": v,
            }
            if len(v) > 1:
                all_spreads.append(max(v) - min(v))
        # Per-partition gap between the best and worst method, which is the
        # quantity a single run reports as a difference between methods.
        for i in range(len(SEEDS)):
            vals = [by[m][i] for m in METHODS if by[m][i] is not None]
            if len(vals) > 1:
                all_gaps.append(max(vals) - min(vals))
        # Does the winner change with the partition?
        winners = []
        for i in range(len(SEEDS)):
            vals = {m: by[m][i] for m in METHODS if by[m][i] is not None}
            if vals:
                winners.append(max(vals, key=vals.get))
        rec["winners"] = winners
        rec["winner_stable"] = len(set(winners)) == 1
        recs.append(rec)

    recs.sort(key=lambda r: r["fst"] or 0)
    print(f"{'pair':<10}{'Fst':>9}  {'method':<11}{'mean':>8}{'sd':>8}{'min':>8}{'max':>8}{'spread':>9}")
    print("-" * 72)
    for r in recs:
        for m in METHODS:
            d = r["methods"].get(m)
            if not d:
                continue
            sd = f"{d['sd']:.4f}" if d["sd"] is not None else "   --"
            first = m == METHODS[0]
            name = r["pair"] if first else ""
            fs = f"{r['fst']:.5f}" if first and r["fst"] else ""
            print(f"{name:<10}{fs:>9}  {m:<11}{d['mean']:>8.4f}{sd:>8}"
                  f"{d['min']:>8.4f}{d['max']:>8.4f}{d['spread']:>9.4f}")
        print(f"{'':<10}{'':>9}  winner per partition: "
              f"{', '.join(r['winners'])}  -> "
              f"{'stable' if r['winner_stable'] else 'CHANGES'}")
        print()

    summary = {
        "pairs": recs,
        "n_partitions": len(SEEDS),
        "spread_within_method_mean": float(np.mean(all_spreads)) if all_spreads else None,
        "spread_within_method_max": float(np.max(all_spreads)) if all_spreads else None,
        "gap_between_methods_mean": float(np.mean(all_gaps)) if all_gaps else None,
        "n_pairs_winner_changes": sum(1 for r in recs if not r["winner_stable"]),
        "n_pairs": len(recs),
    }
    print(f"mean range a single method covers across partitions : "
          f"{summary['spread_within_method_mean']:.4f}")
    print(f"largest such range                                  : "
          f"{summary['spread_within_method_max']:.4f}")
    print(f"mean gap between best and worst method, one draw    : "
          f"{summary['gap_between_methods_mean']:.4f}")
    print(f"pairs where the leading method changes with the draw: "
          f"{summary['n_pairs_winner_changes']} of {summary['n_pairs']}")
    OUT.write_text(json.dumps(summary, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

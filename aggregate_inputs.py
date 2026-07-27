"""Aggregate the haplotype-feature experiment across pairs.

Produces the per-pair paired comparison of the frequency-only network against
the haplotype-aware one, alongside the released tools on the same pairs, and
the summary quantities the manuscript needs.

Everything is paired on seed: the two feature sets share partitions, windows
and initialisation, so the difference isolates the input representation. The
released-tool numbers come from results/realext_*.json and are single runs,
which is why they carry no interval here.
"""

import json
import glob
import os
import re
from pathlib import Path

import numpy as np
from scipy import stats

RES = Path("results")
OUT = RES / "inputs_summary.json"


def load_pair(path):
    rows = json.load(open(path))
    tag = "_".join(rows[0]["pops"])
    by = {}
    for m in ("freq", "haplo"):
        sel = sorted([r for r in rows if r["features"] == m], key=lambda r: r["seed"])
        if sel:
            by[m] = sel
    if "freq" not in by or "haplo" not in by:
        return None
    base = {r["seed"]: r for r in by["freq"]}
    hap = [r for r in by["haplo"] if r["seed"] in base]
    if not hap:
        return None

    def paired(key):
        a = np.array([r[key] for r in hap])
        b = np.array([base[r["seed"]][key] for r in hap])
        d = a - b
        t, p = stats.ttest_rel(a, b) if len(d) > 1 else (float("nan"),) * 2
        return {"freq": float(b.mean()), "haplo": float(a.mean()),
                "delta": float(d.mean()), "p": float(p), "n": len(d),
                "improved": int((d > 0).sum())}

    rec = {"pops": rows[0]["pops"], "tag": tag, "fst": rows[0]["fst"],
           "acc": paired("acc"), "tracts": paired("n_tract_ratio")}
    ext = RES / f"realext_{tag}.json"
    if ext.exists():
        e = json.load(open(ext))
        rec["rfmix"], rec["flare"] = e.get("rfmix"), e.get("flare")
    return rec


def main():
    recs = [r for r in (load_pair(f)
                        for f in sorted(glob.glob("results/tuning/inputs_*.json")))
            if r]
    recs.sort(key=lambda r: r["fst"])

    hdr = (f"{'pair':<9}{'Fst':>8}{'freq':>8}{'haplo':>8}{'delta':>9}{'p':>8}{'imp':>5}"
           f"{'RFMix':>8}{'FLARE':>8}{'beats':>16}")
    print(hdr)
    print("-" * len(hdr))
    for r in recs:
        a = r["acc"]
        rf, fl = r.get("rfmix"), r.get("flare")
        beat = []
        if rf is not None and a["haplo"] > rf:
            beat.append("RFMix")
        if fl is not None and a["haplo"] > fl:
            beat.append("FLARE")
        print(f"{'/'.join(r['pops']):<9}{r['fst']:>8.4f}{a['freq']:>8.4f}{a['haplo']:>8.4f}"
              f"{a['delta']:>+9.4f}{a['p']:>8.4f}{a['improved']}/{a['n']:<3}"
              f"{rf if rf else float('nan'):>8.4f}{fl if fl else float('nan'):>8.4f}"
              f"{('+'.join(beat) if beat else '--'):>16}")

    d = np.array([r["acc"]["delta"] for r in recs])
    fst = np.array([r["fst"] for r in recs])
    tr = np.array([r["tracts"]["delta"] for r in recs])
    print()
    print(f"pairs: {len(recs)}   mean accuracy gain {d.mean():+.4f} "
          f"(positive on {(d > 0).sum()}/{len(d)})")
    print(f"mean tract-ratio change {tr.mean():+.1f} "
          f"(improved on {(tr < 0).sum()}/{len(tr)})")
    if len(recs) > 2:
        rho, p = stats.spearmanr(fst, d)
        print(f"gain vs Fst: Spearman rho={rho:+.3f}, p={p:.3f} "
              f"(negative would mean the gain concentrates at low divergence)")
        t, pt = stats.ttest_1samp(d, 0.0)
        print(f"gain across pairs: mean {d.mean():+.4f}, t={t:.2f}, p={pt:.5f}")
    beats_flare = sum(1 for r in recs
                      if r.get("flare") is not None and r["acc"]["haplo"] > r["flare"])
    beats_rfmix = sum(1 for r in recs
                      if r.get("rfmix") is not None and r["acc"]["haplo"] > r["rfmix"])
    was_beating = sum(1 for r in recs if r.get("flare") is not None
                      and r["acc"]["freq"] > r["flare"])
    print(f"haplotype-aware network exceeds FLARE on {beats_flare}/{len(recs)} pairs "
          f"(frequency-only: {was_beating}/{len(recs)})")
    print(f"haplotype-aware network exceeds RFMix on {beats_rfmix}/{len(recs)} pairs")

    OUT.write_text(json.dumps(recs, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()

"""Assemble the Recomb-Mix comparison across the real-haplotype pairs.

Recomb-Mix is a single run per pair, as RFMix and FLARE are, so it carries no
interval here. The network columns are the five-seed means from the
input-parity experiment, which is what Table 3's last two columns already
report, so the rows below are directly comparable with the published table.
"""

import argparse
import json
import glob
from pathlib import Path

import numpy as np

RES = Path("results")
OUT_TMPL = "recombmix_summary{}.json"


def network_means(tag, suffix=""):
    """Five-seed means for the frequency-only and haplotype-aware networks."""
    path = RES / "tuning" / f"inputs_{tag}{suffix}.json"
    if not path.exists():
        return None, None
    rows = json.load(open(path))
    out = {}
    for m in ("freq", "haplo"):
        sel = [r["acc"] for r in rows if r["features"] == m]
        out[m] = float(np.mean(sel)) if sel else None
    return out.get("freq"), out.get("haplo")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="",
                    help='"" for the 80-haplotype panel, "_max" for 92')
    args = ap.parse_args()
    suffix = args.tag
    recs = []
    for path in sorted(glob.glob(str(RES / f"recombmix_*{suffix}.json"))):
        if "summary" in path:
            continue
        if not suffix and ("_max" in path or "_fd" in path):
            continue
        d = json.load(open(path))
        tag = "_".join(d["pops"])
        ext = RES / f"realext_{tag}{suffix}.json"
        e = json.load(open(ext)) if ext.exists() else {}
        freq, haplo = network_means(tag, suffix)
        recs.append({
            "pops": d["pops"], "tag": tag, "fst": d["fst"],
            "recombmix": d.get("recombmix"),
            "rfmix": e.get("rfmix"), "flare": e.get("flare"),
            "cnn_freq": freq, "cnn_haplo": haplo,
        })
    recs.sort(key=lambda r: r["fst"])

    f = lambda v: f"{v:.4f}" if isinstance(v, float) else "  --  "
    hdr = (f"{'pair':<10}{'Fst':>9}{'RFMix':>9}{'FLARE':>9}{'Recomb':>9}"
           f"{'CNN':>9}{'CNN+hap':>9}   best")
    print(hdr)
    print("-" * len(hdr))
    n_rm_best = n_rm_beats_released = 0
    for r in recs:
        cand = {k: r[k] for k in ("rfmix", "flare", "recombmix", "cnn_freq", "cnn_haplo")
                if isinstance(r[k], float)}
        best = max(cand, key=cand.get) if cand else "?"
        released = [r[k] for k in ("rfmix", "flare") if isinstance(r[k], float)]
        if isinstance(r["recombmix"], float) and released:
            if r["recombmix"] > max(released):
                n_rm_beats_released += 1
        if best == "recombmix":
            n_rm_best += 1
        print(f"{'/'.join(r['pops']):<10}{r['fst']:>9.5f}{f(r['rfmix']):>9}"
              f"{f(r['flare']):>9}{f(r['recombmix']):>9}{f(r['cnn_freq']):>9}"
              f"{f(r['cnn_haplo']):>9}   {best}")

    n = len(recs)
    summary = {
        "n_pairs": n, "panel": suffix or "_80",
        "recombmix_best_overall": n_rm_best,
        "recombmix_beats_rfmix_and_flare": n_rm_beats_released,
        "pairs": recs,
    }
    # How the network now stands against the strongest released tool, which is
    # the comparison the manuscript makes and which Recomb-Mix may change.
    for key, label in (("cnn_freq", "freq"), ("cnn_haplo", "haplo")):
        wins = sum(
            1 for r in recs
            if isinstance(r[key], float)
            and r[key] > max(v for k, v in r.items()
                             if k in ("rfmix", "flare", "recombmix")
                             and isinstance(v, float))
        )
        summary[f"cnn_{label}_beats_all_released"] = wins
        print(f"network ({label}) beats every released tool on {wins} of {n} pairs")
    print(f"Recomb-Mix beats both RFMix and FLARE on "
          f"{n_rm_beats_released} of {n} pairs; best overall on {n_rm_best}")

    out = RES / OUT_TMPL.format(suffix)
    out.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

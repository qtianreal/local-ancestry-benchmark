"""Is match length more discriminative than match rate?

The haplotype channels the network receives summarise, in a local window, the
*fraction* of sites at which the target agrees with each reference haplotype.
The copying models that RFMix and FLARE are built on use something different:
the length of contiguous agreement. Two references agreeing at 95% of sites at
random and one matching perfectly across a long run are nearly identical in
agreement rate and completely different as evidence of shared ancestry.

This measures both statistics as single-feature classifiers on the same data,
with no learning and no smoothing: at each site, call the ancestry whichever
panel scores higher, and score against the exact labels.

    rate    mean of the top-k agreement fractions in a +/- window
            (what the network currently receives)
    length  longest contiguous run of agreement covering the site, maximised
            over the panel (the copying-model statistic)

If length is not the more discriminative of the two on these panels, adding it
cannot explain the gap to FLARE and the idea is dead before it is built.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from lai.real import load_region, split_panel
from lai.sim import hudson_fst
from run_real import build_segment

OUT = Path("results")
WINDOW = 64
TOPK = 5


def match_rate(target, ref, window=WINDOW, topk=TOPK):
    """Mean of the top-k agreement fractions in a centred window, per site."""
    n = target.size
    agree = (target[:, None] == ref).astype(np.int32)
    csum = np.zeros((n + 1, ref.shape[1]), np.int32)
    np.cumsum(agree, axis=0, out=csum[1:])
    idx = np.arange(n)
    hi = np.minimum(idx + window + 1, n)
    lo = np.maximum(idx - window, 0)
    frac = (csum[hi] - csum[lo]).astype(np.float32) / (hi - lo)[:, None]
    part = np.partition(frac, -topk, axis=1)
    return part[:, -topk:].mean(axis=1)


def match_length(target, ref):
    """Longest contiguous agreement run covering each site, over the panel.

    For every reference haplotype the run length covering site i is the number
    of consecutive agreeing sites ending at or after i, computed as a forward
    and a backward pass; the panel maximum is then taken per site. This is the
    set-maximal match length a PBWT would produce, without the index.
    """
    agree = (target[:, None] == ref)                     # (S, R)
    n, r = agree.shape
    fwd = np.zeros((n, r), np.int32)
    bwd = np.zeros((n, r), np.int32)
    run = np.zeros(r, np.int32)
    for i in range(n):
        run = np.where(agree[i], run + 1, 0)
        fwd[i] = run
    run[:] = 0
    for i in range(n - 1, -1, -1):
        run = np.where(agree[i], run + 1, 0)
        bwd[i] = run
    # length of the run through site i, counting i once
    through = np.where(agree, fwd + bwd - 1, 0)
    return through.max(axis=1).astype(np.float32)


def score(admixed, labels, ra, rb, fn, n_targets, rng):
    correct = total = 0
    for j in rng.choice(admixed.shape[1], size=min(n_targets, admixed.shape[1]),
                        replace=False):
        a, b = fn(admixed[:, j], ra), fn(admixed[:, j], rb)
        call = (b > a).astype(np.int8)
        truth = labels[:, j]
        correct += int((call == truth).sum()); total += truth.size
    return correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs",
                    default="CEU,TSI;FIN,GBR;CHB,CDX;CHB,JPT;TSI,PJL;CEU,GIH")
    ap.add_argument("--vcf", default="data/chr22.vcf.gz")
    ap.add_argument("--sites", type=int, default=4096)
    ap.add_argument("--targets", type=int, default=24)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    rows = []
    for spec in args.pairs.split(";"):
        pops = spec.split(",")
        haps, positions = load_region(args.vcf, "chr22", 16_000_000, 51_000_000, pops)
        fst = hudson_fst(haps[pops[0]], haps[pops[1]])
        n = positions.size
        te = slice(int(n * 0.65), n)
        rate, leng = [], []
        for i in range(args.reps):
            rng = np.random.default_rng(4242 + i)
            ra, da = split_panel(haps[pops[0]], 80, 80, rng)
            rb, db = split_panel(haps[pops[1]], 80, 80, rng)
            seg, _ = build_segment(ra[te], rb[te], da[te], db[te],
                                   positions[te], 64, rng)
            sl = slice(0, min(args.sites, seg["admixed"].shape[0]))
            adm, lab = seg["admixed"][sl], seg["labels"][sl]
            A, B = seg["ref_a"][sl], seg["ref_b"][sl]
            g = np.random.default_rng(i)
            rate.append(score(adm, lab, A, B, match_rate, args.targets, g))
            g = np.random.default_rng(i)
            leng.append(score(adm, lab, A, B, match_length, args.targets, g))
        rec = {"pops": pops, "fst": fst,
               "rate": float(np.mean(rate)), "rate_sd": float(np.std(rate)),
               "length": float(np.mean(leng)), "length_sd": float(np.std(leng))}
        rec["gain"] = rec["length"] - rec["rate"]
        rows.append(rec)
        print(f"{'/'.join(pops):<9} Fst={fst:.4f}   rate={rec['rate']:.4f}   "
              f"length={rec['length']:.4f}   {rec['gain']:+.4f}", flush=True)
        (OUT / "matchlen.json").write_text(json.dumps(rows, indent=2))

    g = np.array([r["gain"] for r in rows])
    print(f"\nlength minus rate: mean {g.mean():+.4f}, "
          f"better on {int((g > 0).sum())}/{len(g)} pairs")


if __name__ == "__main__":
    main()

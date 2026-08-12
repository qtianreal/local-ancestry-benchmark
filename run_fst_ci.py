"""How precisely do we know the Fst values the whole study is organised on?

The paper places population pairs on an Fst axis and reports a floor at the
low end of it, where the smallest value is CHB/CHS at about 0.0004. A reader
who works on Fst estimation will ask whether a number that small is
distinguishable from zero at all, and nothing in the paper answered that.

Hudson's estimator here is a ratio of sums over sites, so its uncertainty is
not the standard error of a mean and cannot be read off the per-site spread.
The standard treatment is a delete-one-block jackknife over contiguous
genomic blocks, which is what Bhatia et al. use for the same estimator. We
use 1 Mb blocks: at the rate assumed throughout this study that is 1 cM, well
beyond the scale on which linkage disequilibrium persists, so blocks are
close to independent. Block sizes differ, so the variance uses the weighted
form of Busing et al. rather than the equal-block one.

No new data and no inference: this reads the same VCF region, the same sites
and the same estimator as every other Fst in the manuscript.

    python run_fst_ci.py

Writes results/fst_ci.json.
"""

import json
from pathlib import Path

import numpy as np

from lai.real import load_region

VCF = "data/chr22.vcf.gz"
REGION = ("chr22", 16_000_000, 51_000_000)
BLOCK_BP = 1_000_000

# the eleven benchmark pairs of Table 3
BENCH = [("CHB", "CHS"), ("CEU", "TSI"), ("FIN", "GBR"), ("CHB", "CDX"),
         ("CHB", "JPT"), ("FIN", "TSI"), ("TSI", "PJL"), ("CEU", "GIH"),
         ("CHB", "BEB"), ("CHB", "GIH"), ("CHB", "CEU")]

# The pairs the Introduction and Conclusions name as lying below the floor.
# LANDMARK_POPS is the population set compute_landmarks.py loads, not just the
# populations these three pairs need: the MAF filter is pooled over whatever is
# loaded, so using a smaller set here would put the interval on a different site
# set from the point estimate it accompanies.
LANDMARK = [("CHB", "CHS"), ("IBS", "TSI"), ("GBR", "CEU")]
LANDMARK_POPS = ["CDX", "CEU", "CHB", "CHS", "GBR", "IBS", "JPT", "KHV", "TSI"]


def fst_terms(hap_a, hap_b):
    """Per-site numerator and denominator of Hudson's Fst, as in lai/sim.py."""
    n1, n2 = hap_a.shape[1], hap_b.shape[1]
    p1, p2 = hap_a.mean(axis=1), hap_b.mean(axis=1)
    num = (p1 - p2) ** 2 - p1 * (1 - p1) / (n1 - 1) - p2 * (1 - p2) / (n2 - 1)
    den = p1 * (1 - p2) + p2 * (1 - p1)
    keep = den > 0
    return num[keep], den[keep], keep


def jackknife(num, den, block):
    """Weighted delete-one-block jackknife for a ratio of sums.

    Busing et al. (1999); the weighting matters because blocks hold unequal
    numbers of sites and an unweighted jackknife would misstate the variance.
    """
    tot_n, tot_d = num.sum(), den.sum()
    theta = tot_n / tot_d
    ids = np.unique(block)
    n = len(num)
    m = np.array([(block == b).sum() for b in ids], dtype=float)
    part = np.array([(tot_n - num[block == b].sum()) / (tot_d - den[block == b].sum())
                     for b in ids])
    g = len(ids)
    h = n / m
    theta_j = g * theta - ((1 - m / n) * part).sum()
    var = np.mean((h * theta - (h - 1) * part - theta_j) ** 2 / (h - 1))
    return float(theta), float(np.sqrt(var)), g


def interval(haps, pos, a, b):
    num, den, keep = fst_terms(haps[a], haps[b])
    block = (pos[keep] // BLOCK_BP).astype(np.int64)
    f, se, g = jackknife(num, den, block)
    return {"fst": f, "se": se, "lo": f - 1.96 * se, "hi": f + 1.96 * se,
            "z": f / se, "n_blocks": g, "n_sites": int(keep.sum())}


def main():
    out = {}
    print(f"{'pair':<10}{'Fst':>10}{'SE':>10}{'95% CI':>22}{'Fst/SE':>9}")

    # The site set depends on which populations are loaded together, because
    # the MAF filter is pooled over them. Each value below is therefore
    # computed the same way the number it accompanies was: benchmark pairs one
    # pair at a time, as run_real_external.py does, and the Introduction's
    # landmarks over the pooled load compute_landmarks.py uses.
    print("  -- benchmark pairs, loaded one pair at a time --")
    for a, b in BENCH:
        haps, pos = load_region(VCF, *REGION, [a, b])
        r = interval(haps, pos, a, b)
        out[f"{a}/{b}"] = dict(r, load="pair")
        print(f"{a}/{b:<6}{r['fst']:>10.5f}{r['se']:>10.5f}"
              f"   [{r['lo']:>8.5f}, {r['hi']:>8.5f}]{r['z']:>9.1f}")

    print("  -- Introduction landmarks, pooled load --")
    haps, pos = load_region(VCF, *REGION, LANDMARK_POPS)
    for a, b in LANDMARK:
        r = interval(haps, pos, a, b)
        out[f"landmark:{a}/{b}"] = dict(r, load="pooled")
        print(f"{a}/{b:<6}{r['fst']:>10.5f}{r['se']:>10.5f}"
              f"   [{r['lo']:>8.5f}, {r['hi']:>8.5f}]{r['z']:>9.1f}")

    out["_meta"] = {"vcf": VCF, "region": list(REGION), "block_bp": BLOCK_BP,
                    "method": "weighted delete-one-block jackknife (Busing et al.)",
                    "estimator": "Hudson ratio of averages, as in lai/sim.py",
                    "note": "site set depends on which populations are loaded "
                            "together, so each pair is loaded as its own "
                            "quoted number was"}
    Path("results").mkdir(exist_ok=True)
    Path("results/fst_ci.json").write_text(json.dumps(out, indent=2))
    print("\nwrote results/fst_ci.json")


if __name__ == "__main__":
    main()

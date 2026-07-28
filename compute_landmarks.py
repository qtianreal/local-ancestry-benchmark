"""Divergence landmarks quoted in the Introduction.

These values motivate the whole study -- they are the reason the low-divergence
regime is worth benchmarking -- and they were hand-entered, which is how they
came to disagree with the paper's own Table 2. Computed here with the same
estimator, on the same sites, as every other Fst in the manuscript, and written
to results/landmark_fst.json for make_numbers.py to pick up.

    python compute_landmarks.py
"""

import json
from pathlib import Path

from lai.real import load_region
from lai.sim import hudson_fst

VCF = "data/chr22.vcf.gz"
REGION = ("chr22", 16_000_000, 51_000_000)

# tag -> (population pair, what it illustrates)
LANDMARKS = {
    "HanNS":   (("CHB", "CHS"), "north/south Han cline"),
    "IbsTsi":  (("IBS", "TSI"), "within-Europe, Iberian/Tuscan"),
    "GbrCeu":  (("GBR", "CEU"), "within-Europe, British/Utah"),
    "ChbJpt":  (("CHB", "JPT"), "Han/Japanese"),
    "ChbKhv":  (("CHB", "KHV"), "Han/Kinh"),
    "ChbCdx":  (("CHB", "CDX"), "Han/Dai"),
    "EurEas":  (("CHB", "CEU"), "European/East Asian benchmark"),
}


def main():
    pops = sorted({p for pair, _ in LANDMARKS.values() for p in pair})
    haps, pos = load_region(VCF, *REGION, pops)
    out = {}
    for tag, ((a, b), what) in LANDMARKS.items():
        f = float(hudson_fst(haps[a], haps[b]))
        out[tag] = {"pops": [a, b], "fst": f, "describes": what}
        print(f"  {tag:<8} {a}/{b:<4} {f:.5f}   {what}")
    out["_meta"] = {"vcf": VCF, "region": list(REGION), "n_sites": int(pos.size),
                    "estimator": "Hudson, as in lai/sim.py"}
    Path("results/landmark_fst.json").write_text(json.dumps(out, indent=2))
    print(f"\n{pos.size} sites -> results/landmark_fst.json")


if __name__ == "__main__":
    main()

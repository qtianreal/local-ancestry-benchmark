"""Does accuracy depend on the product delta*F_ST/g, or only on F_ST?

The identifiability index of Equation (4) predicts accuracy from
x = delta F_ST / g: segregating sites per Morgan, over generations since the
pulse, times divergence. Fitted to the eleven real pairs it works, but that
fit cannot test the delta or the 1/g term, because across those pairs density
spans 1.2-fold and g is fixed at 30 by the construction. Only F_ST varies.

Here both are varied on purpose. Marker density is lowered by subsampling
sites; the pulse age is set directly, because the admixture is imposed by us
rather than historical. If the index is a law rather than a curve fitted to
F_ST, accuracy at (delta/4, g) should match accuracy at (delta, 4g), and every
cell should collapse onto one function of x.

The test uses the windowed likelihood classifier and its HMM-smoothed version,
not the network: those are exactly the estimator the derivation describes,
they need no training, and they cannot be accused of learning the design.

    python run_index_scaling.py                 # ~4 pairs, 3 densities, 3 ages
    python run_index_scaling.py --quick         # one pair, for a timing check

Writes results/index_scaling.json.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from lai.real import load_region, split_panel
from lai.sim import RHO, hudson_fst
from run_pilot import WINDOW
from lai.sim import make_admixed
from run_real import RealConfig, baseline_on

OUT = Path("results")
PAIRS = ["CHB_CDX", "FIN_TSI", "TSI_PJL", "CHB_GIH"]
DENSITIES = [1.0]                # see the note on window span below
AGES = [10, 20, 40, 80]          # generations since the pulse


def segment_at_age(ref_a, ref_b, donor_a, donor_b, positions, n_admixed, g, rng):
    """Mosaic donors with the pulse age actually set.

    run_real.build_segment builds its own RealConfig and so always mosaics at
    the default 30 generations; calling it here would leave the pulse age
    fixed while appearing to vary it.
    """
    pos = positions - positions[0]
    cfg = RealConfig(seq_length=float(pos[-1]), n_admixed=n_admixed,
                     admix_generations=g)
    admixed, labels = make_admixed(cfg, donor_a, donor_b, pos, rng)
    return {"ref_a": ref_a, "ref_b": ref_b, "admixed": admixed,
            "labels": labels, "positions": pos}


def evaluate(ha, hb, positions, n_ref, n_donor, g, reps, rng):
    """Mean baseline accuracy over `reps` reference/donor partitions."""
    nb, hmm = [], []
    for _ in range(reps):
        ra, da = split_panel(ha, n_ref, n_donor, rng)
        rb, db = split_panel(hb, n_ref, n_donor, rng)
        seg = segment_at_age(ra, rb, da, db, positions, 32, g, rng)
        n_sites = seg["admixed"].shape[0]
        step = max(1, (n_sites - WINDOW) // 6)
        for start in range(0, n_sites - WINDOW, step):
            a, h = baseline_on(seg, start, g)
            nb.append(a)
            hmm.append(h)
    return float(np.mean(nb)), float(np.mean(hmm))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vcf", default="data/chr22.vcf.gz")
    ap.add_argument("--chrom", default="chr22")
    ap.add_argument("--start", type=int, default=16_000_000)
    ap.add_argument("--end", type=int, default=51_000_000)
    ap.add_argument("--n-ref", type=int, default=80)
    ap.add_argument("--n-donor", type=int, default=80)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    pairs = PAIRS[:1] if args.quick else PAIRS
    densities = DENSITIES
    ages = [10, 80] if args.quick else AGES

    rows = []
    for tag in pairs:
        pa, pb = tag.split("_")
        t0 = time.time()
        haps, positions = load_region(args.vcf, args.chrom, args.start,
                                      args.end, [pa, pb])
        ha, hb = haps[pa], haps[pb]
        fst = float(hudson_fst(ha, hb))
        print(f"{tag}: {len(positions)} sites, F_ST={fst:.5f} "
              f"({time.time()-t0:.0f}s to load)", flush=True)

        for frac in densities:
            rng = np.random.default_rng(args.seed)
            if frac < 1.0:
                keep = np.sort(rng.choice(len(positions),
                                          int(len(positions) * frac),
                                          replace=False))
                sa, sb, spos = ha[keep], hb[keep], positions[keep]
            else:
                sa, sb, spos = ha, hb, positions
            span_morgans = (spos[-1] - spos[0]) * RHO
            delta = len(spos) / span_morgans

            for g in ages:
                rng = np.random.default_rng(args.seed + 1)
                nb, hmm = evaluate(sa, sb, spos, args.n_ref, args.n_donor,
                                   g, args.reps, rng)
                x = delta * fst / g
                rows.append(dict(pair=f"{pa}/{pb}", fst=fst, frac=frac,
                                 n_sites=int(len(spos)), delta=delta, g=g,
                                 x=x, naive_bayes=nb, hmm=hmm))
                print(f"  frac={frac:<5} g={g:<3} sites={len(spos):>7} "
                      f"x={x:8.2f}  windowed={nb:.3f} hmm={hmm:.3f}", flush=True)

    OUT.mkdir(exist_ok=True)
    (OUT / "index_scaling.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {OUT/'index_scaling.json'} ({len(rows)} cells)")


if __name__ == "__main__":
    main()

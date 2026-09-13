"""Loter on the real-haplotype pairs.

Loter (Dias-Alves, Mairal and Blum, MBE 2018) is a haplotype-copying local
ancestry method built for closely related populations and a wide range of
species, so it is a natural fourth released tool for the low-divergence regime
this study concerns.

Everything up to the tool call mirrors run_real_recombmix.py line for line --
same panels, same seed, same split, same evaluation segment, same mosaic
construction -- so the admixed haplotypes Loter sees are the ones RFMix, FLARE
and Recomb-Mix already saw. Loter is run at its published defaults (no tuning),
matching how the other released tools are run.
"""

import os
import tempfile
import argparse
import json
from pathlib import Path

import numpy as np

import loter.locanc.local_ancestry as lc

from lai.export import _dedup_positions
from lai.real import load_region, split_panel
from lai.sim import hudson_fst, make_admixed
from run_external import score
from run_real import RealConfig

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"


def loter_calls(ref_a, ref_b, adm, threads):
    """Per-site ancestry for the admixed haplotypes from Loter.

    Loter wants haplotype-major matrices (haplotypes x sites) of 0/1 alleles;
    our arrays are site-major, so we transpose in and out. We call
    loter_local_ancestry, the haplotype-level bagged optimisation, rather than
    loter_smooth: loter_smooth adds a diploid phase-correction step that assumes
    each consecutive pair of haplotypes are the two homologues of one
    individual, which is false here (our admixed samples are independent mosaic
    haplotypes), and that step degrades accuracy on this construction. Defaults
    otherwise (lambda grid, 20 bagging rounds), so Loter is untuned like the
    other released tools. loter_local_ancestry returns (ancestry, vote_counts);
    we take the ancestry. Its label (0 for the first panel, 1 for the second)
    is resolved against the truth by score's flip check below.
    """
    l_H = [ref_a.T.astype(np.uint8), ref_b.T.astype(np.uint8)]
    h_adm = adm.T.astype(np.uint8)
    res, _votes = lc.loter_local_ancestry(l_H=l_H, h_adm=h_adm, num_threads=threads)
    return np.asarray(res, dtype=np.int8).T  # (n_sites, n_hap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pops", required=True)
    ap.add_argument("--vcf", default="data/chr22.vcf.gz")
    ap.add_argument("--chrom", default="chr22")
    ap.add_argument("--start", type=int, default=16_000_000)
    ap.add_argument("--end", type=int, default=51_000_000)
    ap.add_argument("--n-ref", type=int, default=80)
    ap.add_argument("--n-donor", type=int, default=80)
    ap.add_argument("--gen", type=int, default=30)
    ap.add_argument("--fixed-donors", action="store_true")
    ap.add_argument("--seed", type=int, default=4242,
                    help="reference/donor partition seed; 4242 is the published "
                         "draw, so the default reproduces it exactly")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    pops = args.pops.split(",")
    tag = "_".join(pops)
    haps, positions = load_region(args.vcf, args.chrom, args.start, args.end, pops)
    fst = hudson_fst(haps[pops[0]], haps[pops[1]])
    print(f"{tag}: {positions.size} sites, Fst={fst:.5f}", flush=True)

    # ---- identical to run_real_external.py / run_real_recombmix.py ----
    rng = np.random.default_rng(args.seed)
    ra, da = split_panel(haps[pops[0]], args.n_ref, args.n_donor, rng,
                         donors_from_end=args.fixed_donors)
    rb, db = split_panel(haps[pops[1]], args.n_ref, args.n_donor, rng,
                         donors_from_end=args.fixed_donors)

    n = positions.size
    te = slice(int(n * 0.65), n)
    pos_te = positions[te] - positions[te][0]
    cfg = RealConfig(seq_length=float(pos_te[-1]), n_admixed=64,
                     admix_generations=args.gen)
    admixed, labels = make_admixed(cfg, da[te], db[te], pos_te, rng)

    pos_int, keep = _dedup_positions(pos_te + 1000)
    adm, truth = admixed[keep], labels[keep]
    rat, rbt = ra[te][keep], rb[te][keep]
    n_sites, n_hap = adm.shape
    # ---- end of shared construction ----

    rec = {"pops": pops, "fst": fst, "n_sites": int(n_sites), "gen": args.gen,
           "n_ref": args.n_ref, "version": "loter 1.0.1"}

    try:
        c = loter_calls(rat, rbt, adm, args.threads)
        a, cov = score(c, truth)
        b, _ = score(1 - np.where(c < 0, 0, c), truth)
        rec["loter"] = max(a, b)
        rec["loter_flipped"] = bool(b > a)
        rec["loter_coverage"] = cov
    except Exception as e:  # noqa: BLE001 - record the failure like the other runners
        rec["loter"], rec["loter_err"] = None, str(e)[-300:]

    (OUT / f"loter_{tag}{args.tag}.json").write_text(json.dumps(rec, indent=2))
    v = rec["loter"]
    print(f"{tag}: Fst={fst:.5f}  Loter="
          f"{f'{v:.4f}' if isinstance(v, float) else 'FAILED'}"
          f"  (coverage {rec.get('loter_coverage', float('nan')):.3f})", flush=True)


if __name__ == "__main__":
    main()

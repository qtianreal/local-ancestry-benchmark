"""RFMix and FLARE on the real-haplotype pairs.

The simulated arm showed that a reimplemented likelihood-plus-HMM baseline is
substantially weaker than the released tools, so a real-data comparison that
omitted them would repeat the same error. Everything here mirrors
run_real.py -- same panels, same mosaic construction, same evaluation
segment -- with the admixed haplotypes additionally exported to VCF.
"""

import os
import tempfile
import argparse
import json
from pathlib import Path

import numpy as np

from lai.export import (
    _dedup_positions,
    run,
    write_genetic_map,
    write_sample_map,
    write_vcf,
)
from lai.real import load_region, split_panel
from lai.sim import hudson_fst, make_admixed
from run_external import JAVA, FLARE, RFMIX, parse_flare, parse_rfmix, score
from run_real import RealConfig

# Scratch directory for intermediate files (VCF exports, external-tool
# output). Override with the LAI_WORKDIR environment variable.
WORKROOT = Path(os.environ.get("LAI_WORKDIR", tempfile.gettempdir())) / "lai-bench"

OUT = Path("results")
WORK = WORKROOT / "realext"


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
    ap.add_argument("--tag", default="",
                    help="suffix for the results file; without it a run at a "
                         "different panel size overwrites the published one")
    args = ap.parse_args()

    pops = args.pops.split(",")
    tag = "_".join(pops)
    haps, positions = load_region(args.vcf, args.chrom, args.start, args.end, pops)
    fst = hudson_fst(haps[pops[0]], haps[pops[1]])
    print(f"{tag}: {positions.size} sites, Fst={fst:.5f}", flush=True)

    rng = np.random.default_rng(4242)
    ra, da = split_panel(haps[pops[0]], args.n_ref, args.n_donor, rng,
                         donors_from_end=args.fixed_donors)
    rb, db = split_panel(haps[pops[1]], args.n_ref, args.n_donor, rng,
                         donors_from_end=args.fixed_donors)

    # Same held-out segment run_real.py evaluates on.
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

    d = WORK / tag
    d.mkdir(parents=True, exist_ok=True)
    names = write_vcf(d / "ref.vcf.gz", np.hstack([rat, rbt]), pos_int, args.chrom, "REF")
    write_vcf(d / "query.vcf.gz", adm, pos_int, args.chrom, "ADM")
    write_genetic_map(d / "map.rfmix.tsv", pos_int, args.chrom)
    write_genetic_map(d / "map.plink.tsv", pos_int, args.chrom, plink=True)
    half = rat.shape[1] // 2
    write_sample_map(d / "samples.tsv", names[:half], names[half:])

    rec = {"pops": pops, "fst": fst, "n_sites": int(n_sites), "gen": args.gen}

    ok, tail = run([str(RFMIX), "-f", str(d / "query.vcf.gz"), "-r", str(d / "ref.vcf.gz"),
                    "-m", str(d / "samples.tsv"), "-g", str(d / "map.rfmix.tsv"),
                    "-o", str(d / "rfmix"), f"--chromosome={args.chrom}",
                    "-G", str(args.gen)], d / "rfmix.log")
    if ok and (d / "rfmix.msp.tsv").exists():
        c = parse_rfmix(d / "rfmix.msp.tsv", n_sites, pos_int, n_hap)
        a, cov = score(c, truth)
        b, _ = score(1 - np.where(c < 0, 0, c), truth)
        rec["rfmix"], rec["rfmix_flipped"] = max(a, b), bool(b > a)
    else:
        rec["rfmix"], rec["rfmix_err"] = None, tail[-300:]

    ok, tail = run([JAVA, "-Xmx8g", "-jar", str(FLARE),
                    f"ref={d / 'ref.vcf.gz'}", f"ref-panel={d / 'samples.tsv'}",
                    f"gt={d / 'query.vcf.gz'}", f"map={d / 'map.plink.tsv'}",
                    f"out={d / 'flare'}", "min-mac=1", "min-maf=0",
                    f"gen={args.gen}", "nthreads=4"], d / "flare.log")
    if ok and (d / "flare.anc.vcf.gz").exists():
        c = parse_flare(d / "flare.anc.vcf.gz", n_sites, pos_int, n_hap)
        a, cov = score(c, truth)
        b, _ = score(1 - np.where(c < 0, 0, c), truth)
        rec["flare"], rec["flare_flipped"] = max(a, b), bool(b > a)
    else:
        rec["flare"], rec["flare_err"] = None, tail[-300:]

    (OUT / f"realext_{tag}{args.tag}.json").write_text(json.dumps(rec, indent=2))
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else "FAILED"
    print(f"{tag}: Fst={fst:.5f}  RFMix={fmt(rec['rfmix'])}  FLARE={fmt(rec['flare'])}",
          flush=True)


if __name__ == "__main__":
    main()

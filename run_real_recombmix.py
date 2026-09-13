"""Recomb-Mix on the real-haplotype pairs.

Recomb-Mix (Wei, Zhi and Zhang, Bioinformatics 2025) reports gains over RFMix
and FLARE in exactly the intracontinental regime this study is about, so a
benchmark that omitted it would leave the central comparison open.

Everything here mirrors run_real_external.py line for line -- same panels,
same seed, same split, same evaluation segment, same mosaic construction -- so
the admixed haplotypes Recomb-Mix sees are the ones RFMix and FLARE already
saw. ``--verify-rfmix`` re-runs RFMix and checks it reproduces the published
accuracy, which is what establishes that claim rather than asserting it.
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
    write_hapmap_map,
    write_pop_labels,
    write_sample_map,
    write_vcf,
)
from lai.real import load_region, split_panel
from lai.sim import hudson_fst, make_admixed
from run_external import RFMIX, parse_rfmix, score
from run_real import RealConfig

WORKROOT = Path(os.environ.get("LAI_WORKDIR", tempfile.gettempdir())) / "lai-bench"

ROOT = Path(__file__).resolve().parent
RECOMBMIX = ROOT / "tools" / "recombmix" / "RecombMix"
OUT = ROOT / "results"
WORK = WORKROOT / "realrm"


def parse_recombmix(path, n_sites, positions, n_hap):
    """Expand Recomb-Mix segment calls to per-site, per-haplotype ancestry.

    The output carries three sections. #Ancestry maps population label to a
    zero-based id, #Position lists the sites, and #Result gives one line per
    query haplotype: the haplotype name followed by (start, end, ancestry id)
    triples. Haplotype names are ``ADM{individual}_{0,1}``, which is the order
    write_vcf interleaves them in, so the name yields the column directly.
    """
    calls = np.full((n_sites, n_hap), -1, dtype=np.int8)
    section = None
    for line in Path(path).read_text().splitlines():
        if line.startswith("#"):
            section = line[1:].strip().lower()
            continue
        if section != "result" or not line.strip():
            continue
        f = line.rstrip("\n").split("\t")
        name = f[0]
        ind, hap = name.rsplit("_", 1)
        col = 2 * int(ind.replace("ADM", "")) + int(hap)
        if col >= n_hap:
            continue
        vals = f[1:]
        for k in range(0, len(vals) - 2, 3):
            spos, epos, anc = int(vals[k]), int(vals[k + 1]), int(vals[k + 2])
            lo, hi = np.searchsorted(positions, [spos, epos + 1])
            if hi > lo:
                calls[lo:hi, col] = anc
    return calls


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
    ap.add_argument("--weight", type=float, default=None,
                    help="cross-population penalty (-e); tool default is 1.5")
    ap.add_argument("--verify-rfmix", action="store_true",
                    help="re-run RFMix and compare with results/realext_*.json, "
                         "which checks the construction is unchanged")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    pops = args.pops.split(",")
    tag = "_".join(pops)
    haps, positions = load_region(args.vcf, args.chrom, args.start, args.end, pops)
    fst = hudson_fst(haps[pops[0]], haps[pops[1]])
    print(f"{tag}: {positions.size} sites, Fst={fst:.5f}", flush=True)

    # ---- identical to run_real_external.py from here ----
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

    d = WORK / tag
    d.mkdir(parents=True, exist_ok=True)
    ref = np.hstack([rat, rbt])
    names = write_vcf(d / "ref.vcf", ref, pos_int, args.chrom, "REF", bgzip=False)
    write_vcf(d / "query.vcf", adm, pos_int, args.chrom, "ADM", bgzip=False)
    write_hapmap_map(d / "map.hapmap.txt", pos_int, args.chrom)
    half = rat.shape[1] // 2
    write_pop_labels(d / "labels.txt", names[:half], names[half:])

    rec = {"pops": pops, "fst": fst, "n_sites": int(n_sites), "gen": args.gen,
           "n_ref": args.n_ref, "version": "RecombMix V0.8"}

    cmd = [str(RECOMBMIX), "-p", str(d / "ref.vcf"), "-q", str(d / "query.vcf"),
           "-a", str(d / "labels.txt"), "-g", str(d / "map.hapmap.txt"),
           "-o", str(d), "-i", "recombmix.txt", "-t", str(args.threads)]
    if args.weight is not None:
        cmd += ["-e", str(args.weight)]
        rec["weight"] = args.weight

    ok, tail = run(cmd, d / "recombmix.log")
    if ok and (d / "recombmix.txt").exists():
        c = parse_recombmix(d / "recombmix.txt", n_sites, pos_int, n_hap)
        a, cov = score(c, truth)
        b, _ = score(1 - np.where(c < 0, 0, c), truth)
        rec["recombmix"] = max(a, b)
        rec["recombmix_flipped"] = bool(b > a)
        rec["recombmix_coverage"] = cov
    else:
        rec["recombmix"], rec["recombmix_err"] = None, tail[-300:]

    if args.verify_rfmix:
        write_vcf(d / "ref.vcf.gz", ref, pos_int, args.chrom, "REF")
        write_vcf(d / "query.vcf.gz", adm, pos_int, args.chrom, "ADM")
        write_genetic_map(d / "map.rfmix.tsv", pos_int, args.chrom)
        write_sample_map(d / "samples.tsv", names[:half], names[half:])
        ok, tail = run([str(RFMIX), "-f", str(d / "query.vcf.gz"),
                        "-r", str(d / "ref.vcf.gz"), "-m", str(d / "samples.tsv"),
                        "-g", str(d / "map.rfmix.tsv"), "-o", str(d / "rfmix"),
                        f"--chromosome={args.chrom}", "-G", str(args.gen)],
                       d / "rfmix.log")
        if ok and (d / "rfmix.msp.tsv").exists():
            c = parse_rfmix(d / "rfmix.msp.tsv", n_sites, pos_int, n_hap)
            a, _ = score(c, truth)
            b, _ = score(1 - np.where(c < 0, 0, c), truth)
            rec["rfmix_recheck"] = max(a, b)
            pub = OUT / f"realext_{tag}{args.tag}.json"
            if pub.exists():
                want = json.loads(pub.read_text())["rfmix"]
                rec["rfmix_published"] = want
                rec["rfmix_matches"] = abs(max(a, b) - want) < 1e-9
                print(f"  RFMix recheck {max(a, b):.6f} vs published {want:.6f}"
                      f"  -> {'MATCH' if rec['rfmix_matches'] else 'DIFFERS'}",
                      flush=True)

    (OUT / f"recombmix_{tag}{args.tag}.json").write_text(json.dumps(rec, indent=2))
    v = rec["recombmix"]
    print(f"{tag}: Fst={fst:.5f}  Recomb-Mix="
          f"{f'{v:.4f}' if isinstance(v, float) else 'FAILED'}"
          f"  (coverage {rec.get('recombmix_coverage', float('nan')):.3f})", flush=True)


if __name__ == "__main__":
    main()

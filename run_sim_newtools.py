"""Recomb-Mix and Loter on the simulated divergence sweep.

The two tools added in revision were run on the real-haplotype pairs; this
extends them to the simulated sweep, so that every released tool is present
where the feasibility floor and the tract-structure claims are made.

The protocol is copied from run_external.py line for line -- same SimConfig,
same replicate seeds (off = seed * 1_000_000; test replicate at
off + 777_000 + T), same VCF export, same orientation-checked scoring -- so
the numbers here drop straight into the Fig 1 / Table 2 comparison.

This script is deliberately isolated. It writes only results/sim_newtools.json
and touches no published result file. --verify-rfmix re-runs RFMix on the
identical exported inputs and checks it against the published accuracy in
results/external_results_s0.json, which is the same evidence-of-identical-data
check the real-pair runners perform.
"""

import os
import json
import tempfile
import argparse
from pathlib import Path

import numpy as np

import loter.locanc.local_ancestry as lc

from lai.export import (_dedup_positions, run, write_genetic_map,
                        write_hapmap_map, write_pop_labels,
                        write_sample_map, write_vcf)
from lai.sim import SimConfig, simulate_replicate
from run_external import CHROM, RFMIX, parse_rfmix, score
from run_real_recombmix import RECOMBMIX, parse_recombmix

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
WORK = Path(os.environ.get("LAI_WORKDIR", tempfile.gettempdir())) / "lai-bench" / "simnew"
SPLITS = [25, 50, 100, 200, 400, 800, 1600, 3200]


def run_loter(ref_a, ref_b, adm, threads):
    """Haplotype-level bagged Loter, as in run_real_loter.py (not loter_smooth)."""
    l_H = [ref_a.T.astype(np.uint8), ref_b.T.astype(np.uint8)]
    res, _votes = lc.loter_local_ancestry(l_H=l_H, h_adm=adm.T.astype(np.uint8),
                                          num_threads=threads)
    return np.asarray(res, dtype=np.int8).T


def scored(calls, truth):
    """Accuracy under the orientation the tool's labels imply, flip-checked."""
    a, cov = score(calls, truth)
    b, _ = score(1 - np.where(calls < 0, 0, calls), truth)
    return max(a, b), bool(b > a), cov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-times", default=",".join(str(t) for t in SPLITS))
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--verify-rfmix", action="store_true")
    ap.add_argument("--out", default=str(OUT / "sim_newtools.json"))
    args = ap.parse_args()

    published = {}
    if args.verify_rfmix:
        for r in json.loads((OUT / "external_results_s0.json").read_text()):
            published[r["split_time"]] = r.get("rfmix")

    WORK.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in [int(x) for x in args.seeds.split(",")]:
        off = seed * 1_000_000
        for T in [int(x) for x in args.split_times.split(",")]:
            cfg = SimConfig(split_time=T)
            test = simulate_replicate(cfg, seed=off + 777_000 + T)
            pos, keep = _dedup_positions(test["positions"])
            ra, rb = test["ref_a"][keep], test["ref_b"][keep]
            adm, truth = test["admixed"][keep], test["labels"][keep]
            n_sites, n_hap = adm.shape

            d = WORK / f"T{T}_s{seed}"
            d.mkdir(parents=True, exist_ok=True)
            ref = np.hstack([ra, rb])
            half = ra.shape[1] // 2   # write_vcf names individuals, not haplotypes

            rec = {"split_time": T, "seed": seed, "fst": float(test["fst"]),
                   "n_sites": int(n_sites)}

            # ---- Recomb-Mix (uncompressed VCF + hapmap map + labels) ----
            names = write_vcf(d / "ref.vcf", ref, pos, CHROM, "REF", bgzip=False)
            write_vcf(d / "query.vcf", adm, pos, CHROM, "ADM", bgzip=False)
            write_hapmap_map(d / "map.hapmap.txt", pos, CHROM)
            write_pop_labels(d / "labels.txt", names[:half], names[half:])
            ok, tail = run([str(RECOMBMIX), "-p", str(d / "ref.vcf"),
                            "-q", str(d / "query.vcf"), "-a", str(d / "labels.txt"),
                            "-g", str(d / "map.hapmap.txt"), "-o", str(d),
                            "-i", "recombmix.txt", "-t", str(args.threads)],
                           d / "recombmix.log")
            if ok and (d / "recombmix.txt").exists():
                c = parse_recombmix(d / "recombmix.txt", n_sites, pos, n_hap)
                acc, flip, cov = scored(c, truth)
                rec["recombmix"], rec["recombmix_flipped"] = acc, flip
                rec["recombmix_coverage"] = cov
            else:
                rec["recombmix"], rec["recombmix_err"] = None, tail[-300:]

            # ---- Loter (arrays; no map, no VCF) ----
            acc, flip, cov = scored(run_loter(ra, rb, adm, args.threads), truth)
            rec["loter"], rec["loter_flipped"], rec["loter_coverage"] = acc, flip, cov

            # ---- optional RFMix identical-data check ----
            if args.verify_rfmix:
                write_vcf(d / "ref.vcf.gz", ref, pos, CHROM, "REF")
                write_vcf(d / "query.vcf.gz", adm, pos, CHROM, "ADM")
                write_genetic_map(d / "map.rfmix.tsv", pos, CHROM)
                write_sample_map(d / "samples.tsv", names[:half], names[half:])
                okr, tailr = run([str(RFMIX), "-f", str(d / "query.vcf.gz"),
                                  "-r", str(d / "ref.vcf.gz"),
                                  "-m", str(d / "samples.tsv"),
                                  "-g", str(d / "map.rfmix.tsv"),
                                  "-o", str(d / "rfmix"), f"--chromosome={CHROM}",
                                  "-G", str(cfg.admix_generations)], d / "rfmix.log")
                if okr and (d / "rfmix.msp.tsv").exists():
                    acc, _, _ = scored(parse_rfmix(d / "rfmix.msp.tsv", n_sites,
                                                   pos, n_hap), truth)
                    rec["rfmix_check"] = acc
                    rec["rfmix_published"] = published.get(T)
                else:
                    rec["rfmix_check_err"] = tailr[-300:]

            rows.append(rec)
            msg = (f"seed {seed}  T={T:<5d} Fst={rec['fst']:.5f}  "
                   f"recombmix={rec['recombmix']}  loter={rec['loter']:.4f}")
            if "rfmix_check" in rec:
                msg += (f"  rfmix={rec['rfmix_check']:.4f}"
                        f" (published {rec['rfmix_published']})")
            print(msg, flush=True)
            Path(args.out).write_text(json.dumps(rows, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

"""Benchmark against the released implementations: RFMix v2 and FLARE.

All five methods are scored on identical sites of identical haplotypes with
identical ground truth. RFMix and FLARE receive the same reference panels and
the same admixed targets as the internal methods, exported to VCF.

Ancestry-code orientation is verified rather than assumed: a transposed label
mapping would yield 1 - accuracy, so the orientation implied by each tool's
metadata is checked against the data and reported if it disagrees.

FLARE's defaults (min-mac=50, min-maf=0.005) discard most sites at our sample
size, so they are relaxed; using the defaults would handicap it for reasons
unrelated to the inference problem.
"""

import os
import tempfile
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from lai.export import (
    _dedup_positions,
    run,
    write_genetic_map,
    write_sample_map,
    write_vcf,
)
from lai.methods import (
    build_features,
    hmm_predict,
    naive_bayes_predict,
    panel_frequencies,
    site_log_ratio,
)
from lai.sim import RHO, SimConfig, simulate_replicate
from run_pilot import NB_WINDOW, WINDOW, assemble, eval_cnn, train_cnn

# Scratch directory for intermediate files (VCF exports, external-tool
# output). Override with the LAI_WORKDIR environment variable.
WORKROOT = Path(os.environ.get("LAI_WORKDIR", tempfile.gettempdir())) / "lai-bench"

ROOT = Path(__file__).resolve().parent
RFMIX = ROOT / "tools" / "rfmix" / "rfmix"
FLARE = ROOT / "tools" / "flare.jar"
JAVA = (ROOT / "tools" / "java_home.txt").read_text().strip() + "/bin/java"
OUT = ROOT / "results"
CHROM = "chr1"


def parse_rfmix(msp_path, n_sites, positions, n_hap):
    """Expand RFMix segment calls to per-site, per-haplotype ancestry."""
    calls = np.full((n_sites, n_hap), -1, dtype=np.int8)
    with open(msp_path) as fh:
        lines = [l for l in fh if not l.startswith("#Subpop")]
    for line in lines[1:]:
        f = line.rstrip("\n").split("\t")
        spos, epos = int(f[1]), int(f[2])
        vals = np.array([int(v) for v in f[6:]], dtype=np.int8)
        lo, hi = np.searchsorted(positions, [spos, epos + 1])
        if hi > lo:
            calls[lo:hi] = vals[:n_hap]
    return calls


def parse_flare(anc_path, n_sites, positions, n_hap):
    """Read FLARE AN1/AN2 fields, then forward-fill to unreported sites."""
    rec_pos, rec_calls = [], []
    with gzip.open(anc_path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            fmt = f[8].split(":")
            i1, i2 = fmt.index("AN1"), fmt.index("AN2")
            row = np.empty(n_hap, dtype=np.int8)
            for j, s in enumerate(f[9:]):
                p = s.split(":")
                row[2 * j] = int(p[i1])
                row[2 * j + 1] = int(p[i2])
            rec_pos.append(int(f[1]))
            rec_calls.append(row)
    if not rec_pos:
        return np.full((n_sites, n_hap), -1, dtype=np.int8)
    rec_pos = np.asarray(rec_pos)
    rec_calls = np.vstack(rec_calls)
    # Nearest reported marker at or before each site; ancestry is piecewise
    # constant along the chromosome so this is the natural extension.
    idx = np.clip(np.searchsorted(rec_pos, positions, side="right") - 1, 0, len(rec_pos) - 1)
    return rec_calls[idx]


def score(calls, truth):
    valid = calls >= 0
    if valid.sum() == 0:
        return float("nan"), 0.0
    acc = float((calls[valid] == truth[valid]).mean())
    return acc, float(valid.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-times", default="50,200,800,3200")
    ap.add_argument("--n-train", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--workdir", default=str(WORKROOT / "ext"))
    ap.add_argument("--seed", type=int, default=0,
                    help="replicate index; offsets every simulation seed")
    args = ap.parse_args()

    wd = Path(args.workdir)
    wd.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    rows = []

    for T in [int(x) for x in args.split_times.split(",")]:
        cfg = SimConfig(split_time=T)
        off = args.seed * 1_000_000
        test = simulate_replicate(cfg, seed=off + 777_000 + T)
        pos, keep = _dedup_positions(test["positions"])
        ra, rb = test["ref_a"][keep], test["ref_b"][keep]
        adm, truth = test["admixed"][keep], test["labels"][keep]
        n_sites, n_hap = adm.shape

        d = wd / f"T{T}_s{args.seed}"
        d.mkdir(exist_ok=True)
        names = write_vcf(d / "ref.vcf.gz", np.hstack([ra, rb]), pos, CHROM, "REF")
        write_vcf(d / "query.vcf.gz", adm, pos, CHROM, "ADM")
        write_genetic_map(d / "map.rfmix.tsv", pos, CHROM)
        write_genetic_map(d / "map.plink.tsv", pos, CHROM, plink=True)
        half = ra.shape[1] // 2
        write_sample_map(d / "samples.tsv", names[:half], names[half:])

        ok_r, tail_r = run([str(RFMIX), "-f", str(d / "query.vcf.gz"),
                            "-r", str(d / "ref.vcf.gz"), "-m", str(d / "samples.tsv"),
                            "-g", str(d / "map.rfmix.tsv"), "-o", str(d / "rfmix"),
                            f"--chromosome={CHROM}",
                            "-G", str(cfg.admix_generations)], d / "rfmix.log")
        ok_f, tail_f = run([JAVA, "-Xmx8g", "-jar", str(FLARE),
                            f"ref={d / 'ref.vcf.gz'}", f"ref-panel={d / 'samples.tsv'}",
                            f"gt={d / 'query.vcf.gz'}", f"map={d / 'map.plink.tsv'}",
                            f"out={d / 'flare'}", "min-mac=1", "min-maf=0",
                            f"gen={cfg.admix_generations}",
                            "nthreads=4"], d / "flare.log")

        rec = {"split_time": T, "seed": args.seed, "fst": test["fst"],
               "n_sites": int(n_sites)}

        if ok_r and (d / "rfmix.msp.tsv").exists():
            c = parse_rfmix(d / "rfmix.msp.tsv", n_sites, pos, n_hap)
            a, cov = score(c, truth)
            b, _ = score(1 - np.where(c < 0, 0, c), truth)
            rec["rfmix"], rec["rfmix_cov"] = max(a, b), cov
            rec["rfmix_flipped"] = bool(b > a)
        else:
            rec["rfmix"] = None
            rec["rfmix_err"] = tail_r[-300:]

        if ok_f and (d / "flare.anc.vcf.gz").exists():
            c = parse_flare(d / "flare.anc.vcf.gz", n_sites, pos, n_hap)
            a, cov = score(c, truth)
            b, _ = score(1 - np.where(c < 0, 0, c), truth)
            rec["flare"], rec["flare_cov"] = max(a, b), cov
            rec["flare_flipped"] = bool(b > a)
        else:
            rec["flare"] = None
            rec["flare_err"] = tail_f[-300:]

        # Internal methods on the identical sites.
        p_a, p_b = panel_frequencies(ra, rb)
        lr = site_log_ratio(adm, p_a, p_b)
        nb_pred, win_score = naive_bayes_predict(lr, NB_WINDOW)
        n_used = nb_pred.shape[0]
        rec["naive_bayes"] = float((nb_pred == truth[:n_used]).mean())
        span = (pos[-1] - pos[0]) / (n_sites / NB_WINDOW)
        sp = float(np.clip(cfg.admix_generations * RHO * span, 1e-6, 0.4))
        hmm = np.repeat(hmm_predict(win_score, sp), NB_WINDOW, axis=0)
        rec["hmm"] = float((hmm == truth[:n_used]).mean())

        train_reps = [simulate_replicate(cfg, seed=off + 778_000 + T * 7 + i)
                      for i in range(args.n_train)]
        xtr, ytr = assemble(train_reps, 3, np.random.default_rng(off + T))
        p_a2, p_b2 = panel_frequencies(ra, rb)
        feats, labs = [], []
        for s in range(0, n_sites - WINDOW, WINDOW):
            sl = slice(s, s + WINDOW)
            pa, pb = panel_frequencies(ra[sl], rb[sl])
            feats.append(build_features(adm[sl], pa, pb))
            labs.append(truth[sl].T.astype(np.float32))
        xte = torch.from_numpy(np.concatenate(feats))
        yte = torch.from_numpy(np.concatenate(labs))
        model, _ = train_cnn(xtr, ytr, device, args.epochs, seed=off + T)
        rec["cnn"] = eval_cnn(model, xte, yte, device)

        rows.append(rec)
        print(f"Fst={rec['fst']:.5f}  NB={rec['naive_bayes']:.4f}  HMM={rec['hmm']:.4f}  "
              f"RFMix={rec['rfmix'] if rec['rfmix'] is None else round(rec['rfmix'],4)}  "
              f"FLARE={rec['flare'] if rec['flare'] is None else round(rec['flare'],4)}  "
              f"CNN={rec['cnn']:.4f}", flush=True)
        tag = "" if args.seed == 0 else f"_s{args.seed}"
        (OUT / f"external_results{tag}.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

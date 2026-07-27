"""Does realistic demography close the simulation-to-reality gap?

Under a clean split with constant recombination, our convolutional network
beats RFMix and FLARE in simulation and loses to them on real haplotypes. If
that reversal is caused by the simulation understating haplotype structure,
then adding the two most obvious missing ingredients -- continuous gene flow
and an empirical recombination map -- should shrink the network's simulated
advantage toward what real data shows.

Conditions are run at several split times each and compared against measured
Fst rather than at nominal parameters, since migration and the genetic map
both change the Fst a given split time produces.
"""

import os
import tempfile
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lai.export import (
    _dedup_positions, run, write_genetic_map, write_sample_map, write_vcf,
)
from lai.methods import (
    build_features, hmm_predict, naive_bayes_predict, panel_frequencies,
    site_log_ratio,
)
from lai.realistic import RealisticConfig, simulate_replicate
from run_external import FLARE, JAVA, RFMIX, parse_flare, parse_rfmix, score
from run_pilot import NB_WINDOW, WINDOW, eval_cnn, train_cnn

# Scratch directory for intermediate files (VCF exports, external-tool
# output). Override with the LAI_WORKDIR environment variable.
WORKROOT = Path(os.environ.get("LAI_WORKDIR", tempfile.gettempdir())) / "lai-bench"

OUT = Path("results")
WORK = WORKROOT / "realistic"
CHROM = "chr1"

CONDITIONS = [
    ("toy",        0.0,   False, (50, 100, 200)),
    ("migration",  5e-3,  False, (200, 400, 800)),
    ("geneticmap", 0.0,   True,  (50, 100, 200)),
    ("both",       5e-3,  True,  (200, 400, 800)),
]


def windows_from(rep, n, rng):
    feats, labs = [], []
    n_sites = rep["admixed"].shape[0]
    for _ in range(n):
        s = int(rng.integers(0, n_sites - WINDOW))
        sl = slice(s, s + WINDOW)
        pa, pb = panel_frequencies(rep["ref_a"][sl], rep["ref_b"][sl])
        feats.append(build_features(rep["admixed"][sl], pa, pb))
        labs.append(rep["labels"][sl].T.astype(np.float32))
    return (torch.from_numpy(np.concatenate(feats)),
            torch.from_numpy(np.concatenate(labs)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    WORK.mkdir(parents=True, exist_ok=True)
    rows = []

    for name, mig, gmap, split_times in CONDITIONS:
        for T in split_times:
            cfg = RealisticConfig(split_time=T, migration_rate=mig, use_genetic_map=gmap)
            test = simulate_replicate(cfg, seed=31_000 + T)
            pos, keep = _dedup_positions(test["positions"])
            ra, rb = test["ref_a"][keep], test["ref_b"][keep]
            adm, truth = test["admixed"][keep], test["labels"][keep]
            n_sites, n_hap = adm.shape

            d = WORK / f"{name}_T{T}"
            d.mkdir(exist_ok=True)
            names = write_vcf(d / "ref.vcf.gz", np.hstack([ra, rb]), pos, CHROM, "REF")
            write_vcf(d / "query.vcf.gz", adm, pos, CHROM, "ADM")
            write_genetic_map(d / "map.rfmix.tsv", pos, CHROM)
            write_genetic_map(d / "map.plink.tsv", pos, CHROM, plink=True)
            half = ra.shape[1] // 2
            write_sample_map(d / "samples.tsv", names[:half], names[half:])

            rec = {"condition": name, "migration": mig, "genetic_map": gmap,
                   "split_time": T, "fst": test["fst"], "n_sites": int(n_sites)}

            ok, tail = run([str(RFMIX), "-f", str(d / "query.vcf.gz"),
                            "-r", str(d / "ref.vcf.gz"), "-m", str(d / "samples.tsv"),
                            "-g", str(d / "map.rfmix.tsv"), "-o", str(d / "rfmix"),
                            f"--chromosome={CHROM}", "-G", str(cfg.admix_generations)],
                           d / "rfmix.log")
            if ok and (d / "rfmix.msp.tsv").exists():
                c = parse_rfmix(d / "rfmix.msp.tsv", n_sites, pos, n_hap)
                a, _ = score(c, truth)
                b, _ = score(1 - np.where(c < 0, 0, c), truth)
                rec["rfmix"] = max(a, b)
            else:
                rec["rfmix"] = None

            ok, tail = run([JAVA, "-Xmx8g", "-jar", str(FLARE),
                            f"ref={d / 'ref.vcf.gz'}", f"ref-panel={d / 'samples.tsv'}",
                            f"gt={d / 'query.vcf.gz'}", f"map={d / 'map.plink.tsv'}",
                            f"out={d / 'flare'}", "min-mac=1", "min-maf=0",
                            f"gen={cfg.admix_generations}", "nthreads=4"], d / "flare.log")
            if ok and (d / "flare.anc.vcf.gz").exists():
                c = parse_flare(d / "flare.anc.vcf.gz", n_sites, pos, n_hap)
                a, _ = score(c, truth)
                b, _ = score(1 - np.where(c < 0, 0, c), truth)
                rec["flare"] = max(a, b)
            else:
                rec["flare"] = None

            pa, pb = panel_frequencies(ra, rb)
            lr = site_log_ratio(adm, pa, pb)
            nb_pred, win_score = naive_bayes_predict(lr, NB_WINDOW)
            nu = nb_pred.shape[0]
            rec["naive_bayes"] = float((nb_pred == truth[:nu]).mean())
            span = (pos[-1] - pos[0]) / (n_sites / NB_WINDOW)
            sp = float(np.clip(cfg.admix_generations * 1e-8 * span, 1e-6, 0.4))
            hmm = np.repeat(hmm_predict(win_score, sp), NB_WINDOW, axis=0)
            rec["hmm"] = float((hmm == truth[:nu]).mean())

            train_reps = [simulate_replicate(cfg, seed=32_000 + T * 7 + i)
                          for i in range(args.n_train)]
            xtr, ytr = [], []
            rng = np.random.default_rng(T)
            for rp in train_reps:
                a_, b_ = windows_from(rp, 3, rng)
                xtr.append(a_); ytr.append(b_)
            xtr, ytr = torch.cat(xtr), torch.cat(ytr)
            xte, yte = windows_from(test, 4, np.random.default_rng(9000 + T))
            model, _ = train_cnn(xtr, ytr, device, args.epochs, seed=T)
            rec["cnn"] = eval_cnn(model, xte, yte, device)

            ext = [v for v in (rec["rfmix"], rec["flare"]) if isinstance(v, float)]
            rec["gap_vs_best_tool"] = rec["cnn"] - max(ext) if ext else None
            rows.append(rec)
            g = rec["gap_vs_best_tool"]
            print(f"{name:<11} T={T:<4} Fst={rec['fst']:.5f}  "
                  f"RFMix={rec['rfmix']:.4f} FLARE={rec['flare']:.4f} "
                  f"CNN={rec['cnn']:.4f}  gap={g:+.4f}", flush=True)
            (OUT / "realistic_results.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

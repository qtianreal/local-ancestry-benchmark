"""How much does phasing error cost each method?

Switch errors are applied to reference panels and admixed targets alike, since
in applied use everything is statistically phased. Training data receives the
same error rate as test data, which is the setting most favourable to the
learned method: it can adapt to the noise rather than meeting it unseen.

The prediction under test is an asymmetry. Swapping alleles between the two
haplotypes of one individual leaves allele frequencies exactly unchanged, so a
method reading only frequencies should be immune, while RFMix and FLARE, which
match against reference haplotypes, should degrade.
"""

import os
import tempfile
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lai.export import _dedup_positions, run, write_genetic_map, write_sample_map, write_vcf
from lai.methods import (
    DilatedCNN,
    build_features, hmm_predict, naive_bayes_predict, panel_frequencies, site_log_ratio,
)
from lai.phasing import apply_switch_errors
from lai.sim import RHO, SimConfig, simulate_replicate
from run_external import FLARE, JAVA, RFMIX, parse_flare, parse_rfmix, score
from run_pilot import NB_WINDOW, WINDOW, eval_cnn, train_cnn

# Scratch directory for intermediate files (VCF exports, external-tool
# output). Override with the LAI_WORKDIR environment variable.
WORKROOT = Path(os.environ.get("LAI_WORKDIR", tempfile.gettempdir())) / "lai-bench"

OUT = Path("results")
WORK = WORKROOT / "phasing"
CHROM = "chr1"


def corrupt(rep, rate, seed):
    """Apply switch errors to both reference panels and the admixed targets."""
    pos = rep["positions"]
    rng = np.random.default_rng(seed)
    ra, _ = apply_switch_errors(rep["ref_a"], pos, rate, rng)
    rb, _ = apply_switch_errors(rep["ref_b"], pos, rate, rng)
    adm, lab = apply_switch_errors(rep["admixed"], pos, rate, rng, rep["labels"])
    return {"ref_a": ra, "ref_b": rb, "admixed": adm, "labels": lab,
            "positions": pos, "fst": rep["fst"]}


def fit_haplo(model, x, y, device, epochs, lr=1e-3, batch=32):
    """Train loop matching train_cnn, but for an arbitrary channel count."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lf = torch.nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(x.shape[0])
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            lf(model(x[idx].to(device)), y[idx].to(device)).backward()
            opt.step()
        sch.step()
    return model


def windows(reps, n, rng, with_haps=False):
    f, l = [], []
    for rp in reps:
        ns = rp["admixed"].shape[0]
        for _ in range(n):
            s = int(rng.integers(0, ns - WINDOW)); sl = slice(s, s + WINDOW)
            ra, rb = rp["ref_a"][sl], rp["ref_b"][sl]
            pa, pb = panel_frequencies(ra, rb)
            f.append(build_features(rp["admixed"][sl], pa, pb,
                                    ra if with_haps else None,
                                    rb if with_haps else None))
            l.append(rp["labels"][sl].T.astype(np.float32))
    return torch.from_numpy(np.concatenate(f)), torch.from_numpy(np.concatenate(l))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-times", default="100,400")
    ap.add_argument("--rates", default="0,0.5,2.0", help="switch errors per Mb")
    ap.add_argument("--n-train", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    WORK.mkdir(parents=True, exist_ok=True)
    rows = []

    for T in [int(x) for x in args.split_times.split(",")]:
        cfg = SimConfig(split_time=T)
        base_test = simulate_replicate(cfg, seed=777_000 + T)
        base_train = [simulate_replicate(cfg, seed=778_000 + T * 7 + i)
                      for i in range(args.n_train)]

        for rate in [float(x) for x in args.rates.split(",")]:
            test = corrupt(base_test, rate, 555)
            train = [corrupt(r, rate, 900 + i) for i, r in enumerate(base_train)]

            pos, keep = _dedup_positions(test["positions"])
            ra, rb = test["ref_a"][keep], test["ref_b"][keep]
            adm, truth = test["admixed"][keep], test["labels"][keep]
            n_sites, n_hap = adm.shape

            d = WORK / f"T{T}_r{rate}"
            d.mkdir(exist_ok=True)
            names = write_vcf(d / "ref.vcf.gz", np.hstack([ra, rb]), pos, CHROM, "REF")
            write_vcf(d / "query.vcf.gz", adm, pos, CHROM, "ADM")
            write_genetic_map(d / "map.rfmix.tsv", pos, CHROM)
            write_genetic_map(d / "map.plink.tsv", pos, CHROM, plink=True)
            half = ra.shape[1] // 2
            write_sample_map(d / "samples.tsv", names[:half], names[half:])

            rec = {"split_time": T, "switch_per_mb": rate, "fst": test["fst"]}

            ok, _ = run([str(RFMIX), "-f", str(d / "query.vcf.gz"), "-r", str(d / "ref.vcf.gz"),
                         "-m", str(d / "samples.tsv"), "-g", str(d / "map.rfmix.tsv"),
                         "-o", str(d / "rfmix"), f"--chromosome={CHROM}",
                         "-G", str(cfg.admix_generations)], d / "rfmix.log")
            if ok and (d / "rfmix.msp.tsv").exists():
                c = parse_rfmix(d / "rfmix.msp.tsv", n_sites, pos, n_hap)
                a, _c = score(c, truth); b, _c = score(1 - np.where(c < 0, 0, c), truth)
                rec["rfmix"] = max(a, b)

            ok, _ = run([JAVA, "-Xmx8g", "-jar", str(FLARE), f"ref={d/'ref.vcf.gz'}",
                         f"ref-panel={d/'samples.tsv'}", f"gt={d/'query.vcf.gz'}",
                         f"map={d/'map.plink.tsv'}", f"out={d/'flare'}", "min-mac=1",
                         "min-maf=0", f"gen={cfg.admix_generations}", "nthreads=4"],
                        d / "flare.log")
            if ok and (d / "flare.anc.vcf.gz").exists():
                c = parse_flare(d / "flare.anc.vcf.gz", n_sites, pos, n_hap)
                a, _c = score(c, truth); b, _c = score(1 - np.where(c < 0, 0, c), truth)
                rec["flare"] = max(a, b)

            pa, pb = panel_frequencies(ra, rb)
            lr = site_log_ratio(adm, pa, pb)
            nb, win = naive_bayes_predict(lr, NB_WINDOW)
            nu = nb.shape[0]
            rec["naive_bayes"] = float((nb == truth[:nu]).mean())
            span = (pos[-1] - pos[0]) / (n_sites / NB_WINDOW)
            sp = float(np.clip(cfg.admix_generations * RHO * span, 1e-6, 0.4))
            rec["hmm"] = float((np.repeat(hmm_predict(win, sp), NB_WINDOW, axis=0)
                                == truth[:nu]).mean())

            xtr, ytr = windows(train, 3, np.random.default_rng(T))
            xte, yte = windows([test], 4, np.random.default_rng(5000 + T))
            model, _ = train_cnn(xtr, ytr, device, args.epochs, seed=T)
            rec["cnn"] = eval_cnn(model, xte, yte, device)

            # The haplotype-aware configuration. Switch errors leave allele
            # frequencies exactly unchanged but break haplotype contiguity, so
            # this arm is expected to degrade where the frequency-only one does
            # not. Since it is the configuration we recommend, its phase
            # sensitivity has to be measured rather than assumed.
            xtr_h, ytr_h = windows(train, 3, np.random.default_rng(T), True)
            xte_h, yte_h = windows([test], 4, np.random.default_rng(5000 + T), True)
            torch.manual_seed(T)
            mh = DilatedCNN(in_ch=8).to(device)
            mh = fit_haplo(mh, xtr_h, ytr_h, device, args.epochs)
            rec["cnn_haplo"] = eval_cnn(mh, xte_h, yte_h, device)

            rows.append(rec)
            print(f"T={T:<4} switch={rate:<4}/Mb Fst={rec['fst']:.5f}  "
                  f"NB={rec['naive_bayes']:.4f} HMM={rec['hmm']:.4f} "
                  f"RFMix={rec.get('rfmix', float('nan')):.4f} "
                  f"FLARE={rec.get('flare', float('nan')):.4f} CNN={rec['cnn']:.4f} "
                  f"CNN+hap={rec['cnn_haplo']:.4f}",
                  flush=True)
            (OUT / "phasing_results.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

"""Does the realistic-demography harness reproduce the validated sweep?

`results/realistic_results.json` reports a CNN-minus-best-tool gap of -0.055
for its control (no migration, no empirical map) at Fst ~ 0.009, where the
three-seed external replication gives +0.048 +/- 0.006 under nominally the
same demography. Until that is resolved nothing measured with this harness is
interpretable, so no factorial should be run through it.

NOTES_factorial.md names two candidate causes, both configuration rather than
code: `lai/realistic.py` uses 80 reference haplotypes where `lai/sim.py` uses
100, and run_realistic.py trains for 10x12 replicate-epochs where
run_external.py uses 12x15. This runs the control arm alone with both
differences removed and three seeds, and prints the gap beside the target.

    reproduces  -> the harness is sound, the factorial is worth running
    does not    -> the difference is in lai/realistic.py itself, and the
                   factorial would produce well-replicated wrong numbers

Nothing here writes to results/; the control is a diagnostic, not a result.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from lai.export import (_dedup_positions, write_genetic_map, write_sample_map,
                        write_vcf)
from lai.methods import build_features, panel_frequencies
from lai.realistic import RealisticConfig, simulate_replicate
from run_external import FLARE, JAVA, RFMIX, parse_flare, parse_rfmix, score
from run_pilot import WINDOW, eval_cnn, train_cnn
from run_realistic import CHROM, run

WORK = Path(os.environ.get("LAI_WORKDIR", "/tmp")) / "lai-bench" / "control"

# The external sweep's replicated value at the matched divergence, from
# results/external_aggregate.json: this is what the control has to reproduce.
TARGET_FST = 0.0100
TARGET_GAP = 0.048
TARGET_SD = 0.006


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
    ap.add_argument("--split-time", type=int, default=100,
                    help="control arm split time nearest the target Fst")
    ap.add_argument("--n-ref", type=int, default=100, help="matched to lai/sim.py")
    ap.add_argument("--n-train", type=int, default=12, help="matched to run_external.py")
    ap.add_argument("--epochs", type=int, default=15, help="matched to run_external.py")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    WORK.mkdir(parents=True, exist_ok=True)
    T = args.split_time
    rows = []

    for seed in range(args.seeds):
        cfg = RealisticConfig(split_time=T, migration_rate=0.0,
                              use_genetic_map=False, n_ref=args.n_ref)
        test = simulate_replicate(cfg, seed=31_000 + T + 101 * seed)
        pos, keep = _dedup_positions(test["positions"])
        ra, rb = test["ref_a"][keep], test["ref_b"][keep]
        adm, truth = test["admixed"][keep], test["labels"][keep]

        d = WORK / f"s{seed}"
        d.mkdir(exist_ok=True)
        names = write_vcf(d / "ref.vcf.gz", np.hstack([ra, rb]), pos, CHROM, "REF")
        write_vcf(d / "query.vcf.gz", adm, pos, CHROM, "ADM")
        write_genetic_map(d / "map.rfmix.tsv", pos, CHROM)
        write_genetic_map(d / "map.plink.tsv", pos, CHROM, plink=True)
        half = ra.shape[1] // 2      # names are diploid samples, not haplotypes
        write_sample_map(d / "samples.tsv", names[:half], names[half:])

        rec = {"seed": seed, "fst": float(test["fst"]), "n_ref": args.n_ref}

        n_sites, n_hap = adm.shape

        # Invocations, parser arguments and the ancestry-label flip correction
        # are copied verbatim from run_external.py. The point of the check is
        # that the only difference from the validated sweep is the simulator,
        # so anything else that differs would confound it.
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
            print(f"  seed {seed}: RFMix failed -- {tail[-200:]}", flush=True)

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
            print(f"  seed {seed}: FLARE failed -- {tail[-200:]}", flush=True)

        rng = np.random.default_rng(4242 + seed)
        train_reps = [simulate_replicate(cfg, seed=32_000 + T * 7 + 101 * seed + i)
                      for i in range(args.n_train)]
        xs, ys = [], []
        for rp in train_reps:
            a, b = windows_from(rp, 3, rng)
            xs.append(a); ys.append(b)
        xtr, ytr = torch.cat(xs), torch.cat(ys)
        xte, yte = windows_from(test, 4, np.random.default_rng(9000 + T + seed))
        model, _ = train_cnn(xtr, ytr, device, args.epochs, seed=T + seed)
        rec["cnn"] = float(eval_cnn(model, xte, yte, device))

        tools = [rec[k] for k in ("rfmix", "flare") if k in rec]
        rec["gap"] = rec["cnn"] - max(tools) if tools else float("nan")
        rows.append(rec)
        print(f"  seed {seed}: Fst={rec['fst']:.4f}  cnn={rec['cnn']:.4f}  "
              f"rfmix={rec.get('rfmix', float('nan')):.4f}  "
              f"flare={rec.get('flare', float('nan')):.4f}  "
              f"gap={rec['gap']:+.4f}", flush=True)

    g = np.array([r["gap"] for r in rows], float)
    fst = float(np.mean([r["fst"] for r in rows]))
    print(f"\ncontrol arm, {len(g)} seeds, Fst={fst:.4f}")
    print(f"  gap        {g.mean():+.4f} +/- {g.std(ddof=1):.4f}")
    print(f"  target     {TARGET_GAP:+.4f} +/- {TARGET_SD:.4f}  (external sweep, "
          f"Fst={TARGET_FST:.4f})")
    z = abs(g.mean() - TARGET_GAP) / np.hypot(g.std(ddof=1) / np.sqrt(len(g)), TARGET_SD)
    print(f"  separation {z:.1f} sigma")
    print(f"\n  {'REPRODUCES -- factorial is worth running' if z < 3 else 'DOES NOT REPRODUCE -- do not run the factorial'}")
    (WORK / "control_check.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

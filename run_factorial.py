"""Demographic factorial, replicated across seeds.

The question the manuscript's Limitations names: which simplification in the
simulated demography is responsible for the simulation-to-reality reversal?
Four conditions cross continuous migration with an empirical recombination
map, at three split times each.

The earlier single-seed attempt (results/realistic_results.json) was excluded
because its control arm disagreed with the validated sweep by ~17 sigma.
run_control_check.py established that this was configuration, not code: with
n_ref=100 and the 12x15 training budget the control gives +0.059 +/- 0.026
against the sweep's +0.048 +/- 0.006, a separation of 0.7 sigma. Those
settings are the defaults here.

Results are appended to results/factorial_results.json after every run, and
--resume skips cells already present, so a four-hour job survives an
interruption.

Power, from the control's seed spread of 0.026: at three seeds a between-
condition difference is resolvable at about 0.043. A null result at this size
means "not resolved", not "no effect", and must be reported that way.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from lai.export import (_dedup_positions, write_genetic_map, write_sample_map,
                        write_vcf)
from lai.realistic import RealisticConfig, simulate_replicate
from run_control_check import windows_from
from run_external import FLARE, JAVA, RFMIX, parse_flare, parse_rfmix, score
from run_pilot import eval_cnn, train_cnn
from run_realistic import CHROM, run

OUT = Path("results/factorial_results.json")
WORK = Path(os.environ.get("LAI_WORKDIR", "/tmp")) / "lai-bench" / "factorial"

# Split times differ by condition: migration holds Fst down, so the migrating
# arms need longer splits to span a comparable divergence range.
CONDITIONS = [
    ("toy",        0.0,  False, (50, 100, 200)),
    ("migration",  5e-3, False, (200, 400, 800)),
    ("geneticmap", 0.0,  True,  (50, 100, 200)),
    ("both",       5e-3, True,  (200, 400, 800)),
]


def run_cell(name, mig, gmap, T, seed, args, device):
    cfg = RealisticConfig(split_time=T, migration_rate=mig,
                          use_genetic_map=gmap, n_ref=args.n_ref)
    test = simulate_replicate(cfg, seed=31_000 + T + 101 * seed)
    pos, keep = _dedup_positions(test["positions"])
    ra, rb = test["ref_a"][keep], test["ref_b"][keep]
    adm, truth = test["admixed"][keep], test["labels"][keep]
    n_sites, n_hap = adm.shape

    d = WORK / f"{name}_T{T}_s{seed}"
    d.mkdir(parents=True, exist_ok=True)
    names = write_vcf(d / "ref.vcf.gz", np.hstack([ra, rb]), pos, CHROM, "REF")
    write_vcf(d / "query.vcf.gz", adm, pos, CHROM, "ADM")
    write_genetic_map(d / "map.rfmix.tsv", pos, CHROM)
    write_genetic_map(d / "map.plink.tsv", pos, CHROM, plink=True)
    half = ra.shape[1] // 2
    write_sample_map(d / "samples.tsv", names[:half], names[half:])

    rec = {"condition": name, "migration": mig, "genetic_map": gmap,
           "split_time": T, "seed": seed, "fst": float(test["fst"]),
           "n_ref": args.n_ref}

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
        print(f"      RFMix failed: {tail[-160:]}", flush=True)

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
        print(f"      FLARE failed: {tail[-160:]}", flush=True)

    rng = np.random.default_rng(4242 + seed)
    train_reps = [simulate_replicate(cfg, seed=32_000 + T * 7 + 101 * seed + i)
                  for i in range(args.n_train)]
    xs, ys = [], []
    for rp in train_reps:
        a, b = windows_from(rp, 3, rng)
        xs.append(a); ys.append(b)
    model, _ = train_cnn(torch.cat(xs), torch.cat(ys), device, args.epochs,
                         seed=T + seed)
    xte, yte = windows_from(test, 4, np.random.default_rng(9000 + T + seed))
    rec["cnn"] = float(eval_cnn(model, xte, yte, device))

    tools = [rec[k] for k in ("rfmix", "flare") if rec.get(k) is not None]
    rec["gap"] = rec["cnn"] - max(tools) if tools else float("nan")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-ref", type=int, default=100)
    ap.add_argument("--n-train", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    WORK.mkdir(parents=True, exist_ok=True)
    rows = json.loads(OUT.read_text()) if (args.resume and OUT.exists()) else []
    done = {(r["condition"], r["split_time"], r["seed"]) for r in rows}
    todo = [(n, m, g, T, s) for n, m, g, Ts in CONDITIONS for T in Ts
            for s in range(args.seeds) if (n, T, s) not in done]
    print(f"{len(todo)} runs to do ({len(done)} already present)", flush=True)

    for i, (name, mig, gmap, T, seed) in enumerate(todo, 1):
        rec = run_cell(name, mig, gmap, T, seed, args, device)
        rows.append(rec)
        OUT.write_text(json.dumps(rows, indent=2))
        print(f"  [{i}/{len(todo)}] {name:<11} T={T:<4} s={seed}  "
              f"Fst={rec['fst']:.4f}  cnn={rec['cnn']:.4f}  "
              f"gap={rec['gap']:+.4f}", flush=True)

    print("\n--- gap by condition, mean over seeds and split times ---", flush=True)
    for name, _, _, _ in CONDITIONS:
        g = np.array([r["gap"] for r in rows if r["condition"] == name
                      and not np.isnan(r["gap"])], float)
        if g.size:
            print(f"  {name:<11} {g.mean():+.4f} +/- {g.std(ddof=1) if g.size > 1 else 0:.4f}"
                  f"  (n={g.size})", flush=True)
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()

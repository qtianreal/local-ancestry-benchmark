"""Recomb-Mix and Loter in the remaining simulated analyses.

run_sim_newtools.py put the two tools on the simulated divergence series (per-site
accuracy, Fig 2). This extends them to the three other simulated analyses that
report RFMix and FLARE, on the identical data those analyses use:

  tracts     Table 2 (tract-count and length ratios), Fig S2 (breakpoint
             precision/recall/F1) and the implied admixture dates, on the
             seed-0 replicate of every divergence level, as run_tracts.py,
             run_bp_prf.py and run_dating.py.
  phasing    Table 7: switch errors at 0, 0.5 and 2.0 per Mb on the
             F_ST=0.0100 and 0.0364 levels, corrupted exactly as run_phasing.py.
  factorial  Table S2: the 4 demographic conditions x 3 split times x 3 seeds
             of run_factorial.py.

Every tool gets the constant-rate map the published runs gave RFMix and FLARE.
Scoring functions are imported from the published scripts, not copied. Each
mode records F_ST so the data identity can be checked against the published
result file. This script writes only results/sim_newtools_{mode}.json.
"""

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from lai.export import (_dedup_positions, run, write_hapmap_map,
                        write_pop_labels, write_vcf)
from lai.sim import SimConfig, simulate_replicate
from run_dating import g_hat, tract_lengths_bp
from run_decode_robustness import TOLERANCES_BP, breakpoint_prf
from run_external import CHROM
from run_phasing import corrupt
from run_real_recombmix import RECOMBMIX, parse_recombmix
from run_sim_newtools import run_loter, scored
from run_tracts import SPLITS, tract_stats

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
WORK = Path(os.environ.get("LAI_WORKDIR", tempfile.gettempdir())) / "lai-bench" / "simnew_ext"


def run_recombmix(d, ra, rb, adm, pos, threads):
    d.mkdir(parents=True, exist_ok=True)
    half = ra.shape[1] // 2
    names = write_vcf(d / "ref.vcf", np.hstack([ra, rb]), pos, CHROM, "REF", bgzip=False)
    write_vcf(d / "query.vcf", adm, pos, CHROM, "ADM", bgzip=False)
    write_hapmap_map(d / "map.hapmap.txt", pos, CHROM)
    write_pop_labels(d / "labels.txt", names[:half], names[half:])
    ok, tail = run([str(RECOMBMIX), "-p", str(d / "ref.vcf"), "-q", str(d / "query.vcf"),
                    "-a", str(d / "labels.txt"), "-g", str(d / "map.hapmap.txt"),
                    "-o", str(d), "-i", "recombmix.txt", "-t", str(threads)],
                   d / "recombmix.log")
    if not (ok and (d / "recombmix.txt").exists()):
        raise RuntimeError(f"Recomb-Mix failed in {d}: {tail[-300:]}")
    return parse_recombmix(d / "recombmix.txt", adm.shape[0], pos, adm.shape[1])


def oriented(calls, truth):
    """Accuracy plus the calls in the orientation that accuracy implies."""
    acc, flip, cov = scored(calls, truth)
    if flip:
        calls = np.where(calls < 0, calls, 1 - calls).astype(np.int8)
    return acc, flip, cov, calls


def both_tools(d, ra, rb, adm, truth, pos, threads):
    out = {}
    c = run_recombmix(d, ra, rb, adm, pos, threads)
    out["recombmix"] = oriented(c, truth)
    c = run_loter(ra, rb, adm, threads)
    out["loter"] = oriented(c, truth)
    return out


def mode_tracts(args):
    rows = []
    for T in SPLITS:
        cfg = SimConfig(split_time=T)
        rep = simulate_replicate(cfg, seed=777_000 + T)   # run_external seed 0
        pos, keep = _dedup_positions(rep["positions"])
        ra, rb = rep["ref_a"][keep], rep["ref_b"][keep]
        adm, truth = rep["admixed"][keep], rep["labels"][keep]
        rec = {"split_time": T, "fst": float(rep["fst"]), "n_sites": int(adm.shape[0])}
        for tool, (acc, flip, cov, calls) in both_tools(
                WORK / f"tracts_T{T}", ra, rb, adm, truth, pos, args.threads).items():
            st = {"acc": acc, "flipped": flip, "coverage": cov}
            st.update(tract_stats(calls, truth))
            for tol in TOLERANCES_BP:
                pr, rc, f1 = breakpoint_prf(calls, truth, pos, tol)
                kb = tol // 1000
                st[f"prec_{kb}kb"], st[f"rec_{kb}kb"], st[f"f1_{kb}kb"] = pr, rc, f1
            p = calls.copy(); p[p < 0] = 0
            st["g_hat"] = g_hat(tract_lengths_bp(p, pos))
            rec[tool] = st
        rec["truth_g_hat"] = g_hat(tract_lengths_bp(truth, pos))
        rows.append(rec)
        print(f"T={T:<5d} Fst={rec['fst']:.5f}  " + "  ".join(
            f"{k}: acc={rec[k]['acc']:.4f} n={rec[k]['n_tract_ratio']:.2f} "
            f"len={rec[k]['length_ratio']:.2f} F1={rec[k]['f1_1000kb']:.2f} "
            f"g={rec[k]['g_hat']:.1f}" for k in ("recombmix", "loter")), flush=True)
        (OUT / "sim_newtools_tracts.json").write_text(json.dumps(rows, indent=2))


def mode_phasing(args):
    rows = []
    for T in (100, 400):
        cfg = SimConfig(split_time=T)
        base = simulate_replicate(cfg, seed=777_000 + T)
        for rate in (0.0, 0.5, 2.0):
            test = corrupt(base, rate, 555)
            pos, keep = _dedup_positions(test["positions"])
            ra, rb = test["ref_a"][keep], test["ref_b"][keep]
            adm, truth = test["admixed"][keep], test["labels"][keep]
            rec = {"split_time": T, "switch_per_mb": rate, "fst": float(test["fst"])}
            for tool, (acc, flip, cov, _c) in both_tools(
                    WORK / f"phasing_T{T}_r{rate}", ra, rb, adm, truth, pos,
                    args.threads).items():
                rec[tool], rec[f"{tool}_flipped"], rec[f"{tool}_coverage"] = acc, flip, cov
            rows.append(rec)
            print(f"T={T:<4} switch={rate:<4}/Mb Fst={rec['fst']:.5f}  "
                  f"recombmix={rec['recombmix']:.4f} loter={rec['loter']:.4f}", flush=True)
            (OUT / "sim_newtools_phasing.json").write_text(json.dumps(rows, indent=2))


def mode_factorial(args):
    from lai.realistic import RealisticConfig
    from lai.realistic import simulate_replicate as sim_realistic
    from run_factorial import CONDITIONS
    path = OUT / "sim_newtools_factorial.json"
    rows = json.loads(path.read_text()) if (args.resume and path.exists()) else []
    done = {(r["condition"], r["split_time"], r["seed"]) for r in rows}
    for name, mig, gmap, Ts in CONDITIONS:
        for T in Ts:
            for seed in range(3):
                if (name, T, seed) in done:
                    continue
                cfg = RealisticConfig(split_time=T, migration_rate=mig,
                                      use_genetic_map=gmap, n_ref=100)
                test = sim_realistic(cfg, seed=31_000 + T + 101 * seed)
                pos, keep = _dedup_positions(test["positions"])
                ra, rb = test["ref_a"][keep], test["ref_b"][keep]
                adm, truth = test["admixed"][keep], test["labels"][keep]
                rec = {"condition": name, "split_time": T, "seed": seed,
                       "fst": float(test["fst"])}
                for tool, (acc, flip, cov, _c) in both_tools(
                        WORK / f"fact_{name}_T{T}_s{seed}", ra, rb, adm, truth, pos,
                        args.threads).items():
                    rec[tool], rec[f"{tool}_flipped"], rec[f"{tool}_coverage"] = acc, flip, cov
                rows.append(rec)
                print(f"{name:<11} T={T:<4} s={seed} Fst={rec['fst']:.4f}  "
                      f"recombmix={rec['recombmix']:.4f} loter={rec['loter']:.4f}",
                      flush=True)
                path.write_text(json.dumps(rows, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["tracts", "phasing", "factorial"])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    {"tracts": mode_tracts, "phasing": mode_phasing, "factorial": mode_factorial}[args.mode](args)


if __name__ == "__main__":
    main()

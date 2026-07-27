"""Tract-level accuracy, complementing the per-site figures.

Per-site accuracy can hide a failure mode that matters downstream: a method
may assign most sites correctly while shattering the ancestry mosaic into many
short spurious tracts. Admixture dating in particular reads the tract-length
distribution directly, so a method that halves mean tract length biases the
inferred admixture time even at high per-site accuracy.

Three quantities per method:

  n_tract_ratio   predicted tracts / true tracts. > 1 means fragmentation.
  length_ratio    mean predicted tract length / mean true length.
  bp_error        for each true breakpoint, distance in segregating sites to
                  the nearest predicted breakpoint (median over breakpoints).

All methods are scored on the identical test replicates already used for the
per-site results, so the two views are directly comparable.
"""

import os
import tempfile
import json
from pathlib import Path

import numpy as np
import torch

from lai.export import _dedup_positions
from lai.methods import (
    DilatedCNN, build_features, hmm_predict, naive_bayes_predict,
    panel_frequencies, site_log_ratio,
)
from lai.sim import RHO, SimConfig, simulate_replicate
from run_external import parse_flare, parse_rfmix
from run_pilot import NB_WINDOW, WINDOW

# Scratch directory for intermediate files (VCF exports, external-tool
# output). Override with the LAI_WORKDIR environment variable.
WORKROOT = Path(os.environ.get("LAI_WORKDIR", tempfile.gettempdir())) / "lai-bench"

OUT = Path("results")
OUT_FILE = OUT / "tract_results.json"
EXT = WORKROOT / "ext"
SPLITS = [25, 50, 100, 200, 400, 800, 1600, 3200]


def tract_stats(pred, truth):
    """Compare tract structure of a prediction against ground truth.

    pred, truth : (n_sites, n_haps) int arrays of ancestry labels
    """
    n_ratio, l_ratio, bp_err = [], [], []
    for j in range(truth.shape[1]):
        p, t = pred[:, j], truth[:, j]
        if (p < 0).any():
            continue
        pb = np.flatnonzero(np.diff(p) != 0)
        tb = np.flatnonzero(np.diff(t) != 0)
        n_ratio.append((len(pb) + 1) / (len(tb) + 1))
        l_ratio.append((len(t) / (len(pb) + 1)) / (len(t) / (len(tb) + 1)))
        if len(tb):
            if len(pb):
                d = np.abs(tb[:, None] - pb[None, :]).min(axis=1)
            else:  # no predicted switch: error is distance to the nearest end
                d = np.minimum(tb, len(t) - tb)
            bp_err.append(np.median(d))
    f = lambda a: float(np.mean(a)) if a else float("nan")
    return {"n_tract_ratio": f(n_ratio), "length_ratio": f(l_ratio),
            "bp_error_snps": f(bp_err)}


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    rows = []

    for T in SPLITS:
        cfg = SimConfig(split_time=T)
        # Identical replicate to run_external.py seed 0.
        rep = simulate_replicate(cfg, seed=777_000 + T)
        pos, keep = _dedup_positions(rep["positions"])
        ra, rb = rep["ref_a"][keep], rep["ref_b"][keep]
        adm, truth = rep["admixed"][keep], rep["labels"][keep]
        n_sites, n_hap = adm.shape

        preds = {}
        pa, pb = panel_frequencies(ra, rb)
        lr = site_log_ratio(adm, pa, pb)
        nb, win = naive_bayes_predict(lr, NB_WINDOW)
        nu = nb.shape[0]
        preds["naive_bayes"] = (nb, truth[:nu])
        span = (pos[-1] - pos[0]) / (n_sites / NB_WINDOW)
        sp = float(np.clip(cfg.admix_generations * RHO * span, 1e-6, 0.4))
        preds["hmm"] = (np.repeat(hmm_predict(win, sp), NB_WINDOW, axis=0), truth[:nu])

        d = EXT / f"T{T}"
        if (d / "rfmix.msp.tsv").exists():
            preds["rfmix"] = (parse_rfmix(d / "rfmix.msp.tsv", n_sites, pos, n_hap), truth)
        if (d / "flare.anc.vcf.gz").exists():
            preds["flare"] = (parse_flare(d / "flare.anc.vcf.gz", n_sites, pos, n_hap), truth)

        ckpt = OUT / "cache" / f"cnn_T{T}_s0.pt"
        if ckpt.exists():
            model = DilatedCNN().to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device))
            model.eval()
            cnn = np.zeros_like(truth)
            with torch.no_grad():
                for s in range(0, n_sites - WINDOW + 1, WINDOW):
                    sl = slice(s, s + WINDOW)
                    fa, fb = panel_frequencies(ra[sl], rb[sl])
                    x = torch.from_numpy(build_features(adm[sl], fa, fb)).to(device)
                    out = []
                    for i in range(0, x.shape[0], 64):
                        out.append((torch.sigmoid(model(x[i:i+64])) > 0.5).cpu().numpy())
                    cnn[sl] = np.concatenate(out).T.astype(np.int8)
            last = (n_sites // WINDOW) * WINDOW
            preds["cnn"] = (cnn[:last], truth[:last])

        # Same tiling for the haplotype-aware configuration, so its tract
        # statistics are measured over the identical sequence length as every
        # other row. Tract-count ratio scales with the length it is computed
        # over, so a figure taken from a windowed evaluation is not comparable
        # here and must not be substituted for one.
        ckpt_h = OUT / "cache" / f"cnn_haplo_T{T}.pt"
        if ckpt_h.exists():
            mh = DilatedCNN(in_ch=8).to(device)
            mh.load_state_dict(torch.load(ckpt_h, map_location=device))
            mh.eval()
            ch = np.zeros_like(truth)
            with torch.no_grad():
                for s in range(0, n_sites - WINDOW + 1, WINDOW):
                    sl = slice(s, s + WINDOW)
                    fa, fb = panel_frequencies(ra[sl], rb[sl])
                    x = torch.from_numpy(build_features(adm[sl], fa, fb,
                                                        ra[sl], rb[sl])).to(device)
                    out = []
                    for i in range(0, x.shape[0], 64):
                        out.append((torch.sigmoid(mh(x[i:i+64])) > 0.5).cpu().numpy())
                    ch[sl] = np.concatenate(out).T.astype(np.int8)
            last = (n_sites // WINDOW) * WINDOW
            preds["cnn_haplo"] = (ch[:last], truth[:last])

        rec = {"split_time": T, "fst": rep["fst"]}
        for name, (p, t) in preds.items():
            st = tract_stats(np.asarray(p), np.asarray(t))
            rec[name] = st
        # Preserve any method already scored at this level that we cannot
        # recompute now: the external tools depend on scratch output that is
        # periodically cleaned, and silently dropping them turns published
        # macros into placeholders.
        prior = {r["split_time"]: r for r in (
            json.loads(OUT_FILE.read_text()) if OUT_FILE.exists() else [])}
        for k, v in prior.get(T, {}).items():
            if isinstance(v, dict) and k not in rec:
                rec[k] = v
                print(f"  (kept {k} from previous run)", flush=True)
        rows.append(rec)
        print(f"Fst={rep['fst']:.5f}  " + "  ".join(
            f"{k}:n={rec[k]['n_tract_ratio']:.2f}/len={rec[k]['length_ratio']:.2f}"
            for k in ("hmm", "rfmix", "flare", "cnn") if k in rec), flush=True)
        OUT_FILE.write_text(json.dumps(rows, indent=2))

    print(f"\n{'method':<13}{'n_tract_ratio':>15}{'length_ratio':>14}{'bp_err(SNPs)':>14}")
    for m in ("naive_bayes", "hmm", "rfmix", "flare", "cnn"):
        vals = [r[m] for r in rows if m in r and not np.isnan(r[m]["n_tract_ratio"])]
        if vals:
            print(f"{m:<13}{np.mean([v['n_tract_ratio'] for v in vals]):>15.2f}"
                  f"{np.mean([v['length_ratio'] for v in vals]):>14.2f}"
                  f"{np.nanmean([v['bp_error_snps'] for v in vals]):>14.0f}")


if __name__ == "__main__":
    main()

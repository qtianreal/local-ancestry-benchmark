"""What admixture time would each method's tracts imply?

Reporting that a method fragments tracts is less useful than saying what that
costs downstream. For a single admixture pulse g generations ago, ancestry
tract lengths are approximately exponential with mean 1/g Morgans, so a
method's tract lengths imply an estimate

    g_hat = 1 / mean_tract_length_in_Morgans

Since our simulations use a constant recombination rate, physical length maps
to genetic length directly (1 Morgan = 1/rho base pairs). The true value is
known exactly -- 30 generations -- so the bias each method would introduce
into an admixture-dating analysis can be quoted rather than gestured at.

This is deliberately the simplest estimator; real dating methods fit the full
length distribution and would differ in detail. The point is the magnitude and
direction of the bias, which no refinement will rescue.
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
EXT = WORKROOT / "ext"
SPLITS = [25, 50, 100, 200, 400, 800, 1600, 3200]
TRUE_G = 30


def tract_lengths_bp(labels, positions):
    """Lengths in base pairs of maximal constant-ancestry runs."""
    out = []
    for j in range(labels.shape[1]):
        b = np.flatnonzero(np.diff(labels[:, j]) != 0)
        edges = np.concatenate([[0], b + 1, [labels.shape[0]]])
        seg = positions[np.clip(edges, 0, len(positions) - 1)]
        out.extend(np.diff(seg))
    a = np.asarray(out, dtype=float)
    return a[a > 0]


def g_hat(lengths_bp):
    """Admixture time implied by mean tract length under a single-pulse model."""
    if lengths_bp.size == 0:
        return float("nan")
    morgans = lengths_bp.mean() * RHO
    return float("nan") if morgans <= 0 else 1.0 / morgans


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    rows = []

    for T in SPLITS:
        cfg = SimConfig(split_time=T)
        rep = simulate_replicate(cfg, seed=777_000 + T)
        pos, keep = _dedup_positions(rep["positions"])
        ra, rb = rep["ref_a"][keep], rep["ref_b"][keep]
        adm, truth = rep["admixed"][keep], rep["labels"][keep]
        n_sites, n_hap = adm.shape

        preds = {"truth": truth}
        pa, pb = panel_frequencies(ra, rb)
        lr = site_log_ratio(adm, pa, pb)
        nb, win = naive_bayes_predict(lr, NB_WINDOW)
        nu = nb.shape[0]
        preds["naive_bayes"] = nb
        span = (pos[-1] - pos[0]) / (n_sites / NB_WINDOW)
        sp = float(np.clip(cfg.admix_generations * RHO * span, 1e-6, 0.4))
        preds["hmm"] = np.repeat(hmm_predict(win, sp), NB_WINDOW, axis=0)

        d = EXT / f"T{T}"
        if (d / "rfmix.msp.tsv").exists():
            preds["rfmix"] = parse_rfmix(d / "rfmix.msp.tsv", n_sites, pos, n_hap)
        if (d / "flare.anc.vcf.gz").exists():
            preds["flare"] = parse_flare(d / "flare.anc.vcf.gz", n_sites, pos, n_hap)

        ck = OUT / "cache" / f"cnn_T{T}_s0.pt"
        if ck.exists():
            m = DilatedCNN().to(device)
            m.load_state_dict(torch.load(ck, map_location=device)); m.eval()
            cnn = np.zeros_like(truth)
            with torch.no_grad():
                for s in range(0, n_sites - WINDOW + 1, WINDOW):
                    sl = slice(s, s + WINDOW)
                    fa, fb = panel_frequencies(ra[sl], rb[sl])
                    x = torch.from_numpy(build_features(adm[sl], fa, fb)).to(device)
                    o = [(torch.sigmoid(m(x[i:i+64])) > 0.5).cpu().numpy()
                         for i in range(0, x.shape[0], 64)]
                    cnn[sl] = np.concatenate(o).T.astype(np.int8)
            preds["cnn"] = cnn

        rec = {"split_time": T, "fst": rep["fst"]}
        for name, p in preds.items():
            p = np.asarray(p)
            if (p < 0).any():
                p = p.copy(); p[p < 0] = 0
            n = min(p.shape[0], len(pos))
            rec[name] = g_hat(tract_lengths_bp(p[:n], pos[:n]))
        rows.append(rec)
        print(f"Fst={rep['fst']:.5f}  truth={rec['truth']:6.1f}  "
              + "  ".join(f"{k}={rec[k]:7.1f}" for k in
                          ("naive_bayes", "hmm", "rfmix", "flare", "cnn") if k in rec),
              flush=True)
        (OUT / "dating_results.json").write_text(json.dumps(rows, indent=2))

    print(f"\nImplied admixture time (true value {TRUE_G} generations), mean over levels:")
    for k in ("truth", "naive_bayes", "hmm", "rfmix", "flare", "cnn"):
        v = [r[k] for r in rows if k in r and np.isfinite(r[k])]
        if v:
            print(f"  {k:<12} {np.mean(v):8.1f} generations   "
                  f"({np.mean(v)/TRUE_G:5.1f}x true)")


if __name__ == "__main__":
    main()

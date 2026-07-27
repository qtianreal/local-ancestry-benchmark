"""Does a switching process fix the network's tract fragmentation?

The trained network is left untouched. Its per-site logits are decoded twice:
once by independent thresholding, as reported elsewhere in this study, and
once by Viterbi decoding under a recombination-derived transition prior. Any
difference is attributable to the decoding alone.

Three quantities are compared, since fragmentation and accuracy need not move
together: per-site accuracy, tract-count ratio, and the admixture time the
resulting tracts imply.
"""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from lai.decode import viterbi_decode
from lai.export import _dedup_positions
from lai.methods import DilatedCNN, build_features, panel_frequencies
from lai.sim import RHO, SimConfig, simulate_replicate
from run_dating import g_hat, tract_lengths_bp
from run_pilot import WINDOW
from run_tracts import tract_stats

OUT = Path("results")
SPLITS = [25, 50, 100, 200, 400, 800, 1600, 3200]
SEEDS = [0, 1, 2]


@torch.no_grad()
def site_logits(model, adm, ra, rb, device, with_haps=False):
    """Per-site logits over the whole replicate, tiled in training-size windows."""
    n_sites = adm.shape[0]
    out = np.zeros(adm.T.shape, dtype=np.float32)
    for s in range(0, n_sites - WINDOW + 1, WINDOW):
        sl = slice(s, s + WINDOW)
        pa, pb = panel_frequencies(ra[sl], rb[sl])
        x = torch.from_numpy(build_features(adm[sl], pa, pb,
                                            ra[sl] if with_haps else None,
                                            rb[sl] if with_haps else None)).to(device)
        chunks = [model(x[i:i + 64]).cpu().numpy() for i in range(0, x.shape[0], 64)]
        out[:, sl] = np.concatenate(chunks)
    last = (n_sites // WINDOW) * WINDOW
    return out[:, :last].T, last          # (n_sites, n_haps)


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    rows = []

    for T in SPLITS:
        cfg = SimConfig(split_time=T)
        for seed in SEEDS:
            ck = OUT / "cache" / f"cnn_T{T}_s{seed}.pt"
            if not ck.exists():
                continue
            rep = simulate_replicate(cfg, seed=777_000 + T)
            pos, keep = _dedup_positions(rep["positions"])
            ra, rb = rep["ref_a"][keep], rep["ref_b"][keep]
            adm, truth = rep["admixed"][keep], rep["labels"][keep]

            model = DilatedCNN().to(device)
            model.load_state_dict(torch.load(ck, map_location=device))
            model.eval()

            logits, last = site_logits(model, adm, ra, rb, device)
            p, tr = pos[:last], truth[:last]

            raw = (logits > 0).astype(np.int8)
            crf = viterbi_decode(logits, p, cfg.admix_generations, rho=RHO)

            rec = {"split_time": T, "seed": seed, "fst": rep["fst"],
                   "config": "freq"}
            for tag, pred in (("raw", raw), ("crf", crf)):
                st = tract_stats(pred, tr)
                rec[f"{tag}_acc"] = float((pred == tr).mean())
                rec[f"{tag}_ntract"] = st["n_tract_ratio"]
                rec[f"{tag}_len"] = st["length_ratio"]
                rec[f"{tag}_g"] = g_hat(tract_lengths_bp(pred, p))
            rec["true_g"] = g_hat(tract_lengths_bp(tr, p))
            rows.append(rec)
            print(f"Fst={rep['fst']:.5f} s{seed}  "
                  f"acc {rec['raw_acc']:.4f}->{rec['crf_acc']:.4f}   "
                  f"tracts {rec['raw_ntract']:7.1f}x->{rec['crf_ntract']:5.2f}x   "
                  f"g {rec['raw_g']:7.0f}->{rec['crf_g']:6.0f} (true {rec['true_g']:.0f})",
                  flush=True)
            (OUT / "crf_results.json").write_text(json.dumps(rows, indent=2))

        # The haplotype-aware configuration, decoded identically. Viterbi acts
        # on logits alone, so it should transfer, but the promoted method is
        # the decoded haplotype-aware one and its tract structure has to be
        # measured rather than assumed to inherit.
        ckh = OUT / "cache" / f"cnn_haplo_T{T}.pt"
        if ckh.exists():
            rep = simulate_replicate(cfg, seed=777_000 + T)
            pos, keep = _dedup_positions(rep["positions"])
            ra, rb = rep["ref_a"][keep], rep["ref_b"][keep]
            adm, truth = rep["admixed"][keep], rep["labels"][keep]

            mh = DilatedCNN(in_ch=8).to(device)
            mh.load_state_dict(torch.load(ckh, map_location=device))
            mh.eval()
            logits, last = site_logits(mh, adm, ra, rb, device, with_haps=True)
            p_, tr = pos[:last], truth[:last]
            raw = (logits > 0).astype(np.int8)
            crf = viterbi_decode(logits, p_, cfg.admix_generations, rho=RHO)

            rec = {"split_time": T, "seed": 0, "fst": rep["fst"],
                   "config": "haplo"}
            for tag, pred in (("raw", raw), ("crf", crf)):
                st = tract_stats(pred, tr)
                rec[f"{tag}_acc"] = float((pred == tr).mean())
                rec[f"{tag}_ntract"] = st["n_tract_ratio"]
                rec[f"{tag}_len"] = st["length_ratio"]
                rec[f"{tag}_g"] = g_hat(tract_lengths_bp(pred, p_))
            rec["true_g"] = g_hat(tract_lengths_bp(tr, p_))
            rows.append(rec)
            print(f"Fst={rep['fst']:.5f} haplo  "
                  f"acc {rec['raw_acc']:.4f}->{rec['crf_acc']:.4f}   "
                  f"tracts {rec['raw_ntract']:7.1f}x->{rec['crf_ntract']:5.2f}x",
                  flush=True)
            (OUT / "crf_results.json").write_text(json.dumps(rows, indent=2))

    print(f"\n{'':<12}{'accuracy':>20}{'tract ratio':>22}{'implied g':>20}")
    for cfg_name in ("freq", "haplo"):
        sub = [r for r in rows if r.get("config", "freq") == cfg_name]
        if not sub:
            continue
        print(f"  [{cfg_name}]")
        for tag, name in (("raw", "thresholded"), ("crf", "Viterbi")):
            a = np.mean([r[f"{tag}_acc"] for r in sub])
            n = np.mean([r[f"{tag}_ntract"] for r in sub])
            g = np.mean([r[f"{tag}_g"] for r in sub if np.isfinite(r[f"{tag}_g"])])
            print(f"  {name:<12}{a:>20.4f}{n:>21.2f}x{g:>20.0f}")
    print(f"{'truth':<12}{'':>20}{1.0:>21.2f}x"
          f"{np.mean([r['true_g'] for r in rows]):>20.0f}")


if __name__ == "__main__":
    main()

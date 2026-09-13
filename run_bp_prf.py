"""Breakpoint precision/recall for every method on the simulated sweep.

Reviewer 2 asked for a breakpoint precision/recall comparison that includes the
released tools, not only the network. Table 2 already gives the cross-method
tract-count comparison; this adds the localisation view (precision, recall, F1
against a fixed tolerance) for the same methods, plotted in the supplement.

This script is deliberately isolated. It writes only results/bp_prf.json and
touches none of the published result files. The released tools are re-run fresh
(their earlier simulated scratch output was cleaned), so their runs here are
independent draws from the same seeded simulation; RFMix in particular is not
bit-reproducible, so the per-site accuracies implied here may differ marginally
from Fig 1 / Table 2, which are left untouched. The network predictions reuse
the cached checkpoints; nothing is retrained.
"""

import os
import tempfile
import json
from pathlib import Path

import numpy as np
import torch

from lai.export import (_dedup_positions, run, write_genetic_map,
                        write_sample_map, write_vcf)
from lai.methods import (DilatedCNN, build_features, hmm_predict,
                        naive_bayes_predict, panel_frequencies, site_log_ratio)
from lai.sim import RHO, SimConfig, simulate_replicate
from run_pilot import NB_WINDOW, WINDOW
from run_external import CHROM, FLARE, JAVA, RFMIX, parse_flare, parse_rfmix
from run_decode_robustness import TOLERANCES_BP, breakpoint_prf
from run_tracts import SPLITS

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
WORK = Path(os.environ.get("LAI_WORKDIR", tempfile.gettempdir())) / "lai-bench" / "bpprf"


def cnn_predict(ckpt, in_ch, adm, ra, rb, n_sites, device):
    model = DilatedCNN(in_ch=in_ch).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    pred = np.zeros((n_sites, adm.shape[1]), dtype=np.int8)
    with torch.no_grad():
        for s in range(0, n_sites - WINDOW + 1, WINDOW):
            sl = slice(s, s + WINDOW)
            fa, fb = panel_frequencies(ra[sl], rb[sl])
            feats = (build_features(adm[sl], fa, fb) if in_ch == 4
                     else build_features(adm[sl], fa, fb, ra[sl], rb[sl]))
            x = torch.from_numpy(feats).to(device)
            out = [(torch.sigmoid(model(x[i:i + 64])) > 0.5).cpu().numpy()
                   for i in range(0, x.shape[0], 64)]
            pred[sl] = np.concatenate(out).T.astype(np.int8)
    last = (n_sites // WINDOW) * WINDOW
    return pred[:last]


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    WORK.mkdir(parents=True, exist_ok=True)
    rows = []
    for T in SPLITS:
        cfg = SimConfig(split_time=T)
        rep = simulate_replicate(cfg, seed=777_000 + T)   # matches run_external seed 0
        pos, keep = _dedup_positions(rep["positions"])
        ra, rb = rep["ref_a"][keep], rep["ref_b"][keep]
        adm, truth = rep["admixed"][keep], rep["labels"][keep]
        n_sites, n_hap = adm.shape

        preds = {}
        # released tools, re-run fresh to scratch (independent draws)
        d = WORK / f"T{T}"
        d.mkdir(exist_ok=True)
        names = write_vcf(d / "ref.vcf.gz", np.hstack([ra, rb]), pos, CHROM, "REF")
        write_vcf(d / "query.vcf.gz", adm, pos, CHROM, "ADM")
        write_genetic_map(d / "map.rfmix.tsv", pos, CHROM)
        write_genetic_map(d / "map.plink.tsv", pos, CHROM, plink=True)
        half = ra.shape[1] // 2
        write_sample_map(d / "samples.tsv", names[:half], names[half:])
        ok_r, _ = run([str(RFMIX), "-f", str(d / "query.vcf.gz"),
                       "-r", str(d / "ref.vcf.gz"), "-m", str(d / "samples.tsv"),
                       "-g", str(d / "map.rfmix.tsv"), "-o", str(d / "rfmix"),
                       f"--chromosome={CHROM}", "-G", str(cfg.admix_generations)],
                      d / "rfmix.log")
        if ok_r and (d / "rfmix.msp.tsv").exists():
            preds["rfmix"] = parse_rfmix(d / "rfmix.msp.tsv", n_sites, pos, n_hap)
        ok_f, _ = run([JAVA, "-Xmx8g", "-jar", str(FLARE),
                       f"ref={d / 'ref.vcf.gz'}", f"ref-panel={d / 'samples.tsv'}",
                       f"gt={d / 'query.vcf.gz'}", f"map={d / 'map.plink.tsv'}",
                       f"out={d / 'flare'}", "min-mac=1", "min-maf=0",
                       f"gen={cfg.admix_generations}", "nthreads=4"], d / "flare.log")
        if ok_f and (d / "flare.anc.vcf.gz").exists():
            preds["flare"] = parse_flare(d / "flare.anc.vcf.gz", n_sites, pos, n_hap)

        # our own methods on the identical sites
        pa, pb = panel_frequencies(ra, rb)
        lr = site_log_ratio(adm, pa, pb)
        nb, win = naive_bayes_predict(lr, NB_WINDOW)
        preds["naive_bayes"] = nb
        span = (pos[-1] - pos[0]) / (n_sites / NB_WINDOW)
        sp = float(np.clip(cfg.admix_generations * RHO * span, 1e-6, 0.4))
        preds["hmm"] = np.repeat(hmm_predict(win, sp), NB_WINDOW, axis=0)

        ck = OUT / "cache" / f"cnn_T{T}_s0.pt"
        if ck.exists():
            preds["cnn"] = cnn_predict(ck, 4, adm, ra, rb, n_sites, device)
        ckh = OUT / "cache" / f"cnn_haplo_T{T}.pt"
        if ckh.exists():
            preds["cnn_haplo"] = cnn_predict(ckh, 8, adm, ra, rb, n_sites, device)

        rec = {"split_time": T, "fst": float(rep["fst"])}
        for name, pred in preds.items():
            m = pred.shape[0]
            st = {}
            for tol in TOLERANCES_BP:
                pr, rc, f1 = breakpoint_prf(pred, truth[:m], pos[:m], tol)
                kb = tol // 1000
                st[f"prec_{kb}kb"], st[f"rec_{kb}kb"], st[f"f1_{kb}kb"] = pr, rc, f1
            rec[name] = st
        rows.append(rec)
        print(f"Fst={rep['fst']:.5f}  " + "  ".join(
            f"{k}:F1={rec[k]['f1_1000kb']:.2f}" for k in
            ("naive_bayes", "hmm", "rfmix", "flare", "cnn", "cnn_haplo")
            if k in rec), flush=True)
        (OUT / "bp_prf.json").write_text(json.dumps(rows, indent=2))
    print("wrote results/bp_prf.json")


if __name__ == "__main__":
    main()

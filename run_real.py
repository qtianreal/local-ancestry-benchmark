"""Real-haplotype arm: 1000 Genomes source panels, identical everything else.

Two differences from the simulation arm require care.

First, there is only one set of real haplotypes, so replicates cannot be
independent draws from a generative process. Replication here comes from
disjoint genomic segments plus independent reference/donor partitions, which
is weaker and is reported as such.

Second, and more importantly, training and evaluation windows are taken from
*genomically disjoint* segments separated by a buffer. Within the simulation
arm each replicate was an independent coalescent realisation, so train/test
independence was automatic. On a real chromosome, linkage disequilibrium makes
nearby windows correlated, and sampling train and test windows from the same
segment would leak.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lai.methods import (
    build_features,
    hmm_predict,
    naive_bayes_predict,
    panel_frequencies,
    site_log_ratio,
)
from lai.real import load_region, split_panel
from lai.sim import RHO, hudson_fst, make_admixed
from run_pilot import NB_WINDOW, WINDOW, eval_cnn, train_cnn

OUT = Path("results")


class RealConfig:
    """Mirrors SimConfig fields consumed by make_admixed."""

    def __init__(self, seq_length, n_admixed=64, admix_generations=30, admix_prop=0.5):
        self.seq_length = seq_length
        self.n_admixed = n_admixed
        self.admix_generations = admix_generations
        self.admix_prop = admix_prop


def build_segment(ref_a, ref_b, donor_a, donor_b, positions, n_admixed, rng):
    """Mosaic donors over one genomic segment and return panels + labels."""
    pos = positions - positions[0]
    cfg = RealConfig(seq_length=float(pos[-1]), n_admixed=n_admixed)
    admixed, labels = make_admixed(cfg, donor_a, donor_b, pos, rng)
    return {"ref_a": ref_a, "ref_b": ref_b, "admixed": admixed,
            "labels": labels, "positions": pos}, cfg


def windows_from(seg, n_windows, rng, width=WINDOW):
    feats, labs = [], []
    n_sites = seg["admixed"].shape[0]
    for _ in range(n_windows):
        s = int(rng.integers(0, n_sites - width))
        sl = slice(s, s + width)
        p_a, p_b = panel_frequencies(seg["ref_a"][sl], seg["ref_b"][sl])
        feats.append(build_features(seg["admixed"][sl], p_a, p_b))
        labs.append(seg["labels"][sl].T.astype(np.float32))
    return (torch.from_numpy(np.concatenate(feats)),
            torch.from_numpy(np.concatenate(labs)))


def baseline_on(seg, start, admix_generations, width=WINDOW):
    sl = slice(start, start + width)
    p_a, p_b = panel_frequencies(seg["ref_a"][sl], seg["ref_b"][sl])
    lr = site_log_ratio(seg["admixed"][sl], p_a, p_b)
    nb_pred, win_score = naive_bayes_predict(lr, NB_WINDOW)
    truth = seg["labels"][sl][: nb_pred.shape[0]]
    pos = seg["positions"][sl]
    span = (pos[-1] - pos[0]) / (len(pos) / NB_WINDOW)
    sp = float(np.clip(admix_generations * RHO * span, 1e-6, 0.4))
    hmm = np.repeat(hmm_predict(win_score, sp), NB_WINDOW, axis=0)
    return float((nb_pred == truth).mean()), float((hmm == truth).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vcf", default="data/chr22.vcf.gz")
    ap.add_argument("--chrom", default="chr22")
    ap.add_argument("--start", type=int, default=16_000_000)
    ap.add_argument("--end", type=int, default=51_000_000)
    ap.add_argument("--pops", default="CHB,CHS")
    ap.add_argument("--n-ref", type=int, default=80)
    ap.add_argument("--n-donor", type=int, default=80)
    ap.add_argument("--reps", type=int, default=6, help="ref/donor partitions per segment")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--dilations", default="",
                    help="comma-separated; empty uses the default stack")
    args = ap.parse_args()

    pops = args.pops.split(",")
    print(f"loading {pops} from {args.chrom}:{args.start}-{args.end} ...", flush=True)
    haps, positions = load_region(args.vcf, args.chrom, args.start, args.end, pops)
    n_sites = positions.size
    print(f"{n_sites} biallelic SNVs; haplotypes: "
          + ", ".join(f"{p}={haps[p].shape[1]}" for p in pops), flush=True)

    fst = hudson_fst(haps[pops[0]], haps[pops[1]])
    print(f"Hudson Fst({pops[0]}, {pops[1]}) = {fst:.5f}", flush=True)

    # Genomically disjoint train / test segments with a buffer between them.
    cut = int(n_sites * 0.60)
    buf = int(n_sites * 0.05)
    tr_sl = slice(0, cut)
    te_sl = slice(cut + buf, n_sites)
    print(f"train sites {tr_sl.start}-{tr_sl.stop}, test sites {te_sl.start}-{te_sl.stop} "
          f"(buffer {buf} sites)", flush=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    nb_all, hmm_all, xtr_l, ytr_l, xte_l, yte_l = [], [], [], [], [], []

    for rep in range(args.reps):
        rng = np.random.default_rng(4242 + rep)
        ra, da = split_panel(haps[pops[0]], args.n_ref, args.n_donor, rng)
        rb, db = split_panel(haps[pops[1]], args.n_ref, args.n_donor, rng)

        seg_tr, cfg = build_segment(ra[tr_sl], rb[tr_sl], da[tr_sl], db[tr_sl],
                                    positions[tr_sl], 64, rng)
        seg_te, _ = build_segment(ra[te_sl], rb[te_sl], da[te_sl], db[te_sl],
                                  positions[te_sl], 64, rng)

        x, y = windows_from(seg_tr, 3, rng)
        xtr_l.append(x); ytr_l.append(y)
        x, y = windows_from(seg_te, 1, np.random.default_rng(999 + rep))
        xte_l.append(x); yte_l.append(y)

        start = (seg_te["admixed"].shape[0] - WINDOW) // 2
        nb, hmm = baseline_on(seg_te, start, cfg.admix_generations)
        nb_all.append(nb); hmm_all.append(hmm)
        print(f"  rep {rep}: NB={nb:.4f} HMM={hmm:.4f}", flush=True)

    xtr, ytr = torch.cat(xtr_l), torch.cat(ytr_l)
    xte, yte = torch.cat(xte_l), torch.cat(yte_l)
    print(f"train {tuple(xtr.shape)}  test {tuple(xte.shape)}", flush=True)

    dils = tuple(int(x) for x in args.dilations.split(",")) if args.dilations else None
    model, _ = train_cnn(xtr, ytr, device, args.epochs, seed=7,
                         width=args.width, dilations=dils)
    cnn = eval_cnn(model, xte, yte, device)

    rec = {"pops": pops, "chrom": args.chrom, "start": args.start, "end": args.end,
           "n_sites": int(n_sites), "fst": fst,
           "naive_bayes": float(np.mean(nb_all)), "naive_bayes_sd": float(np.std(nb_all)),
           "hmm": float(np.mean(hmm_all)), "hmm_sd": float(np.std(hmm_all)),
           "cnn": cnn, "reps": args.reps,
           "n_ref": args.n_ref, "n_donor": args.n_donor}
    tag = "_".join(pops) + (f"_{args.dilations.replace(',','-')}" if args.dilations else "")
    (OUT / f"real_{tag}.json").write_text(json.dumps(rec, indent=2))
    print(f"\n{tag}: Fst={fst:.5f}  NB={rec['naive_bayes']:.4f}  "
          f"HMM={rec['hmm']:.4f}  CNN={cnn:.4f}  (gap {cnn - rec['hmm']:+.4f})")
    torch.save(model.state_dict(), OUT / "cache" / f"cnn_real_{tag}.pt")


if __name__ == "__main__":
    main()

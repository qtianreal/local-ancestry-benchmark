"""Give the network the information the released tools get, and re-measure.

RFMix and FLARE consume individual reference haplotypes and a genetic map. The
network consumes panel allele frequencies and no positional information, so it
treats consecutive segregating sites as equally spaced. The manuscript
discloses that asymmetry and names closing it as the open question; this runs
the experiment.

    freq    the published four channels
    dist    + log inter-site spacing, standardised per window. This is the
            information a constant-rate genetic map carries, and it is free.
    haplo   + four top-k haplotype-match channels against each panel. This is
            the copying signal, and it is expensive: match scoring is
            O(sites x targets x panel).
    both    dist + haplo

The protocol is run_real.py's, not the reduced single-partition harness used
for the architecture screens: six independent reference/donor partitions,
genomically disjoint train and test segments with a buffer, so the numbers are
comparable with Table 3 rather than only with each other. Seeds vary the
network initialisation with the data held fixed, so configurations pair on
seed. Pooling six partitions damps initialisation noise: the baseline seed
SD here is ~0.006, against ~0.010 in the single-partition screen harness, so
effects near 0.01 are resolvable where the screens could not see 0.02.

Results go to results/tuning/ so they cannot be picked up by the globs that
build the manuscript's real-data table and figure.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lai.methods import (
    DilatedCNN,
    build_features,
    distance_channel,
    matchlen_channels,
    panel_frequencies,
)
from lai.real import load_region, split_panel
from lai.sim import hudson_fst
from run_attn import train
from run_pilot import WINDOW
from run_real import RealConfig, build_segment
from run_tracts import tract_stats

OUT = Path("results/tuning")

CHANNELS = {"freq": 4, "dist": 5, "haplo": 8, "both": 9,
            # match-length channels: the copying-model statistic the rate
            # channels discard. "mlen" replaces the rate channels, "hapmlen"
            # keeps both, since the two were complementary in the single-
            # feature comparison (rate led on CEU/TSI, length on the rest).
            "mlen": 8, "hapmlen": 12}


def windows_from(seg, n_windows, rng, mode, width=WINDOW):
    feats, labs = [], []
    n_sites = seg["admixed"].shape[0]
    for _ in range(n_windows):
        s = int(rng.integers(0, n_sites - width))
        sl = slice(s, s + width)
        p_a, p_b = panel_frequencies(seg["ref_a"][sl], seg["ref_b"][sl])
        if mode in ("haplo", "both", "hapmlen"):
            f = build_features(seg["admixed"][sl], p_a, p_b,
                               seg["ref_a"][sl], seg["ref_b"][sl])
        else:
            f = build_features(seg["admixed"][sl], p_a, p_b)
        if mode in ("mlen", "hapmlen"):
            ml = matchlen_channels(seg["admixed"][sl], seg["ref_a"][sl],
                                   seg["ref_b"][sl])
            f = np.concatenate([f, ml], axis=1)
        if mode in ("dist", "both"):
            d = distance_channel(seg["positions"][sl], f.shape[0])
            f = np.concatenate([f, d], axis=1)
        feats.append(f)
        labs.append(seg["labels"][sl].T.astype(np.float32))
    return (torch.from_numpy(np.concatenate(feats)),
            torch.from_numpy(np.concatenate(labs)))


def evaluate(model, x, y, device, batch=64):
    """Per-site accuracy and tract structure from the same predictions.

    Accuracy alone is the wrong primary metric for positional information:
    Viterbi decoding, which uses inter-site distance, cut fragmentation 71-fold
    while moving accuracy by 0.0002. If a distance channel helps anything it
    should show in tract structure, and scoring accuracy only would discard it.
    """
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, x.shape[0], batch):
            p = (torch.sigmoid(model(x[i:i + batch].to(device))) > 0.5)
            preds.append(p.to(torch.int8).cpu().numpy())
    pred = np.concatenate(preds)                 # (n_windows*n_haps, n_sites)
    truth = y.numpy().astype(np.int8)
    acc = float((pred == truth).mean())
    # tract_stats wants (n_sites, n_haps); each row here is one haplotype window.
    ts = tract_stats(pred.T, truth.T)
    return acc, ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pops", default="CHB,CDX")
    ap.add_argument("--vcf", default="data/chr22.vcf.gz")
    ap.add_argument("--features", default="freq,dist",
                    help="comma-separated subset of " + ",".join(CHANNELS))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n-ref", type=int, default=80,
                    help="reference haplotypes per population. Expected "
                         "set-maximal match length grows with panel size, so "
                         "this interacts with the match-length channels.")
    ap.add_argument("--n-donor", type=int, default=80)
    ap.add_argument("--fixed-donors", action="store_true",
                    help="take donors from the end of the permutation so the "
                         "evaluation data do not change with --n-ref")
    ap.add_argument("--tag", default="",
                    help="suffix for the results file, to keep panel-size "
                         "variants from overwriting the default run")
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=15)
    args = ap.parse_args()

    pops = args.pops.split(",")
    haps, positions = load_region(args.vcf, "chr22", 16_000_000, 51_000_000, pops)
    fst = hudson_fst(haps[pops[0]], haps[pops[1]])
    n = positions.size
    cut, buf = int(n * 0.60), int(n * 0.05)
    tr_sl, te_sl = slice(0, cut), slice(cut + buf, n)
    print(f"{'/'.join(pops)}: {n} sites, Fst={fst:.5f}", flush=True)

    # Build the six partitions once; every configuration and seed reuses them,
    # so nothing but the input representation and the initialisation varies.
    segs = []
    for rep in range(args.reps):
        rng = np.random.default_rng(4242 + rep)
        ra, da = split_panel(haps[pops[0]], args.n_ref, args.n_donor, rng,
                             donors_from_end=args.fixed_donors)
        rb, db = split_panel(haps[pops[1]], args.n_ref, args.n_donor, rng,
                             donors_from_end=args.fixed_donors)
        seg_tr, _ = build_segment(ra[tr_sl], rb[tr_sl], da[tr_sl], db[tr_sl],
                                  positions[tr_sl], 64, rng)
        seg_te, _ = build_segment(ra[te_sl], rb[te_sl], da[te_sl], db[te_sl],
                                  positions[te_sl], 64, rng)
        segs.append((rep, seg_tr, seg_te))

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # Merge with any earlier invocation for this pair, so feature sets can be
    # added incrementally without discarding the baseline they pair against.
    out_path = OUT / f"inputs_{'_'.join(pops)}{args.tag}.json"
    rows = json.loads(out_path.read_text()) if out_path.exists() else []
    fresh = set(args.features.split(","))
    rows = [r for r in rows if r["features"] not in fresh]
    if rows:
        print(f"  merging with {len(rows)} existing rows "
              f"({sorted({r['features'] for r in rows})})", flush=True)
    for mode in args.features.split(","):
        xs, ys, xe, ye = [], [], [], []
        for rep, seg_tr, seg_te in segs:
            rng = np.random.default_rng(4242 + rep)
            x, y = windows_from(seg_tr, 3, rng, mode)
            xs.append(x); ys.append(y)
            x, y = windows_from(seg_te, 1, np.random.default_rng(999 + rep), mode)
            xe.append(x); ye.append(y)
        xtr, ytr = torch.cat(xs), torch.cat(ys)
        xte, yte = torch.cat(xe), torch.cat(ye)
        print(f"  [{mode}] train {tuple(xtr.shape)} test {tuple(xte.shape)}", flush=True)

        for seed in range(args.seeds):
            torch.manual_seed(seed)
            model = DilatedCNN(in_ch=CHANNELS[mode]).to(device)
            model = train(model, xtr, ytr, device, args.epochs, 1e-3, 32)
            acc, ts = evaluate(model, xte, yte, device)
            rows.append({"pops": pops, "fst": fst, "features": mode, "seed": seed,
                         "in_ch": CHANNELS[mode], "acc": acc, "reps": args.reps,
                         "n_ref": args.n_ref,
                         **ts})
            print(f"    {mode:<6} seed {seed}: acc={acc:.4f}  "
                  f"tracts={ts['n_tract_ratio']:.1f}x  "
                  f"len={ts['length_ratio']:.2f}", flush=True)
            OUT.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(rows, indent=2))

    print(f"\n--- {'/'.join(pops)} paired on seed, vs freq ---", flush=True)
    base = {r["seed"]: r["acc"] for r in rows if r["features"] == "freq"}
    present = [m for m in CHANNELS if any(r["features"] == m for r in rows)]
    for mode in present:
        sel = [r for r in rows if r["features"] == mode]
        a = [r["acc"] for r in sel]
        d = [r["acc"] - base[r["seed"]] for r in sel if r["seed"] in base]
        tr = [r["n_tract_ratio"] for r in sel]
        print(f"  {mode:<6} acc {np.mean(a):.4f} +/- {np.std(a):.4f}   "
              f"paired delta {np.mean(d):+.4f}   "
              f"tract ratio {np.mean(tr):.1f}x", flush=True)


if __name__ == "__main__":
    main()

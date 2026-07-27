"""Is there headroom in combining the CNN with a released tool?

Any combiner -- an extra input channel carrying FLARE's posterior, stacking,
distillation -- is bounded above by the *oracle*: the accuracy obtained if an
omniscient selector picked the better method at every site. If the methods err
in the same places the oracle sits on top of the best single method and no
architecture can extract anything. If their errors are decorrelated the
headroom is real.

This reuses the external tools' saved per-site output and the cached CNN
weights, so nothing is retrained and no tool is re-run.

Reported per pair:

    single-method accuracies       must reproduce results/realext_*.json
    oracle(CNN, FLARE)             upper bound on any CNN+FLARE combiner
    oracle(CNN, FLARE, RFMix)      upper bound with both tools
    vote(CNN, FLARE, RFMix)        an actually-deployable combiner, no training
    P(CNN right | FLARE wrong)     the exploitable complement

The alignment between the CNN's site indexing and the tools' is the one thing
that can silently invalidate all of this, so the recomputed tool accuracies
are checked against the stored values and the script aborts on mismatch.
"""

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from lai.export import _dedup_positions
from lai.methods import DilatedCNN, build_features, panel_frequencies
from lai.real import load_region, split_panel
from lai.sim import hudson_fst, make_admixed
from run_external import parse_flare, parse_rfmix, score
from run_pilot import WINDOW
from run_real import RealConfig

WORK = Path(os.environ.get("LAI_WORKDIR", tempfile.gettempdir())) / "lai-bench" / "realext"
OUT = Path("results")


def cnn_calls(model, adm, ref_a, ref_b, device, width=WINDOW):
    """Per-site CNN calls over the whole segment, in training-sized tiles."""
    n_sites = adm.shape[0]
    calls = np.zeros_like(adm, dtype=np.int8)
    model.eval()
    with torch.no_grad():
        for s in range(0, n_sites, width):
            sl = slice(s, min(s + width, n_sites))
            p_a, p_b = panel_frequencies(ref_a[sl], ref_b[sl])
            feats = build_features(adm[sl], p_a, p_b)
            x = torch.from_numpy(feats).to(device)
            pred = (torch.sigmoid(model(x)) > 0.5).to(torch.int8).cpu().numpy()
            calls[sl] = pred.T
    return calls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pops", required=True)
    ap.add_argument("--vcf", default="data/chr22.vcf.gz")
    ap.add_argument("--chrom", default="chr22")
    ap.add_argument("--start", type=int, default=16_000_000)
    ap.add_argument("--end", type=int, default=51_000_000)
    ap.add_argument("--gen", type=int, default=30)
    args = ap.parse_args()

    pops = args.pops.split(",")
    tag = "_".join(pops)
    d = WORK / tag
    stored = json.loads((OUT / f"realext_{tag}.json").read_text())

    haps, positions = load_region(args.vcf, args.chrom, args.start, args.end, pops)
    fst = hudson_fst(haps[pops[0]], haps[pops[1]])

    # Replay run_real_external.py's construction exactly: same seed, same draw
    # order, same held-out segment.
    rng = np.random.default_rng(4242)
    ra, da = split_panel(haps[pops[0]], 80, 80, rng)
    rb, db = split_panel(haps[pops[1]], 80, 80, rng)
    n = positions.size
    te = slice(int(n * 0.65), n)
    pos_te = positions[te] - positions[te][0]
    cfg = RealConfig(seq_length=float(pos_te[-1]), n_admixed=64,
                     admix_generations=args.gen)
    admixed, labels = make_admixed(cfg, da[te], db[te], pos_te, rng)
    pos_int, keep = _dedup_positions(pos_te + 1000)
    adm, truth = admixed[keep], labels[keep]
    rat, rbt = ra[te][keep], rb[te][keep]
    n_sites, n_hap = adm.shape

    tools = {}
    for name, parser, fn in (("rfmix", parse_rfmix, d / "rfmix.msp.tsv"),
                             ("flare", parse_flare, d / "flare.anc.vcf.gz")):
        if not fn.exists():
            print(f"{tag}: {name} output missing, skipping pair")
            return
        c = parser(fn, n_sites, pos_int, n_hap)
        if stored.get(f"{name}_flipped"):
            c = 1 - np.where(c < 0, 0, c)
        acc, _ = score(c, truth)
        if abs(acc - stored[name]) > 1e-6:
            raise SystemExit(f"{tag}: {name} realigned to {acc:.6f} but "
                             f"{stored[name]:.6f} was stored -- alignment is wrong")
        tools[name] = c

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = DilatedCNN().to(device)
    model.load_state_dict(torch.load(OUT / "cache" / f"cnn_real_{tag}.pt",
                                     map_location=device))
    cnn = cnn_calls(model, adm, rat, rbt, device)

    # Restrict to sites every method called, so the comparison is like-for-like.
    valid = (tools["rfmix"] >= 0) & (tools["flare"] >= 0)
    t = truth[valid]
    ok = {"cnn": cnn[valid] == t,
          "flare": tools["flare"][valid] == t,
          "rfmix": tools["rfmix"][valid] == t}

    vote = ((cnn[valid].astype(int) + tools["flare"][valid].astype(int)
             + tools["rfmix"][valid].astype(int)) >= 2).astype(np.int8)
    rec = {
        "pops": pops, "fst": fst, "coverage": float(valid.mean()),
        "cnn": float(ok["cnn"].mean()),
        "flare": float(ok["flare"].mean()),
        "rfmix": float(ok["rfmix"].mean()),
        "oracle_cnn_flare": float((ok["cnn"] | ok["flare"]).mean()),
        "oracle_all": float((ok["cnn"] | ok["flare"] | ok["rfmix"]).mean()),
        "vote": float((vote == t).mean()),
        "agree_cnn_flare": float((cnn[valid] == tools["flare"][valid]).mean()),
        "cnn_right_flare_wrong": float((ok["cnn"] & ~ok["flare"]).mean()),
        "flare_right_cnn_wrong": float((~ok["cnn"] & ok["flare"]).mean()),
    }
    best = max(rec["cnn"], rec["flare"], rec["rfmix"])
    rec["headroom_cnn_flare"] = rec["oracle_cnn_flare"] - max(rec["cnn"], rec["flare"])
    rec["vote_gain"] = rec["vote"] - best

    (OUT / f"oracle_{tag}.json").write_text(json.dumps(rec, indent=2))
    print(f"{tag}  Fst={fst:.4f}  cov={rec['coverage']:.3f}\n"
          f"  CNN={rec['cnn']:.4f}  FLARE={rec['flare']:.4f}  RFMix={rec['rfmix']:.4f}\n"
          f"  oracle(CNN,FLARE)={rec['oracle_cnn_flare']:.4f}  "
          f"headroom over best of the two = {rec['headroom_cnn_flare']:+.4f}\n"
          f"  oracle(all three)={rec['oracle_all']:.4f}\n"
          f"  vote(all three)={rec['vote']:.4f}  vs best single {best:.4f} "
          f"({rec['vote_gain']:+.4f})\n"
          f"  CNN right where FLARE wrong: {rec['cnn_right_flare_wrong']:.4f}; "
          f"converse {rec['flare_right_cnn_wrong']:.4f}", flush=True)


if __name__ == "__main__":
    main()

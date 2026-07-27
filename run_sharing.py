"""Does simulated data carry as much haplotype sharing as real data?

The manuscript explains the network's real-data deficit by arguing that a
clean split with constant effective size generates less haplotype sharing than
real chromosomes carry. That is an empirical claim about the simulator, and it
can be tested without reference to any network.

We measure how well haplotype matching *alone* separates ancestry. For each
site of an admixed haplotype we compute the mean of the top-k agreement
fractions against each reference panel in a local window, call the ancestry
whichever panel matches better, and score against the exact labels. This is a
single-feature classifier using only haplotype identity -- no allele
frequencies, no learning, no smoothing -- so it isolates the quantity in
question.

Run over matched F_ST in simulation and on real 1000 Genomes pairs. If the
real curve sits above the simulated one at the same divergence, the simulator
understates the exploitable haplotype signal and the manuscript's explanation
holds. If the curves coincide, the explanation needs revisiting.

Note the channels used by the network are standardised per channel, which
destroys the between-panel comparison; this computes the raw agreement
fractions instead.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from lai.real import load_region, split_panel
from lai.sim import SimConfig, hudson_fst, simulate_replicate
from run_pilot import WINDOW
from run_real import build_segment

OUT = Path("results")
MATCH_WINDOW = 64
TOPK = 5


def topk_match(target, ref, window=MATCH_WINDOW, topk=TOPK):
    """Mean of the top-k agreement fractions, per site. target (S,), ref (S,R)."""
    n = target.size
    match = (target[:, None] == ref).astype(np.int32)      # (S, R)
    csum = np.zeros((n + 1, ref.shape[1]), np.int32)
    np.cumsum(match, axis=0, out=csum[1:])
    idx = np.arange(n)
    hi = np.minimum(idx + window + 1, n)
    lo = np.maximum(idx - window, 0)
    frac = (csum[hi] - csum[lo]).astype(np.float32) / (hi - lo)[:, None]
    part = np.partition(frac, -topk, axis=1)
    return part[:, -topk:].mean(axis=1)


def sharing_accuracy(admixed, labels, ref_a, ref_b, n_targets=64, rng=None):
    """Accuracy of calling ancestry by whichever panel matches better."""
    rng = rng or np.random.default_rng(0)
    cols = rng.choice(admixed.shape[1], size=min(n_targets, admixed.shape[1]),
                      replace=False)
    correct = total = 0
    gaps = []
    for j in cols:
        ma = topk_match(admixed[:, j], ref_a)
        mb = topk_match(admixed[:, j], ref_b)
        call = (mb > ma).astype(np.int8)          # 1 => ancestry B
        truth = labels[:, j]
        correct += int((call == truth).sum())
        total += truth.size
        # Signed margin toward the true panel, in agreement-fraction units.
        gaps.append(float(np.mean(np.where(truth == 0, ma - mb, mb - ma))))
    return correct / total, float(np.mean(gaps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-times", default="25,50,100,200,400,800")
    ap.add_argument("--pairs", default="CHB,CHS;CEU,TSI;FIN,GBR;CHB,CDX;TSI,PJL;CEU,GIH")
    ap.add_argument("--vcf", default="data/chr22.vcf.gz")
    ap.add_argument("--sites", type=int, default=WINDOW)
    ap.add_argument("--reps", type=int, default=3,
                    help="independent simulated replicates / donor partitions")
    args = ap.parse_args()

    rows = []
    for T in [int(x) for x in args.split_times.split(",")]:
        accs, gaps, fsts = [], [], []
        for i in range(args.reps):
            rep = simulate_replicate(SimConfig(split_time=T), seed=777_000 + T + 31 * i)
            sl = slice(0, min(args.sites, rep["admixed"].shape[0]))
            a, g = sharing_accuracy(rep["admixed"][sl], rep["labels"][sl],
                                    rep["ref_a"][sl], rep["ref_b"][sl],
                                    rng=np.random.default_rng(i))
            accs.append(a); gaps.append(g); fsts.append(rep["fst"])
        rows.append({"source": "simulated", "label": f"T={T}",
                     "fst": float(np.mean(fsts)),
                     "match_accuracy": float(np.mean(accs)),
                     "match_accuracy_sd": float(np.std(accs)),
                     "match_margin": float(np.mean(gaps)), "reps": args.reps})
        print(f"  sim  T={T:<5} Fst={np.mean(fsts):.5f}  match-only acc="
              f"{np.mean(accs):.4f}+/-{np.std(accs):.4f}  margin={np.mean(gaps):+.5f}",
              flush=True)

    for spec in args.pairs.split(";"):
        pops = spec.split(",")
        haps, positions = load_region(args.vcf, "chr22", 16_000_000, 51_000_000, pops)
        fst = hudson_fst(haps[pops[0]], haps[pops[1]])
        accs, gaps = [], []
        n = positions.size
        te = slice(int(n * 0.65), n)
        for i in range(args.reps):
            rng = np.random.default_rng(4242 + i)
            ra, da = split_panel(haps[pops[0]], 80, 80, rng)
            rb, db = split_panel(haps[pops[1]], 80, 80, rng)
            seg, _ = build_segment(ra[te], rb[te], da[te], db[te],
                                   positions[te], 64, rng)
            sl = slice(0, min(args.sites, seg["admixed"].shape[0]))
            a, g = sharing_accuracy(seg["admixed"][sl], seg["labels"][sl],
                                    seg["ref_a"][sl], seg["ref_b"][sl],
                                    rng=np.random.default_rng(i))
            accs.append(a); gaps.append(g)
        rows.append({"source": "real", "label": "/".join(pops), "fst": fst,
                     "match_accuracy": float(np.mean(accs)),
                     "match_accuracy_sd": float(np.std(accs)),
                     "match_margin": float(np.mean(gaps)), "reps": args.reps})
        print(f"  real {'/'.join(pops):<9} Fst={fst:.5f}  match-only acc="
              f"{np.mean(accs):.4f}+/-{np.std(accs):.4f}  margin={np.mean(gaps):+.5f}",
              flush=True)

    (OUT / "sharing.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {OUT / 'sharing.json'}")


if __name__ == "__main__":
    main()

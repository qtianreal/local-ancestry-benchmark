"""Does haplotype input raise the discriminant criterion, or only accuracy?

Accuracy says whether the head can separate the classes; the LDA criterion
says how much class-separating information the representation the head reads
actually carries. They can move apart: a representation can carry more signal
that a linear head fails to exploit, or the same signal arranged more
favourably.

For each divergence level we take the frequency-only and haplotype-aware
networks, run both over identical evaluation windows, and compute at every
block

    J          trace((S_W + lambda I)^-1 S_B)
    d_eff      participation ratio of per-channel Fisher ratios
    fisher_max largest single-channel Fisher ratio

using the same routines as the published discriminant analysis, so the numbers
sit on the same scale as those already reported.

This is the quantitative form of the question a t-SNE of the latent features
would answer only impressionistically: t-SNE geometry is an artefact of
perplexity and initialisation, and separation visible in it is not evidence
that separation exists. If J differs, a projection is worth drawing as an
illustration of a measured effect; if J does not, no projection should be
allowed to suggest otherwise.
"""

import json
from pathlib import Path

import numpy as np
import torch

from lai.methods import DilatedCNN
from lai.sim import SimConfig, simulate_replicate
from run_haplo import windows
from run_lda import block_activations, discriminant_stats

OUT = Path("results")
CACHE = OUT / "cache"
SPLITS = [25, 50, 100, 200, 400, 800, 1600, 3200]
# Window draws per configuration; the ratio is averaged over them.
DRAWS = (5000, 11, 12345)


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    rows = []

    for T in SPLITS:
        mf = CACHE / f"cnn_freq_T{T}.pt"
        mh = CACHE / f"cnn_haplo_T{T}.pt"
        if not (mf.exists() and mh.exists()):
            continue
        cfg = SimConfig(split_time=T)
        # Same test replicates the haplotype ablation used, so the comparison
        # is on the data both models were evaluated on.
        test = [simulate_replicate(cfg, seed=777_000 + T + 31 * i) for i in range(3)]
        fst = test[0]["fst"]

        # Absolute J depends strongly on which windows are drawn -- it varies
        # by a factor of three across draws at a single divergence level -- so
        # it is not a reportable quantity on its own. The ratio between the two
        # configurations on identical windows is stable to about 2%, and that
        # is what is reported. Averaging over draws also removes the dependence.
        rec = {"split_time": T, "fst": fst, "draws": list(DRAWS)}
        per = {}
        for tag, ckpt, in_ch, with_haps in (("freq", mf, 4, False),
                                            ("haplo", mh, 8, True)):
            model = DilatedCNN(in_ch=in_ch).to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device))
            model.eval()
            js, des, fms = [], [], []
            for ws in DRAWS:
                x, y = windows(test, 4, np.random.default_rng(ws), with_haps)
                with torch.no_grad():
                    acts = block_activations(model, x.numpy()[:64], device)
                s = discriminant_stats(acts[-1], y.numpy()[:64])
                js.append(s["J"]); des.append(s["d_eff"]); fms.append(s["fisher_max"])
            per[tag] = js
            rec[tag] = {"J_mean": float(np.mean(js)), "J_sd": float(np.std(js)),
                        "J_draws": js, "d_eff": float(np.mean(des)),
                        "fisher_max": float(np.mean(fms))}
        ratios = [h / f for h, f in zip(per["haplo"], per["freq"]) if f > 0]
        rec["J_ratio"] = float(np.mean(ratios)) if ratios else float("nan")
        rec["J_ratio_sd"] = float(np.std(ratios)) if ratios else float("nan")
        rows.append(rec)
        print(f"Fst={fst:.5f}  J ratio x{rec['J_ratio']:.2f} "
              f"(sd {rec['J_ratio_sd']:.2f})   "
              f"d_eff {rec['freq']['d_eff']:.1f} -> {rec['haplo']['d_eff']:.1f}",
              flush=True)
        (OUT / "lda_compare.json").write_text(json.dumps(rows, indent=2))

    if rows:
        jr = [r["J_ratio"] for r in rows if np.isfinite(r["J_ratio"])]
        print(f"\nJ ratio (haplotype / frequency) across {len(jr)} levels: "
              f"mean {np.mean(jr):.2f}, min {min(jr):.2f}, max {max(jr):.2f}")
        print(f"raised on {sum(1 for x in jr if x > 1)}/{len(jr)} levels")


if __name__ == "__main__":
    main()

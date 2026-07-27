"""Collect every intervention we tried that was not a change of input.

The manuscript claims that architecture, objective, capacity and training data
each move real-data accuracy by little, and that changing what the network is
shown moves it by more. That claim was previously asserted from a single
sentence with no table behind it. This assembles the underlying runs into one
file, which both make_numbers.py and make_figures.py read, so the table, the
figure panel and the prose cannot disagree.

Each entry is paired on seed against its own control, which is why the arms
have different controls: the attention and state-space arms are ablations of
the same trained configuration, the pretraining arms are compared against
training on real data alone, and the discriminant arm against the same
haplotype-aware network without the extra loss term.

Writes results/interventions.json.
"""

import json
from pathlib import Path

import numpy as np

OUT = Path("results")

# name, file, control config, treatment config, pair, what kind of change
ARMS = [
    ("attn", "Self-attention layer", "tuning/attn_CHB_CDX.json",
     "none", "attn-pooled", "CHB/CDX", "architecture"),
    ("ssm", "State-space layer", "tuning/ssm_CHB_CDX.json",
     "none", "mamba-lite", "CHB/CDX", "architecture"),
    ("lda", "Discriminant objective", "tuning/ldatrain_TSI_PJL.json",
     "haplo", "haplo+lda", "TSI/PJL", "objective"),
    ("ssl", "Self-supervised pretraining", "tuning/ssl_CHB_CDX_haplo.json",
     "base", "ssl+base", "CHB/CDX", "training"),
    ("more", "More labelled data", "tuning/ssl_CHB_CDX_haplo.json",
     "base", "more", "CHB/CDX", "training"),
]


def paired(path, control, treatment, key="acc"):
    rows = json.loads((OUT / path).read_text())
    base = {r["seed"]: r[key] for r in rows if r.get("config") == control}
    d = [r[key] - base[r["seed"]] for r in rows
         if r.get("config") == treatment and r["seed"] in base]
    return d


def main():
    out = []
    for tag, label, path, ctrl, treat, pair, kind in ARMS:
        d = paired(path, ctrl, treat)
        out.append({"tag": tag, "label": label, "pair": pair, "kind": kind,
                    "n": len(d), "deltas": [float(x) for x in d],
                    "mean": float(np.mean(d)),
                    "positive": int(np.sum(np.array(d) > 0))})

    # Simulation pretraining is recorded per feature set rather than by config.
    sc = json.loads((OUT / "tuning/scalinghap_CHB_CDX.json").read_text())
    for mode, label in (("freq", "Simulation pretraining, frequency-only"),
                        ("haplo", "Simulation pretraining, haplotype-aware")):
        d = [r["sim_plus_real"] - r["real_only"] for r in sc
             if r["features"] == mode]
        out.append({"tag": f"sim{mode}", "label": label, "pair": "CHB/CDX",
                    "kind": "training", "n": len(d),
                    "deltas": [float(x) for x in d], "mean": float(np.mean(d)),
                    "positive": int(np.sum(np.array(d) > 0))})

    # Capacity: one run per arm, so no pairing is possible. Reported as the
    # largest absolute departure from the baseline across the capacity arms,
    # with the parameter range those arms span.
    tune = json.loads((OUT / "tune_CHB_CDX.json").read_text())
    by = {r["config"]: r for r in tune}
    base = by["baseline"]["acc"]
    caps = [by[c] for c in ("narrow", "wide") if c in by]
    worst = max(caps, key=lambda r: abs(r["acc"] - base))
    out.append({"tag": "cap", "label": "Capacity, 15-fold parameter range",
                "pair": "CHB/CDX", "kind": "capacity", "n": len(caps),
                "unpaired": True,
                "deltas": [float(r["acc"] - base) for r in caps],
                "mean": float(worst["acc"] - base),
                "positive": int(sum(r["acc"] > base for r in caps)),
                "param_lo": min(r["params"] for r in caps),
                "param_hi": max(r["params"] for r in caps),
                "caveat": "single run per arm; sweep selects on an evaluated pair"})

    # The input change, for contrast: the same quantity the manuscript reports
    # for the haplotype channels at the enlarged panel, below the divergence
    # cut where they help.
    mp = json.loads((OUT / "maxpanel_summary.json").read_text())
    d = [v["haplo"] - v["freq"] for v in mp.values() if v["fst"] < 0.04]
    out.append({"tag": "input", "label": "Haplotype-matching channels",
                "pair": f"{len(d)} pairs", "kind": "input",
                "n": len(d), "deltas": [float(x) for x in d],
                "mean": float(np.mean(d)),
                "positive": int(np.sum(np.array(d) > 0))})

    (OUT / "interventions.json").write_text(json.dumps(out, indent=2))
    print(f"{'intervention':<42} {'pair':<10} {'n':>3} {'mean':>9} {'pos':>7}")
    for r in out:
        print(f"{r['label']:<42} {r['pair'][:10]:<10} {r['n']:>3} "
              f"{r['mean']:>+9.4f} {r['positive']:>3}/{r['n']}")
    print(f"\nwrote {OUT / 'interventions.json'}")


if __name__ == "__main__":
    main()

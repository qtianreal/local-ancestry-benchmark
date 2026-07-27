"""How unevenly are segregating sites spaced, in simulation versus real data?

RFMix and FLARE receive a genetic map and therefore know the distance between
consecutive sites; the network receives no positional information and treats
every adjacent pair as equally spaced. That asymmetry costs nothing when sites
are near-uniformly distributed and progressively more as spacing becomes
heavy-tailed, so its size is an empirical question rather than a matter of
principle.

Reports the coefficient of variation of inter-site spacing and the ratio of
the 99th to the 50th percentile, for one simulated replicate and for the real
chromosome 22 region used throughout.
"""

import json
from pathlib import Path

import numpy as np

from lai.real import load_region
from lai.sim import SimConfig, simulate_replicate

OUT = Path("results")


def summarise(pos):
    s = np.diff(np.asarray(pos, dtype=np.float64))
    s = s[s > 0]
    return {"n_gaps": int(s.size), "mean_bp": float(s.mean()),
            "sd_bp": float(s.std()),
            "cv": float(s.std() / s.mean()),
            "p99_over_p50": float(np.percentile(s, 99) / np.percentile(s, 50))}


def main():
    rec = {}
    rep = simulate_replicate(SimConfig(split_time=100), seed=1)
    rec["simulated"] = summarise(rep["positions"])
    _, pos = load_region("data/chr22.vcf.gz", "chr22", 16_000_000, 51_000_000,
                         ["CHB", "CDX"])
    rec["real"] = summarise(pos)
    rec["cv_ratio"] = rec["real"]["cv"] / rec["simulated"]["cv"]
    (OUT / "spacing.json").write_text(json.dumps(rec, indent=2))
    for k in ("simulated", "real"):
        v = rec[k]
        print(f"{k:<10} n={v['n_gaps']:>7}  mean={v['mean_bp']:>7.1f}bp  "
              f"CV={v['cv']:.2f}  p99/p50={v['p99_over_p50']:.1f}")
    print(f"real spacing is {rec['cv_ratio']:.1f}x as variable")


if __name__ == "__main__":
    main()

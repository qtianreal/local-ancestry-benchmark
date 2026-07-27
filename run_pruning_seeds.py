"""Seed-replicated block-group ablation.

Single-block ablation is not used: with residual connections the remaining
blocks compensate for any one removal, which makes per-block importance
systematically understate the importance of a group. Blocks are therefore
removed in contiguous groups from the long-dilation end.

Because accuracy is bounded below by chance, the reported quantity is the
fraction of *above-chance* accuracy that survives removal, which is
comparable across divergence levels in a way that raw accuracy is not.
"""

import json
from pathlib import Path

import numpy as np
import torch

from lai.methods import DilatedCNN

OUT = Path("results")
CACHE = OUT / "cache"
DILATIONS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
CHANCE = 0.5
GROUPS = {"drop_dil8plus": 3, "drop_dil4plus": 2, "drop_dil2plus": 1}


@torch.no_grad()
def accuracy(model, x, y, device, batch=64):
    correct = tot = 0
    for i in range(0, x.shape[0], batch):
        xb = torch.from_numpy(x[i : i + batch].astype(np.float32)).to(device)
        pred = (torch.sigmoid(model(xb)) > 0.5).cpu().numpy().astype(np.int8)
        correct += (pred == y[i : i + batch]).sum()
        tot += y[i : i + batch].size
    return float(correct / tot)


def retention(pruned, full):
    """Fraction of above-chance accuracy retained after pruning."""
    denom = full - CHANCE
    return float((pruned - CHANCE) / denom) if denom > 1e-6 else float("nan")


def main():
    seeds = sorted(int(p.stem.split("_s")[-1]) for p in OUT.glob("main_results_s*.json"))
    fst_by_T = {}
    for s in seeds:
        for r in json.loads((OUT / f"main_results_s{s}.json").read_text()):
            fst_by_T.setdefault(r["split_time"], []).append(r["fst"])

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    records = []

    for T in sorted(fst_by_T):
        for s in seeds:
            mp = CACHE / f"cnn_T{T}_s{s}.pt"
            tp = CACHE / f"test_T{T}_s{s}.npz"
            if not (mp.exists() and tp.exists()):
                continue
            model = DilatedCNN().to(device)
            model.load_state_dict(torch.load(mp, map_location=device))
            model.eval()
            d = np.load(tp)
            x, y = d["x"], d["y"]

            model.skip_blocks = ()
            full = accuracy(model, x, y, device)

            rec = {"split_time": T, "seed": s,
                   "fst": float(np.mean(fst_by_T[T])), "full_acc": full}
            for name, start in GROUPS.items():
                model.skip_blocks = tuple(range(start, len(DILATIONS)))
                acc = accuracy(model, x, y, device)
                rec[name] = acc
                rec[f"{name}_retention"] = retention(acc, full)
            model.skip_blocks = ()
            records.append(rec)

    (OUT / "pruning_seeds.json").write_text(json.dumps(records, indent=2))

    print(f"{'Fst':>8} {'n':>2} {'full':>14} {'retain(dil>=8 removed)':>26}")
    by_T = {}
    for r in records:
        by_T.setdefault(r["split_time"], []).append(r)
    summary = []
    for T in sorted(by_T):
        g = by_T[T]
        full = np.array([r["full_acc"] for r in g])
        ret = np.array([r["drop_dil8plus_retention"] for r in g])
        summary.append({"split_time": T, "fst": g[0]["fst"], "n": len(g),
                        "full_acc": float(full.mean()), "full_acc_sd": float(full.std(ddof=1)) if len(g) > 1 else 0.0,
                        "retention": float(ret.mean()),
                        "retention_sd": float(ret.std(ddof=1)) if len(g) > 1 else 0.0})
        print(f"{g[0]['fst']:8.5f} {len(g):2d} {full.mean():.3f}+/-{(full.std(ddof=1) if len(g)>1 else 0):.3f} "
              f"{ret.mean():18.3f}+/-{(ret.std(ddof=1) if len(g)>1 else 0):.3f}")
    (OUT / "pruning_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""Structured block ablation: which length-scales does the task require?

The network is a residual stack of dilated blocks, so block i contributes
roughly 4 * dilation_i segregating sites of receptive field. Because each
block is residual, deleting it is exactly the identity map -- the network
stays well-formed without retraining, which makes single-block ablation a
clean measure of marginal importance.

Two independent estimates of the length-scale the model relies on:

  1. Ablation importance per block, summarised as an importance-weighted
     mean dilation.
  2. Input-gradient extent: how far from a focal site the gradient of that
     site's logit remains non-negligible.

If the learned method's advantage at low divergence comes from integrating
weak evidence over longer stretches, both should grow as Fst falls.
"""

import json
from pathlib import Path

import numpy as np
import torch

from lai.methods import DilatedCNN

OUT = Path("results")
CACHE = OUT / "cache"
DILATIONS = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def load(T, device):
    model = DilatedCNN().to(device)
    model.load_state_dict(torch.load(CACHE / f"cnn_T{T}.pt", map_location=device))
    model.eval()
    d = np.load(CACHE / f"test_T{T}.npz")
    return model, d["x"], d["y"]


@torch.no_grad()
def accuracy(model, x, y, device, batch=64):
    correct = tot = 0
    for i in range(0, x.shape[0], batch):
        xb = torch.from_numpy(x[i : i + batch].astype(np.float32)).to(device)
        pred = (torch.sigmoid(model(xb)) > 0.5).cpu().numpy().astype(np.int8)
        correct += (pred == y[i : i + batch]).sum()
        tot += y[i : i + batch].size
    return float(correct / tot)


def gradient_extent(model, x, device, n=16, half=512):
    """Mean |d logit_centre / d input| as a function of distance from centre."""
    model.zero_grad(set_to_none=True)
    xb = torch.from_numpy(x[:n].astype(np.float32)).to(device).requires_grad_(True)
    logits = model(xb)
    centre = logits.shape[1] // 2
    logits[:, centre].sum().backward()

    g = xb.grad.abs().sum(dim=1).mean(dim=0).detach().cpu().numpy()  # (n_sites,)
    lo, hi = centre - half, centre + half + 1
    profile = g[lo:hi]
    offsets = np.arange(-half, half + 1)

    total = profile.sum()
    if total <= 0:
        return profile, offsets, float("nan")
    # Half-width containing 90% of gradient mass, in segregating sites.
    order = np.argsort(-profile)
    cum = np.cumsum(profile[order]) / total
    k = int(np.searchsorted(cum, 0.90)) + 1
    width = float(np.abs(offsets[order[:k]]).max())
    return profile, offsets, width


def main():
    rows = json.loads((OUT / "main_results.json").read_text())
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    out = []

    for r in rows:
        T = r["split_time"]
        model, x, y = load(T, device)

        model.skip_blocks = ()
        base = accuracy(model, x, y, device)

        importance = []
        for i in range(len(DILATIONS)):
            model.skip_blocks = (i,)
            importance.append(base - accuracy(model, x, y, device))
        model.skip_blocks = ()

        imp = np.array(importance)
        pos = np.clip(imp, 0, None)
        wmean = (float((pos * np.array(DILATIONS)).sum() / pos.sum())
                 if pos.sum() > 0 else float("nan"))

        _, _, gwidth = gradient_extent(model, x, device)

        rec = {
            "split_time": T,
            "fst": r["fst"],
            "baseline_acc": base,
            "block_importance": [float(v) for v in imp],
            "weighted_mean_dilation": wmean,
            "gradient_half_width_snps": gwidth,
        }
        out.append(rec)
        print(f"Fst={r['fst']:.5f} acc={base:.4f}  wmeanDil={wmean:6.1f}  "
              f"gradHW={gwidth:6.0f}  imp=["
              + " ".join(f"{v:+.3f}" for v in imp) + "]", flush=True)

    (OUT / "pruning_results.json").write_text(json.dumps(out, indent=2))
    print("\nwrote results/pruning_results.json")


if __name__ == "__main__":
    main()

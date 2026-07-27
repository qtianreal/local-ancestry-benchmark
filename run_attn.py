"""Does one self-attention layer help the dilated CNN?

The dilated stack has a receptive field of 2045 SNPs inside a 4096-SNP
evaluation window, so no output site sees the whole window. A single
self-attention layer placed after the stack removes that ceiling: every site
can read every other. If the network's real-data deficit stems from limited
context, this is the cheapest thing that would close it.

Two placements of the same idea, because full-resolution attention over 4096
tokens costs 16.8M pairs per head and may simply be untrainable at this data
scale:

    none          the published architecture, unchanged
    attn-pooled   mean-pool 16x -> attend over 256 tokens -> upsample, added
                  residually. Ancestry tracts here span hundreds of SNPs, so
                  16-SNP pooling discards little of the signal.
    attn-full     attention at full 4096 resolution, no pooling. Opt-in via
                  --full-attn and not part of the reported result: it is the
                  same mechanism as attn-pooled at finer granularity, and
                  ancestry tracts here span hundreds of SNPs, so there is
                  little for the extra resolution to resolve.

Both use 4 heads and are added residually, so at initialisation the model is
the baseline plus a near-zero perturbation; any difference is what attention
learns to add, not a different starting point.

Three seeds per configuration, all sharing identical training and evaluation
windows, so configurations are compared pairwise on seed. A single run cannot
separate an architectural effect from initialisation noise at these accuracies
-- earlier single-run comparisons in this project did not reproduce.

As in run_tune.py, selection happens on the pair the deficit was measured on,
so any gain here is an optimistic bound rather than a reportable accuracy.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lai.methods import DilatedCNN, build_features, panel_frequencies
from lai.real import load_region, split_panel
from lai.sim import hudson_fst, make_admixed
from run_pilot import WINDOW, eval_cnn
from run_real import RealConfig

OUT = Path("results/tuning")
CHUNK = 4  # examples per attention call at full resolution; see SelfAttention


class SelfAttention(nn.Module):
    """One multi-head self-attention layer, residual, optionally pooled.

    The output projection is zero-initialised so the layer starts as an exact
    identity: the baseline and the attention variants begin training from the
    same function, and any divergence is attributable to the layer rather than
    to a perturbed initialisation.
    """

    def __init__(self, width, heads=4, pool=1):
        super().__init__()
        self.heads, self.pool = heads, pool
        self.norm = nn.GroupNorm(8, width)
        self.qkv = nn.Conv1d(width, 3 * width, kernel_size=1)
        self.proj = nn.Conv1d(width, width, kernel_size=1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        n, c, L = x.shape
        h = self.norm(x)
        if self.pool > 1:
            h = F.avg_pool1d(h, self.pool)
        q, k, v = self.qkv(h).chunk(3, dim=1)
        # (N, heads, T, head_dim) for scaled_dot_product_attention
        shape = lambda t: t.view(-1, self.heads, c // self.heads,
                                 t.shape[-1]).transpose(2, 3)
        # At full resolution a batch-32 attention tensor is 32*4*4096^2 = 2^31
        # elements, one past the Metal NDArray limit -- and it aborts the
        # process rather than raising, so it cannot be caught. Attention is
        # independent across examples, so evaluating in batch chunks is
        # arithmetically identical while keeping every allocation small; the
        # optimiser still sees the full batch.
        step = n if self.pool > 1 else max(1, CHUNK)
        out = [F.scaled_dot_product_attention(
                   shape(q[i:i + step]), shape(k[i:i + step]), shape(v[i:i + step]))
               for i in range(0, n, step)]
        a = torch.cat(out, dim=0).transpose(2, 3).reshape(n, c, -1)
        if self.pool > 1:
            a = F.interpolate(a, size=L, mode="linear", align_corners=False)
        return x + self.proj(a)


class AttnCNN(DilatedCNN):
    """The published network with one self-attention layer before the head."""

    def __init__(self, pool, heads=4, **kw):
        super().__init__(**kw)
        self.attn = SelfAttention(kw.get("width", 64), heads=heads, pool=pool)

    def forward(self, x):
        h = self.stem(x)
        for block in self.blocks:
            h = h + block(h)
        return self.head(self.attn(h)).squeeze(1)


def windows(seg, n, rng, width=WINDOW):
    f, l = [], []
    ns = seg["admixed"].shape[0]
    for _ in range(n):
        s = int(rng.integers(0, ns - width))
        sl = slice(s, s + width)
        pa, pb = panel_frequencies(seg["ref_a"][sl], seg["ref_b"][sl])
        f.append(build_features(seg["admixed"][sl], pa, pb))
        l.append(seg["labels"][sl].T.astype(np.float32))
    return (torch.from_numpy(np.concatenate(f)),
            torch.from_numpy(np.concatenate(l)))


def train(model, x, y, device, epochs, lr, batch):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lf = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(x.shape[0])
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            lf(model(x[idx].to(device)), y[idx].to(device)).backward()
            opt.step()
        sch.step()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pops", default="CHB,CDX")
    ap.add_argument("--vcf", default="data/chr22.vcf.gz")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--full-attn", action="store_true",
                    help="also run attention at full 4096 resolution (slow; "
                         "not run by default -- see module docstring)")
    args = ap.parse_args()

    pops = args.pops.split(",")
    haps, positions = load_region(args.vcf, "chr22", 16_000_000, 51_000_000, pops)
    fst = hudson_fst(haps[pops[0]], haps[pops[1]])
    print(f"{'/'.join(pops)}: {positions.size} sites, Fst={fst:.5f}", flush=True)

    rng = np.random.default_rng(4242)
    ra, da = split_panel(haps[pops[0]], 80, 80, rng)
    rb, db = split_panel(haps[pops[1]], 80, 80, rng)

    n = positions.size
    cut, buf = int(n * 0.60), int(n * 0.05)

    def seg(sl, seed):
        pos = positions[sl] - positions[sl][0]
        cfg = RealConfig(seq_length=float(pos[-1]), n_admixed=64)
        adm, lab = make_admixed(cfg, da[sl], db[sl], pos, np.random.default_rng(seed))
        return {"ref_a": ra[sl], "ref_b": rb[sl], "admixed": adm,
                "labels": lab, "positions": pos}

    seg_tr, seg_te = seg(slice(0, cut), 1), seg(slice(cut + buf, n), 2)
    # Fixed window draws: every configuration and seed sees identical data,
    # so differences are architectural and the seeds can be paired.
    xtr, ytr = windows(seg_tr, 18, np.random.default_rng(7))
    xte, yte = windows(seg_te, 6, np.random.default_rng(8))
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"train {tuple(xtr.shape)} test {tuple(xte.shape)} on {device}", flush=True)

    CONFIGS = [("none", None), ("attn-pooled", 16)]
    if args.full_attn:
        CONFIGS.append(("attn-full", 1))
    rows = []
    for name, pool in CONFIGS:
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            model = (DilatedCNN() if pool is None else AttnCNN(pool=pool)).to(device)
            try:
                model = train(model, xtr, ytr, device, args.epochs, 1e-3, args.batch)
                acc = eval_cnn(model, xte, yte, device)
            except RuntimeError as exc:  # e.g. attention at full resolution OOM
                print(f"  {name:<12} seed {seed}: FAILED ({exc})", flush=True)
                continue
            params = sum(p.numel() for p in model.parameters())
            rows.append({"config": name, "seed": seed, "acc": acc,
                         "params": params, "pool": pool, "fst": fst,
                         "pops": pops, "epochs": args.epochs})
            print(f"  {name:<12} seed {seed}: acc={acc:.4f} params={params/1000:.0f}k",
                  flush=True)
            OUT.mkdir(parents=True, exist_ok=True)
            (OUT / f"attn_{'_'.join(pops)}.json").write_text(json.dumps(rows, indent=2))

    print("\n--- paired on seed, vs 'none' ---", flush=True)
    base = {r["seed"]: r["acc"] for r in rows if r["config"] == "none"}
    for name, _ in CONFIGS:
        acc = [r["acc"] for r in rows if r["config"] == name]
        if not acc:
            continue
        d = [r["acc"] - base[r["seed"]] for r in rows
             if r["config"] == name and r["seed"] in base]
        print(f"  {name:<12} {np.mean(acc):.4f} +/- {np.std(acc):.4f}"
              f"   paired delta {np.mean(d):+.4f} (n={len(d)})", flush=True)


if __name__ == "__main__":
    main()

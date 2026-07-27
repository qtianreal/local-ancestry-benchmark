"""A Mamba-style selective state-space layer, as a cheaper route to global context.

Same question as run_attn.py -- does removing the 2045-SNP receptive-field
ceiling help? -- with a linear-time mechanism instead of a quadratic one. A
selective SSM carries a running state along the sequence, so every site sees
every earlier site at O(L) cost rather than O(L^2).

The layer is made **bidirectional** (forward scan + reversed scan, summed).
This matters for a fair comparison: the dilated stack and self-attention are
both non-causal, so a causal SSM would be handicapped by construction and a
null result would be uninterpretable.

    none         the published architecture, unchanged
    mamba        selective SSM, d_state=8, bidirectional
    mamba-lite   d_state=1 -- one decay per channel, ~8x cheaper in the scan

The recurrence h_t = a_t h_{t-1} + b_t is evaluated with a Hillis-Steele
associative scan (log2(L)=12 steps for a 4096-SNP window) rather than a Python
loop over 4096 timesteps. The scan is verified against a sequential reference
in test_scan() below.

As in run_attn.py: output projections are zero-initialised so each variant is
*exactly* the baseline at initialisation, three seeds share identical windows
so configurations pair on seed, and results are quarantined to results/tuning/
because selection happens on the evaluation pair.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lai.methods import DilatedCNN
from lai.real import load_region, split_panel
from lai.sim import hudson_fst
from run_attn import OUT, train, windows
from run_pilot import eval_cnn
from run_real import RealConfig


def pscan(a, b):
    """Inclusive scan of h_t = a_t * h_{t-1} + b_t along the last dim.

    Composing (A,B) at t with (A,B) at t-d gives h_t = A_t A_{t-d} h_{t-2d} +
    (B_t + A_t B_{t-d}), so B must be updated before A. h_{-1} = 0, hence the
    identity padding (1 for A, 0 for B).
    """
    L = a.shape[-1]
    d = 1
    while d < L:
        a_sh = F.pad(a[..., :-d], (d, 0), value=1.0)
        b_sh = F.pad(b[..., :-d], (d, 0), value=0.0)
        b = b + a * b_sh
        a = a * a_sh
        d *= 2
    return b


class MambaBlock(nn.Module):
    """Minimal selective SSM (S6) block, bidirectional, residual, zero-init out."""

    def __init__(self, width, d_state=8, d_conv=4):
        super().__init__()
        self.d_state = d_state
        self.norm = nn.GroupNorm(8, width)
        self.in_proj = nn.Conv1d(width, 2 * width, kernel_size=1)  # x and gate z
        self.conv = nn.Conv1d(width, width, kernel_size=d_conv,
                              padding=d_conv - 1, groups=width)
        self.d_conv = d_conv
        # Selective parameters: per-timestep step size, input and output maps.
        self.x_proj = nn.Conv1d(width, 2 * d_state + 1, kernel_size=1)
        self.dt_bias = nn.Parameter(torch.zeros(width))
        # A is learned through log to keep the decay exp(dt*A) inside (0,1).
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32))
            .repeat(width, 1))
        self.D = nn.Parameter(torch.ones(width))
        self.out_proj = nn.Conv1d(width, width, kernel_size=1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def scan(self, x, dt, B, C):
        """x,dt: (N,C,L); B,C: (N,S,L) -> (N,C,L)."""
        A = -torch.exp(self.A_log)[None, :, :, None]        # (1,C,S,1)
        dA = torch.exp(dt[:, :, None, :] * A)               # (N,C,S,L)
        dBx = dt[:, :, None, :] * B[:, None, :, :] * x[:, :, None, :]
        h = pscan(dA, dBx)                                  # (N,C,S,L)
        return (h * C[:, None, :, :]).sum(dim=2)            # (N,C,L)

    def forward(self, u):
        n, c, L = u.shape
        h = self.norm(u)
        x, z = self.in_proj(h).chunk(2, dim=1)
        x = F.silu(self.conv(x)[..., :L])
        p = self.x_proj(x)
        dt = F.softplus(p[:, :1] + self.dt_bias[None, :, None])  # (N,C,L)
        B, C = p[:, 1:1 + self.d_state], p[:, 1 + self.d_state:]

        y = self.scan(x, dt, B, C)
        # Reversed pass: the task is non-causal, so context must flow both ways.
        rev = lambda t: torch.flip(t, dims=[-1])
        y = y + rev(self.scan(rev(x), rev(dt), rev(B), rev(C)))
        y = y + self.D[None, :, None] * x
        return u + self.out_proj(y * F.silu(z))


class SSMCNN(DilatedCNN):
    """The published network with one selective-SSM layer before the head."""

    def __init__(self, d_state=8, **kw):
        super().__init__(**kw)
        self.ssm = MambaBlock(kw.get("width", 64), d_state=d_state)

    def forward(self, x):
        h = self.stem(x)
        for block in self.blocks:
            h = h + block(h)
        return self.head(self.ssm(h)).squeeze(1)


def test_scan():
    """Parallel scan must match a sequential reference."""
    torch.manual_seed(0)
    a, b = torch.rand(2, 3, 64) * 0.9 + 0.05, torch.randn(2, 3, 64)
    ref = torch.zeros_like(b)
    h = torch.zeros(2, 3)
    for t in range(64):
        h = a[..., t] * h + b[..., t]
        ref[..., t] = h
    err = (pscan(a, b) - ref).abs().max().item()
    assert err < 1e-4, f"scan mismatch {err}"
    return err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pops", default="CHB,CDX")
    ap.add_argument("--vcf", default="data/chr22.vcf.gz")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    print(f"parallel scan vs sequential reference: max err {test_scan():.2e}", flush=True)

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
        from lai.sim import make_admixed
        pos = positions[sl] - positions[sl][0]
        cfg = RealConfig(seq_length=float(pos[-1]), n_admixed=64)
        adm, lab = make_admixed(cfg, da[sl], db[sl], pos, np.random.default_rng(seed))
        return {"ref_a": ra[sl], "ref_b": rb[sl], "admixed": adm,
                "labels": lab, "positions": pos}

    seg_tr, seg_te = seg(slice(0, cut), 1), seg(slice(cut + buf, n), 2)
    # Identical draws to run_attn.py, so the two sweeps share a baseline.
    xtr, ytr = windows(seg_tr, 18, np.random.default_rng(7))
    xte, yte = windows(seg_te, 6, np.random.default_rng(8))
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"train {tuple(xtr.shape)} test {tuple(xte.shape)} on {device}", flush=True)

    CONFIGS = [("none", None), ("mamba-lite", 1), ("mamba", 8)]
    rows = []
    for name, d_state in CONFIGS:
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            model = (DilatedCNN() if d_state is None
                     else SSMCNN(d_state=d_state)).to(device)
            try:
                model = train(model, xtr, ytr, device, args.epochs, 1e-3, args.batch)
                acc = eval_cnn(model, xte, yte, device)
            except RuntimeError as exc:
                print(f"  {name:<12} seed {seed}: FAILED ({exc})", flush=True)
                continue
            params = sum(p.numel() for p in model.parameters())
            rows.append({"config": name, "seed": seed, "acc": acc, "params": params,
                         "d_state": d_state, "fst": fst, "pops": pops,
                         "epochs": args.epochs})
            print(f"  {name:<12} seed {seed}: acc={acc:.4f} params={params/1000:.0f}k",
                  flush=True)
            OUT.mkdir(parents=True, exist_ok=True)
            (OUT / f"ssm_{'_'.join(pops)}.json").write_text(json.dumps(rows, indent=2))

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

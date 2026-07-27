"""GroupNorm vs batch normalisation, as an ablation rather than an argument.

The study uses GroupNorm; comparable convolutional LAI architectures use batch
normalisation. The Methods give a reason to prefer per-example normalisation
here -- a batch is a set of haplotypes from one genomic window, so its members
share the local LD background and, through the admixture process, correlated
ancestry, meaning batch statistics are not independent of the label being
predicted. This measures whether that reasoning has any consequence for
accuracy.

Two quantities are reported per configuration:

    acc         held-out accuracy in eval mode (the reported setting)
    acc_batch   held-out accuracy with normalisation using *batch* statistics
                instead of running ones

Their difference is the train/evaluation discrepancy. For GroupNorm it must be
exactly zero, since GroupNorm normalises each example independently and has no
running statistics -- which makes it a control on the measurement itself. For
batch normalisation it is whatever the batch coupling costs, measured rather
than asserted.

Seeding happens before construction and the norm layers are swapped
afterwards. Both GroupNorm and BatchNorm initialise to constant weight and
bias and consume no RNG, so the convolution weights are bit-identical across
the two arms: the configurations differ in normalisation and nothing else.
Three seeds, identical windows, paired on seed as elsewhere.
"""

import argparse
import json

import numpy as np
import torch
import torch.nn as nn

from lai.methods import DilatedCNN
from lai.real import load_region, split_panel
from lai.sim import hudson_fst, make_admixed
from run_attn import OUT, train, windows
from run_pilot import eval_cnn
from run_real import RealConfig


def to_batchnorm(module):
    """Replace every GroupNorm with a BatchNorm1d over the same channels."""
    for name, child in module.named_children():
        if isinstance(child, nn.GroupNorm):
            setattr(module, name, nn.BatchNorm1d(child.num_channels))
        else:
            to_batchnorm(child)
    return module


def eval_batch_stats(model, x, y, device, batch=32):
    """Accuracy with normalisation computed from the evaluation batch.

    train() mode makes BatchNorm use the statistics of whatever haplotypes
    happen to share the batch. Gradients are not needed, so this measures only
    the prediction shift, not any parameter change.
    """
    model.train()
    correct = tot = 0
    with torch.no_grad():
        for i in range(0, x.shape[0], batch):
            xb = x[i:i + batch].to(device)
            pred = (torch.sigmoid(model(xb)) > 0.5).float().cpu()
            correct += (pred == y[i:i + batch]).sum().item()
            tot += y[i:i + batch].numel()
    return correct / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pops", default="CHB,CDX")
    ap.add_argument("--vcf", default="data/chr22.vcf.gz")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
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
    # Identical draws to run_attn.py / run_ssm.py, so baselines must reproduce.
    xtr, ytr = windows(seg_tr, 18, np.random.default_rng(7))
    xte, yte = windows(seg_te, 6, np.random.default_rng(8))
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"train {tuple(xtr.shape)} test {tuple(xte.shape)} on {device}", flush=True)

    rows = []
    for name in ("groupnorm", "batchnorm"):
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            model = DilatedCNN()
            if name == "batchnorm":
                model = to_batchnorm(model)
            model = model.to(device)
            model = train(model, xtr, ytr, device, args.epochs, 1e-3, args.batch)
            acc = eval_cnn(model, xte, yte, device)
            accb = eval_batch_stats(model, xte, yte, device, args.batch)
            rows.append({"config": name, "seed": seed, "acc": acc,
                         "acc_batch_stats": accb, "discrepancy": acc - accb,
                         "params": sum(p.numel() for p in model.parameters()),
                         "fst": fst, "pops": pops, "epochs": args.epochs})
            print(f"  {name:<10} seed {seed}: acc={acc:.4f}  "
                  f"batch-stat acc={accb:.4f}  discrepancy={acc - accb:+.4f}",
                  flush=True)
            OUT.mkdir(parents=True, exist_ok=True)
            (OUT / f"norm_{'_'.join(pops)}.json").write_text(json.dumps(rows, indent=2))

    print("\n--- paired on seed, batchnorm vs groupnorm ---", flush=True)
    base = {r["seed"]: r["acc"] for r in rows if r["config"] == "groupnorm"}
    for name in ("groupnorm", "batchnorm"):
        sel = [r for r in rows if r["config"] == name]
        acc = [r["acc"] for r in sel]
        disc = [r["discrepancy"] for r in sel]
        d = [r["acc"] - base[r["seed"]] for r in sel if r["seed"] in base]
        print(f"  {name:<10} acc {np.mean(acc):.4f} +/- {np.std(acc):.4f}   "
              f"paired delta {np.mean(d):+.4f}   "
              f"train/eval discrepancy {np.mean(disc):+.4f}", flush=True)


if __name__ == "__main__":
    main()

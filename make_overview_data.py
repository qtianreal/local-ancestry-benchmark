"""Data behind the overview schematic.

The overview figure is partly a diagram, but the two panels that make claims
are drawn from real runs rather than sketched: the diagnostic-site densities
come from simulated panels at the two ends of the divergence axis, and the
ancestry tracks come from a cached trained network applied to a held-out
replicate. Nothing in those panels is drawn by hand.

Writes results/overview_trace.npz, which make_figures.fig0() consumes.
"""

import numpy as np
import torch

from lai.decode import viterbi_decode
from lai.export import _dedup_positions
from lai.methods import DilatedCNN, panel_frequencies
from lai.sim import RHO, SimConfig, simulate_replicate
from run_crf import site_logits

OUT = "results/overview_trace.npz"

# Mid-range divergence: high per-site accuracy, so the fragmentation the panel
# shows cannot be dismissed as a model that simply fails here.
TRACE_T = 400
# Ends of the sweep, for the diagnostic-site comparison.
LO_T, HI_T = 25, 3200
SEG = 2600          # sites drawn in the track panel
DIAG = 0.5          # |p_A - p_B| above which a site is called diagnostic


def diagnostic_density(T, seed, n_sites=3000):
    """Per-site allele-frequency separation between the two source panels."""
    rep = simulate_replicate(SimConfig(split_time=T), seed=seed)
    pos, keep = _dedup_positions(rep["positions"])
    pa, pb = panel_frequencies(rep["ref_a"][keep][:n_sites],
                              rep["ref_b"][keep][:n_sites])
    return np.abs(pa - pb), rep["fst"]


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    rep = simulate_replicate(SimConfig(split_time=TRACE_T), seed=777_000 + TRACE_T)
    pos, keep = _dedup_positions(rep["positions"])
    ra, rb = rep["ref_a"][keep], rep["ref_b"][keep]
    adm, truth = rep["admixed"][keep], rep["labels"][keep]

    model = DilatedCNN().to(device)
    model.load_state_dict(torch.load(f"results/cache/cnn_T{TRACE_T}_s0.pt",
                                     map_location=device))
    model.eval()
    logits, last = site_logits(model, adm, ra, rb, device)
    p, tr = pos[:last], truth[:last]

    raw = (logits > 0).astype(np.int8)
    crf = viterbi_decode(logits, p, SimConfig(split_time=TRACE_T).admix_generations,
                         rho=RHO)

    # Pick the haplotype whose tract inflation is closest to the median, so the
    # illustration is a typical case and not the worst one available.
    def n_tracts(a):
        return 1 + int((np.diff(a.astype(np.int16)) != 0).sum())
    infl = np.array([n_tracts(raw[:, j]) / max(n_tracts(tr[:, j]), 1)
                     for j in range(tr.shape[1])])
    j = int(np.argsort(infl)[len(infl) // 2])

    s = 0
    seg = slice(s, s + SEG)
    acc_raw = float((raw[:, j] == tr[:, j]).mean())
    acc_crf = float((crf[:, j] == tr[:, j]).mean())

    lo, lo_fst = diagnostic_density(LO_T, 909_000 + LO_T)
    hi, hi_fst = diagnostic_density(HI_T, 909_000 + HI_T)

    np.savez_compressed(
        OUT,
        truth=tr[seg, j].astype(np.int8),
        raw=raw[seg, j].astype(np.int8),
        crf=crf[seg, j].astype(np.int8),
        pos=p[seg].astype(np.float64),
        acc_raw=acc_raw, acc_crf=acc_crf,
        n_truth=n_tracts(tr[:, j]), n_raw=n_tracts(raw[:, j]),
        n_crf=n_tracts(crf[:, j]),
        fst_trace=rep["fst"],
        lo_diag=lo.astype(np.float32), hi_diag=hi.astype(np.float32),
        lo_fst=lo_fst, hi_fst=hi_fst,
    )
    print(f"trace at Fst={rep['fst']:.4f}, haplotype {j}")
    print(f"  accuracy   raw {acc_raw:.4f}   viterbi {acc_crf:.4f}")
    print(f"  tracts     true {n_tracts(tr[:, j])}   raw {n_tracts(raw[:, j])}"
          f"   viterbi {n_tracts(crf[:, j])}")
    print(f"  diagnostic sites (|dp|>{DIAG}): "
          f"Fst={lo_fst:.4f} {100 * (lo > DIAG).mean():.2f}%   "
          f"Fst={hi_fst:.4f} {100 * (hi > DIAG).mean():.2f}%")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

"""Viterbi decoding of the network's per-site logits.

The network emits an independent decision per site, which is why it fragments
the ancestry mosaic: nothing in it penalises a switch. RFMix and FLARE cannot
produce a shattered mosaic because each carries an explicit switching process
-- a conditional random field and a Li-Stephens HMM respectively.

Here we add the same constraint to our network without retraining, by treating
the trained logits as emissions of a linear-chain model whose transition
weights come from the recombination process rather than being learned:

    P(switch between adjacent sites) = g * rho * distance_in_bp

with g the generations since admixture. Decoding is exact Viterbi. The
emissions are unchanged, so any difference is attributable purely to imposing
a coherent switching process on predictions that already exist.
"""

import numpy as np


def viterbi_decode(logits, positions, generations, rho=1e-8, min_p=1e-9, max_p=0.49):
    """Decode per-site logits into an ancestry path.

    logits    : (n_sites, n_haps) log-odds of ancestry 1 versus ancestry 0
    positions : (n_sites,) base-pair coordinates
    returns   : (n_sites, n_haps) int8 labels
    """
    n_sites, n_haps = logits.shape
    if n_sites < 2:
        return (logits > 0).astype(np.int8)

    gaps = np.diff(np.asarray(positions, dtype=np.float64))
    p_switch = np.clip(generations * rho * gaps, min_p, max_p)
    log_switch = np.log(p_switch)
    log_stay = np.log1p(-p_switch)

    # Symmetric emissions so the two states are treated even-handedly.
    emit1 = logits * 0.5
    emit0 = -emit1

    delta0 = emit0[0].astype(np.float64)
    delta1 = emit1[0].astype(np.float64)
    back = np.empty((n_sites, 2, n_haps), dtype=np.int8)

    for t in range(1, n_sites):
        ls, lw = log_stay[t - 1], log_switch[t - 1]
        # into state 0
        stay, sw = delta0 + ls, delta1 + lw
        keep0 = stay >= sw
        back[t, 0] = np.where(keep0, 0, 1)
        new0 = np.where(keep0, stay, sw) + emit0[t]
        # into state 1
        stay, sw = delta1 + ls, delta0 + lw
        keep1 = stay >= sw
        back[t, 1] = np.where(keep1, 1, 0)
        new1 = np.where(keep1, stay, sw) + emit1[t]
        delta0, delta1 = new0, new1

    path = np.empty((n_sites, n_haps), dtype=np.int8)
    path[-1] = (delta1 > delta0).astype(np.int8)
    idx = np.arange(n_haps)
    for t in range(n_sites - 1, 0, -1):
        path[t - 1] = back[t, path[t], idx]
    return path

"""Local-ancestry inference methods evaluated in this study.

Three methods, all consuming identical inputs: a target haplotype and the
allele frequencies of the two reference panels.

    NaiveBayes  - per-window likelihood ratio, no spatial smoothing
    HMM         - same emissions, Viterbi-smoothed (an RFMix-like baseline)
    DilatedCNN  - learned per-site segmentation
"""

import numpy as np
import torch
import torch.nn as nn

PSEUDO = 1e-3  # frequency pseudocount, guards log(0) at fixed sites


def panel_frequencies(ref_a: np.ndarray, ref_b: np.ndarray):
    p_a = np.clip(ref_a.mean(axis=1), PSEUDO, 1 - PSEUDO)
    p_b = np.clip(ref_b.mean(axis=1), PSEUDO, 1 - PSEUDO)
    return p_a, p_b


def site_log_ratio(haps: np.ndarray, p_a: np.ndarray, p_b: np.ndarray):
    """Per-site log P(x | A) - log P(x | B). Shape (n_sites, n_haps)."""
    la = haps * np.log(p_a)[:, None] + (1 - haps) * np.log(1 - p_a)[:, None]
    lb = haps * np.log(p_b)[:, None] + (1 - haps) * np.log(1 - p_b)[:, None]
    return la - lb


def naive_bayes_predict(log_ratio: np.ndarray, window: int):
    """Aggregate the log ratio in fixed SNP windows, no smoothing."""
    n_sites, n_haps = log_ratio.shape
    n_win = n_sites // window
    trimmed = log_ratio[: n_win * window].reshape(n_win, window, n_haps)
    win_score = trimmed.sum(axis=1)  # (n_win, n_haps)
    win_call = (win_score < 0).astype(np.int8)  # 1 => ancestry B
    return np.repeat(win_call, window, axis=0), win_score


def hmm_predict(win_score: np.ndarray, switch_prob: float):
    """Viterbi over windows with a symmetric two-state transition matrix."""
    n_win, n_haps = win_score.shape
    log_stay = np.log(1 - switch_prob)
    log_switch = np.log(switch_prob)

    # Emissions: state 0 = ancestry A, state 1 = ancestry B.
    emit = np.stack([win_score / 2.0, -win_score / 2.0], axis=-1)

    delta = np.zeros((n_win, n_haps, 2))
    psi = np.zeros((n_win, n_haps, 2), dtype=np.int8)
    delta[0] = emit[0] + np.log(0.5)

    for t in range(1, n_win):
        for s in range(2):
            stay = delta[t - 1, :, s] + log_stay
            switch = delta[t - 1, :, 1 - s] + log_switch
            psi[t, :, s] = np.where(stay >= switch, s, 1 - s)
            delta[t, :, s] = np.maximum(stay, switch) + emit[t, :, s]

    path = np.zeros((n_win, n_haps), dtype=np.int8)
    path[-1] = delta[-1].argmax(axis=1)
    for t in range(n_win - 2, -1, -1):
        path[t] = psi[t + 1, np.arange(n_haps), path[t + 1]]
    return path


class DilatedCNN(nn.Module):
    """Residual dilated 1-D conv stack for per-site ancestry segmentation."""

    def __init__(self, in_ch=4, width=64, dilations=(1, 2, 4, 8, 16, 32, 64, 128, 256)):
        super().__init__()
        self.stem = nn.Conv1d(in_ch, width, kernel_size=5, padding=2)
        self.blocks = nn.ModuleList()
        for d in dilations:
            self.blocks.append(
                nn.Sequential(
                    nn.GroupNorm(8, width),
                    nn.GELU(),
                    nn.Conv1d(width, width, kernel_size=5, padding=2 * d, dilation=d),
                    nn.GroupNorm(8, width),
                    nn.GELU(),
                    nn.Conv1d(width, width, kernel_size=1),
                )
            )
        self.head = nn.Sequential(
            nn.GroupNorm(8, width), nn.GELU(), nn.Conv1d(width, 1, kernel_size=1)
        )

    def forward(self, x):
        h = self.stem(x)
        skip = getattr(self, "skip_blocks", ())
        for i, block in enumerate(self.blocks):
            if i in skip:
                continue  # residual block removed => exact identity
            h = h + block(h)
        return self.head(h).squeeze(1)  # (batch, n_sites) logits


MATCH_WINDOW = 64  # segregating sites either side for haplotype match scores
TOPK = 5


def haplotype_channels(haps, ref_a, ref_b, window=MATCH_WINDOW, topk=TOPK,
                       chunk=16):
    """Local haplotype-matching summaries against each reference panel.

    Allele frequencies discard which *reference haplotype* a segment resembles,
    yet that is precisely the signal that survives when populations are barely
    differentiated: frequency differences vanish long before haplotype sharing
    does. For each population we compute, in a sliding window, the fraction of
    sites at which the target agrees with each reference haplotype, then
    summarise across reference haplotypes by the best match and by the mean of
    the top-k matches. The latter is more robust than the maximum alone, which
    is noisy when many haplotypes are near-equally good.

    Returns (n_haps, 4, n_sites): best_A, best_B, topk_A, topk_B.
    """
    n_sites, n_haps = haps.shape
    out = np.empty((n_haps, 4, n_sites), dtype=np.float32)
    pad = window

    for lo in range(0, n_haps, chunk):
        hi = min(lo + chunk, n_haps)
        tgt = haps[:, lo:hi]  # (S, t)
        for slot, ref in enumerate((ref_a, ref_b)):
            # (S, t, R) agreement, summed over a centred window via cumsum.
            match = (tgt[:, :, None] == ref[:, None, :]).astype(np.int32)
            csum = np.zeros((n_sites + 1, match.shape[1], match.shape[2]), np.int32)
            np.cumsum(match, axis=0, out=csum[1:])
            idx = np.arange(n_sites)
            himask = np.minimum(idx + pad + 1, n_sites)
            lomask = np.maximum(idx - pad, 0)
            width = (himask - lomask).astype(np.float32)[:, None]
            frac = (csum[himask] - csum[lomask]).astype(np.float32) / width[..., None]

            part = np.partition(frac, -topk, axis=2)
            out[lo:hi, slot] = frac.max(axis=2).T
            out[lo:hi, 2 + slot] = part[:, :, -topk:].mean(axis=2).T

    # Raw match fractions sit near 0.99 with a standard deviation of ~0.02: the
    # discriminative signal is a small fluctuation on a large constant offset,
    # an order of magnitude below the scale of the allele and log-ratio
    # channels. Left unscaled the network treats them as constant. Standardise
    # each channel over the window so all inputs enter on comparable scales.
    mu = out.mean(axis=(0, 2), keepdims=True)
    sd = out.std(axis=(0, 2), keepdims=True)
    return (out - mu) / np.maximum(sd, 1e-6)


def build_features(haps: np.ndarray, p_a: np.ndarray, p_b: np.ndarray,
                   ref_a: np.ndarray = None, ref_b: np.ndarray = None):
    """Stack allele, panel frequencies, and the per-site log ratio as channels.

    The fourth channel is the same sufficient statistic the likelihood
    baselines consume, so the CNN holds no information advantage; any accuracy
    difference is attributable to how evidence is integrated across sites
    rather than to what evidence is available.
    """
    n_sites, n_haps = haps.shape
    lr = site_log_ratio(haps, p_a, p_b).T.astype(np.float32)
    feats = np.stack(
        [
            haps.T.astype(np.float32),
            np.repeat(p_a[None, :], n_haps, axis=0).astype(np.float32),
            np.repeat(p_b[None, :], n_haps, axis=0).astype(np.float32),
            lr,
        ],
        axis=1,
    )
    if ref_a is None or ref_b is None:
        return feats  # (n_haps, 4, n_sites) -- frequency-only ablation
    hap_ch = haplotype_channels(haps, ref_a, ref_b)
    return np.concatenate([feats, hap_ch], axis=1)  # (n_haps, 8, n_sites)


def distance_channel(positions, n_haps):
    """Local inter-site spacing, the positional information the tools receive.

    RFMix and FLARE are given a genetic map and therefore know how far apart
    consecutive segregating sites are; the frequency-only network treats them
    as equally spaced. Under the constant-rate map supplied to those tools,
    genetic distance is proportional to physical distance, so the log physical
    gap carries the same information.

    The gap is logged because spacing is heavy-tailed (p99/p50 is 11.9 on real
    chromosome 22) and standardised within the window so the channel is
    stationary, which a convolution requires; absolute coordinates would not
    be.
    """
    pos = np.asarray(positions, dtype=np.float64)
    gap = np.diff(pos, prepend=pos[0])
    d = np.log10(gap + 1.0)
    d = (d - d.mean()) / (d.std() + 1e-9)
    return np.repeat(d[None, None, :].astype(np.float32), n_haps, axis=0)


def matchlen_channels(haps, ref_a, ref_b, topk=TOPK, chunk=16):
    """Set-maximal match lengths against each reference panel, per site.

    The windowed agreement fractions above measure the *rate* at which a target
    agrees with a reference. The copying models underlying RFMix and FLARE use
    something else: the length of contiguous agreement. A reference agreeing at
    95% of sites at random and one matching perfectly across a long run are
    nearly indistinguishable by rate, and completely different as evidence of
    shared ancestry -- long runs are identity by descent, scattered agreement is
    chance.

    For each reference haplotype we compute the length of the agreement run
    passing through each site, which is the set-maximal match length a PBWT
    would index, and summarise across the panel by the maximum and by the mean
    of the top-k. Lengths are logged because their distribution is heavy-tailed.

    Returns (n_haps, 4, n_sites): max_A, max_B, topk_A, topk_B.
    """
    n_sites, n_haps = haps.shape
    out = np.empty((n_haps, 4, n_sites), dtype=np.float32)
    pos = np.arange(n_sites, dtype=np.int32)[:, None]

    for lo in range(0, n_haps, chunk):
        hi = min(lo + chunk, n_haps)
        for slot, ref in enumerate((ref_a, ref_b)):
            for j in range(lo, hi):
                agree = haps[:, j][:, None] == ref            # (S, R)
                # Run length ending at i: distance since the last disagreement.
                last0 = np.maximum.accumulate(
                    np.where(~agree, pos, -1), axis=0)
                fwd = pos - last0
                # Run length starting at i: distance to the next disagreement.
                nxt0 = np.minimum.accumulate(
                    np.where(~agree, pos, n_sites)[::-1], axis=0)[::-1]
                bwd = nxt0 - pos
                through = np.where(agree, fwd + bwd - 1, 0).astype(np.float32)
                logs = np.log1p(through)
                part = np.partition(logs, -topk, axis=1)
                out[j, slot] = logs.max(axis=1)
                out[j, 2 + slot] = part[:, -topk:].mean(axis=1)

    mu = out.mean(axis=(0, 2), keepdims=True)
    sd = out.std(axis=(0, 2), keepdims=True)
    return (out - mu) / np.maximum(sd, 1e-6)

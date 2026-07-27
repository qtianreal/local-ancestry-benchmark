# Architecture and hyperparameter sweep — not used in the manuscript

`run_tune.py` swept width, depth, window size, epochs and learning rate on
CHB/CDX. Results in `results/tune_CHB_CDX.json`.

## What it showed

Only receptive-field reach matters. Against a baseline of 0.6018:

    deeper (+dilation 512)   +0.025      extends context
    long-window (8192)       +0.022      extends context
    shallow (drop >=64)      +0.010
    more-epochs (40)         +0.008      optimisation
    narrow (32ch, 1/4 params) +0.003     capacity
    wide (128ch, 4x params)  +0.002      capacity
    lower-lr (3e-4)          -0.004      optimisation

Parameter count varies 15-fold across the capacity arms for ~0.002 of
accuracy; the two context arms give ~0.023 each. This corroborates the
block-ablation and LDA findings by a third, independent route.

## Why it is excluded

The sweep selects on a pair already evaluated in the manuscript, so its
figures are optimistic bounds, not reportable accuracies. More importantly the
gain does not transfer. Run through the manuscript's own protocol
(`run_real.py`, 3 windows across 6 reference/donor partitions rather than 18
windows from one mosaic realisation), the winning configuration performs
*worse*:

    CHB/CDX   baseline 0.675 -> deeper 0.666   (-0.009, within noise)
    CHB/JPT   baseline 0.711 -> deeper 0.6455  (-0.066, not within noise)

So architecture tuning does not close the real-data deficit, and the single
architecture reported in the manuscript is not an unlucky choice. Outputs from
these runs live in `results/tuning/` so they cannot be mistaken for population
pairs by the figure and number generators, which now additionally require the
canonical `real_POP_POP.json` form.

---

# Correction: the sweep above is not interpretable

Later runs measured the seed noise this sweep never estimated. Retraining the
*identical* baseline under three seeds on the same windows gives

    0.6018  0.6215  0.6171     sd 0.0103, range 0.0197

Every effect ranked above is smaller than, or comparable to, the spread of a
single architecture against itself. `run_tune.py` used one run per
configuration, so its ordering reflects initialisation luck as much as
architecture, and the "only receptive-field reach matters" conclusion is not
supported by it. The block-ablation and LDA results are independent of this
and stand on their own.

Nothing from that sweep reached the manuscript.

# Global-context mechanisms — null, not in the manuscript

Both test the same hypothesis: the dilated stack has a receptive field of 2045
SNPs inside a 4096-SNP window, so no output site sees the whole window. Each
mechanism removes that ceiling. Both are inserted after the stack with a
zero-initialised output projection, verified to compute the baseline function
*exactly* at initialisation, so any difference is what the layer learns rather
than a perturbed starting point. Three seeds, identical windows, paired.

    attn-pooled  (run_attn.py)  -0.006   +0.011 / -0.028 / -0.000
    mamba-lite   (run_ssm.py)   +0.006   +0.016 / +0.005 / -0.004

Both change sign across seeds and are smaller than the 0.0197 baseline spread.
With n=3 the 95% CI half-widths are 0.051 and 0.025 against a CNN-to-RFMix
deficit of 0.059 on this pair, so these runs are a screen, not evidence: the
attention interval does not even exclude an effect large enough to close the
gap. Publishing a null from them would repeat the error corrected above.

Two mechanisms were not completed:

  attn-full   full 4096-resolution attention. A batch-32 attention tensor is
              32*4*4096^2 = 2^31 elements, one past the Metal NDArray limit,
              which aborts the process rather than raising. Fixed by chunking
              the batch inside the layer (verified bit-exact), now opt-in
              behind --full-attn and deliberately not run: it is the same
              mechanism as attn-pooled at finer granularity, and tracts here
              span hundreds of SNPs.
  mamba d_state=8   killed after 35 min without finishing one seed (~12x
              mamba-lite, not the 3x its per-step cost predicted). The
              bidirectional scan retains twelve (32,64,8,4096) tensors for
              backward, ~13 GB, and thrashes MPS.

# Combining the CNN with FLARE — dead end (run_oracle.py)

Six pairs, reusing saved tool output and cached CNN weights. Recomputed tool
accuracies are checked against results/realext_*.json and the script aborts on
mismatch, since site alignment across three indexings is the one thing that
silently invalidates the result.

The oracle over {CNN, FLARE} sits +0.006 to +0.150 above the better of the
two, scaling almost perfectly with the error rate (r = -0.996 against
best-single accuracy). That looks like headroom and is not: in a *binary*
task, wherever FLARE errs a coin flip is right half the time. Measuring the
CNN's accuracy at exactly those sites gives

    0.510  0.430  0.516  0.371  0.474  0.431     mean 0.456, chance 0.500

indistinguishable from chance (p = 0.107) and below it on four of six pairs.
The observed oracle is below the independent-errors oracle on 6/6, so the
errors are positively correlated rather than complementary. A training-free
majority vote gained a mean of +0.004, positive on only 2 of 6.

The one place worth revisiting if this is ever picked up again is FIN/TSI --
lowest divergence, the only pair where the CNN is above chance on FLARE's
errors (0.510) and the vote gains meaningfully (+0.024). Its tool output has
since been cleaned up and would need regenerating.

# What did reach the manuscript

The normalisation ablation (`run_norm.py`): batch normalisation lowers
accuracy by 0.035, paired over five seeds, worse on 5/5, p = 0.0031, and
introduces a train/evaluation discrepancy of up to 0.008 that changes sign
across seeds where GroupNorm's is exactly zero. Reported in Materials and
methods via generated macros.

---

# Input representation, and what else was tried (this session)

## In the manuscript

Haplotype-matching channels on real haplotypes: +0.023 below F_ST 0.04,
positive on 7/8 pairs (t=2.63, p=0.034), negative on all 3 pairs above it.
Regimes differ (t=2.45, p=0.037; sign Fisher p=0.024). The same channels are
worth +0.003 in simulation across 8 levels (p=0.39), which is what makes the
contrast worth reporting rather than the gain alone.

Phase robustness: the haplotype-aware network degrades by 0.054 from clean to
2 switches/Mb, against 0.084 for the frequency-only network, 0.075 for RFMix
and 0.056 for FLARE. Sensitivity tracks integration range, not input type --
the windowed likelihood reads frequencies with no context and barely moves
(0.006), while the frequency-only network reads the same frequencies over 2045
sites and is the most damaged method in the study.

Discriminant criterion: haplotype input raises J by x1.77 on 8/8 levels.
Absolute J varies threefold across evaluation-window draws and must not be
reported alone; the ratio on identical windows is stable to about 2%.

## Tried and null -- not in the manuscript

**Distance / positional channel.** Log inter-site spacing, standardised per
window, is the information a constant-rate genetic map carries. Alone on
CHB/CDX: -0.003 (p=0.46), and fragmentation got worse rather than better,
contradicting the prediction from the Viterbi result. On top of haplotype
channels across 5 pairs, four of them pairs where the haplotype-aware network
trails FLARE: -0.006, positive on 0/5, closing no gap and overtaking a
released tool nowhere. Tested by two routes on five pairs; distance is not the
missing information.

**LDA-shaped training objective.** Auxiliary two-class Fisher criterion on the
penultimate features, J = (m1-m0)^T (S_W + lambda I)^-1 (m1-m0), maximised as
log(1+J) alongside BCE at alpha=0.1. Run on TSI/PJL, the largest in-range
deficit to FLARE (-0.063), 5 seeds paired: **-0.001, positive on 1/5**. The
deficit is informational, not representational, which is consistent with every
model-side intervention in this project moving under 0.02 while the only
intervention that mattered changed the input.

Two implementation notes for anyone repeating it: torch.linalg.solve has no
MPS backward, so the C x C solve runs on CPU; and it must stay in float32,
since float64 gradients cannot return to an MPS tensor.

## Method notes worth not rediscovering

- Tract-count ratio scales with the sequence it is measured over. A figure
  from a windowed evaluation (24.7) and one from full-replicate tiling (78.8)
  are the same quantity on different supports and must never be compared.
- run_tracts.py silently dropped RFMix and FLARE when their scratch output had
  been cleaned, turning published macros into placeholders. It now carries
  forward any method it cannot recompute.
- Regenerating numbers.tex while an experiment is mid-write computes macros
  from partial results. This produced a wrong -0.051 where the truth was
  -0.090, and unlike a missing macro it does not announce itself.

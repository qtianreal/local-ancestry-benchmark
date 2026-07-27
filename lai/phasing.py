"""Switch-error injection, to relax the perfect-phasing assumption.

Applied LAI runs on statistically phased haplotypes, not true ones. A switch
error at position k exchanges the two haplotypes of a diploid individual from
k onward. This has an informative asymmetry:

  reference panels  swapping alleles between the two haplotypes of the same
                    individual leaves the allele-frequency spectrum exactly
                    unchanged, so frequency-based methods are unaffected by
                    construction, while reference haplotypes become chimeric
                    and haplotype-matching degrades.

  admixed targets   the observed haplotype acquires a spurious ancestry
                    transition wherever a switch occurs. Ground-truth labels
                    are swapped identically, so the labels remain correct for
                    the *observed* haplotypes; the task simply acquires
                    breakpoints that no recombination produced.

Haplotypes are paired as individuals (2i, 2i+1), matching how they are written
to VCF for the external tools.
"""

import numpy as np


def _switch_points(n_ind, length, rate_per_mb, rng):
    lam = rate_per_mb * length / 1e6
    return [np.sort(rng.uniform(0, length, size=rng.poisson(lam))) for _ in range(n_ind)]


def apply_switch_errors(haps, positions, rate_per_mb, rng, labels=None):
    """Introduce phase switch errors into paired haplotypes.

    haps : (n_sites, n_haps) with n_haps even; columns 2i, 2i+1 are one
           individual. labels, if given, is swapped identically so that the
           ancestry annotation continues to describe the observed haplotypes.
    """
    if rate_per_mb <= 0:
        return haps, labels
    haps = haps.copy()
    labels = None if labels is None else labels.copy()
    n_ind = haps.shape[1] // 2
    length = float(positions[-1] - positions[0]) or 1.0
    for i, points in enumerate(_switch_points(n_ind, length, rate_per_mb, rng)):
        a, b = 2 * i, 2 * i + 1
        for pt in points:
            k = int(np.searchsorted(positions, positions[0] + pt))
            if k >= haps.shape[0]:
                continue
            haps[k:, [a, b]] = haps[k:, [b, a]]
            if labels is not None:
                labels[k:, [a, b]] = labels[k:, [b, a]]
    return haps, labels


def frequency_preserved(before, after, tol=0):
    """Sanity check: per-site allele counts must be identical after swapping."""
    return int(np.abs(before.sum(axis=1) - after.sum(axis=1)).max()) <= tol

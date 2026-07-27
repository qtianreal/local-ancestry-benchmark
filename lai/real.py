"""Load phased 1000 Genomes haplotypes as source panels.

The real-data arm reuses the mosaicking construction unchanged: only the
origin of the source haplotypes differs. Local ancestry labels therefore
remain exact, while linkage disequilibrium, the allele frequency spectrum and
haplotype diversity are empirical rather than coalescent.

Parsing note: the phased panel carries FORMAT=GT only, so for biallelic sites
every sample field is exactly three characters ("a|b") on a four-character
stride. That makes the genotype block fixed-width and lets us index it with
numpy instead of splitting several hundred thousand lines in Python.
"""

from pathlib import Path

import numpy as np
import pysam

PANEL = Path("data/kg.panel")


def population_samples(pop: str):
    """Sample IDs belonging to a 1000 Genomes population code."""
    ids = []
    with open(PANEL) as fh:
        next(fh)
        for line in fh:
            f = line.split()
            if f[1] == pop:
                ids.append(f[0])
    return ids


def load_region(vcf_path, chrom, start, end, pops, maf=0.01, max_sites=None):
    """Return phased haplotypes for the requested populations over a region.

    Sites are retained if they are biallelic SNVs, fully called, and
    polymorphic above ``maf`` in the pooled sample. Pooling the populations
    for the frequency filter avoids ascertaining sites on the very frequency
    difference the task depends on.

    Returns
    -------
    haps : dict pop -> (n_sites, 2 * n_individuals) int8
    positions : (n_sites,) float
    """
    tbx = pysam.TabixFile(str(vcf_path))
    header = [l for l in tbx.header if l.startswith("#CHROM")][-1].split("\t")
    col = {s: i for i, s in enumerate(header)}

    wanted, slices = {}, []
    offset = 0
    for pop in pops:
        ids = [s for s in population_samples(pop) if s in col]
        wanted[pop] = ids
        slices.append((pop, offset, offset + 2 * len(ids)))
        offset += 2 * len(ids)

    # Column index within the genotype block (block starts after 9 fixed cols).
    flat = [col[s] - 9 for pop in pops for s in wanted[pop]]
    # Two characters per sample: allele 1 at 4j, allele 2 at 4j+2.
    take = np.empty(2 * len(flat), dtype=np.int64)
    take[0::2] = [4 * j for j in flat]
    take[1::2] = [4 * j + 2 for j in flat]

    zero = ord("0")
    rows, pos = [], []
    for line in tbx.fetch(chrom, start, end):
        f = line.split("\t", 9)
        ref, alt = f[3], f[4]
        if len(ref) != 1 or len(alt) != 1 or "," in alt:
            continue  # biallelic SNVs only
        blob = np.frombuffer(f[9].encode("ascii"), dtype=np.uint8)
        if take[-1] >= blob.size:
            continue
        g = blob[take]
        if np.any((g != zero) & (g != zero + 1)):
            continue  # missing or non-0/1 allele coding
        g = (g - zero).astype(np.int8)
        p = g.mean()
        if p < maf or p > 1 - maf:
            continue
        rows.append(g)
        pos.append(float(f[1]))
        if max_sites and len(rows) >= max_sites:
            break

    tbx.close()
    if not rows:
        raise RuntimeError("no sites passed filtering")

    mat = np.vstack(rows)  # (n_sites, total_haplotypes)
    positions = np.asarray(pos, dtype=np.float64)
    return {pop: mat[:, a:b] for pop, a, b in slices}, positions


def split_panel(haps, n_ref, n_donor, rng, donors_from_end=False):
    """Partition a population's haplotypes into disjoint reference and donor sets.

    With donors_from_end, the donor set is taken from the end of the
    permutation instead of immediately after the reference set. That makes the
    donors -- and therefore the admixed targets built from them -- invariant to
    n_ref, which is required when reference panel size is the variable under
    test: otherwise changing n_ref silently changes the evaluation data too.
    The default preserves the original behaviour for every published result.
    """
    n = haps.shape[1]
    if n < n_ref + n_donor:
        raise ValueError(f"need {n_ref + n_donor} haplotypes, have {n}")
    perm = rng.permutation(n)
    if donors_from_end:
        return haps[:, perm[:n_ref]], haps[:, perm[n - n_donor:]]
    return haps[:, perm[:n_ref], ], haps[:, perm[n_ref : n_ref + n_donor]]

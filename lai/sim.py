"""Coalescent simulation of two-way admixture at controlled divergence.

Design notes
------------
Reference panels and admixture donors are drawn from disjoint sample sets so
that a donor haplotype never appears in the panel used to classify it.

Admixed haplotypes are built by mosaicking donor haplotypes along a Poisson
recombination process. This yields *exact* local-ancestry labels by
construction, rather than labels inferred from a tree sequence, which keeps the
ground truth independent of any modelling choice.
"""

from dataclasses import dataclass

import msprime
import numpy as np

MU = 1.25e-8  # per-base per-generation mutation rate
RHO = 1e-8  # per-base per-generation recombination rate


@dataclass
class SimConfig:
    split_time: float  # generations since A/B divergence
    ne: int = 10_000
    seq_length: float = 1e7
    n_ref: int = 100  # reference haplotypes per population
    n_donor: int = 100  # donor haplotypes per population
    n_admixed: int = 64
    admix_generations: int = 30  # generations since the admixture pulse
    admix_prop: float = 0.5  # expected ancestry fraction from population A


def simulate_panels(cfg: SimConfig, seed: int):
    """Simulate a clean split and return haplotypes for both populations."""
    n_per_pop = cfg.n_ref + cfg.n_donor

    demography = msprime.Demography()
    demography.add_population(name="ANC", initial_size=cfg.ne)
    demography.add_population(name="A", initial_size=cfg.ne)
    demography.add_population(name="B", initial_size=cfg.ne)
    demography.add_population_split(
        time=cfg.split_time, derived=["A", "B"], ancestral="ANC"
    )

    ts = msprime.sim_ancestry(
        samples={"A": n_per_pop, "B": n_per_pop},
        demography=demography,
        sequence_length=cfg.seq_length,
        recombination_rate=RHO,
        ploidy=1,
        random_seed=seed,
    )
    ts = msprime.sim_mutations(ts, rate=MU, random_seed=seed + 1)

    geno = ts.genotype_matrix().astype(np.int8)  # (n_sites, n_samples)
    positions = ts.sites_position.astype(np.float64)

    # msprime's finite-sites mutation model permits recurrent mutation, which
    # yields allele states above 1 at a small fraction of sites (0.03-0.07% at
    # these parameters). The likelihood baselines and the CNN both assume
    # biallelic 0/1 coding, and external tools reject such records outright, so
    # these sites are dropped rather than recoded.
    biallelic = (geno <= 1).all(axis=1)
    geno = geno[biallelic]
    positions = positions[biallelic]

    hap_a = geno[:, :n_per_pop]
    hap_b = geno[:, n_per_pop:]
    return hap_a, hap_b, positions


def hudson_fst(hap_a: np.ndarray, hap_b: np.ndarray) -> float:
    """Hudson's Fst estimator, computed as a ratio of averages."""
    n1, n2 = hap_a.shape[1], hap_b.shape[1]
    p1 = hap_a.mean(axis=1)
    p2 = hap_b.mean(axis=1)

    num = (p1 - p2) ** 2 - p1 * (1 - p1) / (n1 - 1) - p2 * (1 - p2) / (n2 - 1)
    den = p1 * (1 - p2) + p2 * (1 - p1)

    keep = den > 0
    return float(num[keep].sum() / den[keep].sum())


def make_admixed(cfg, donor_a, donor_b, positions, rng):
    """Mosaic donor haplotypes into admixed haplotypes with exact labels.

    Returns
    -------
    haps : (n_sites, n_admixed) int8 allele matrix
    labels : (n_sites, n_admixed) int8, 0 = ancestry A, 1 = ancestry B
    """
    n_sites = len(positions)
    length = cfg.seq_length
    # Expected crossovers accumulated since the pulse.
    rate = cfg.admix_generations * RHO * length

    haps = np.zeros((n_sites, cfg.n_admixed), dtype=np.int8)
    labels = np.zeros((n_sites, cfg.n_admixed), dtype=np.int8)

    for j in range(cfg.n_admixed):
        n_bp = rng.poisson(rate)
        breaks = np.sort(rng.uniform(0, length, size=n_bp))
        edges = np.concatenate([[0.0], breaks, [length]])

        for k in range(len(edges) - 1):
            lo, hi = edges[k], edges[k + 1]
            idx = np.searchsorted(positions, [lo, hi])
            if idx[1] <= idx[0]:
                continue
            # Each segment independently inherits from A or B.
            from_a = rng.random() < cfg.admix_prop
            pool = donor_a if from_a else donor_b
            donor = rng.integers(pool.shape[1])
            haps[idx[0]:idx[1], j] = pool[idx[0]:idx[1], donor]
            labels[idx[0]:idx[1], j] = 0 if from_a else 1

    return haps, labels


def simulate_replicate(cfg: SimConfig, seed: int):
    """One independent replicate: panels, admixed haplotypes, labels, Fst."""
    hap_a, hap_b, positions = simulate_panels(cfg, seed)

    ref_a, donor_a = hap_a[:, : cfg.n_ref], hap_a[:, cfg.n_ref :]
    ref_b, donor_b = hap_b[:, : cfg.n_ref], hap_b[:, cfg.n_ref :]

    rng = np.random.default_rng(seed + 7919)
    admixed, labels = make_admixed(cfg, donor_a, donor_b, positions, rng)

    return {
        "ref_a": ref_a,
        "ref_b": ref_b,
        "admixed": admixed,
        "labels": labels,
        "positions": positions,
        "fst": hudson_fst(ref_a, ref_b),
    }

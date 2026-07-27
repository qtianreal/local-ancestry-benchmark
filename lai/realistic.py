"""Simulation under progressively more realistic demography.

The simulation-to-reality reversal reported for the toy model leaves open
which simplification is responsible. This module varies two of them
independently so the question can be answered rather than speculated about:

  migration    continuous symmetric gene flow after the split, rather than a
               clean split with none. Real closely-related pairs -- northern
               and southern Han in particular -- are separated by a cline, not
               an isolation event.

  genetic map  an empirical recombination map in place of a constant rate.
               The PyrhoCHB map for chromosome 22 has a mean rate roughly
               three times our constant and hotspots ~50x its own mean, so
               haplotype block structure is far more heterogeneous.

Admixture breakpoints are placed on the genetic rather than the physical map
when an empirical map is used, since crossovers follow genetic distance.
"""

from dataclasses import dataclass

import msprime
import numpy as np
import stdpopsim

MU = 1.25e-8
RHO_CONST = 1e-8


@dataclass
class RealisticConfig:
    split_time: float
    migration_rate: float = 0.0        # symmetric, per generation, since split
    use_genetic_map: bool = False
    ne: int = 10_000
    left: float = 20e6
    right: float = 30e6
    n_ref: int = 80
    n_donor: int = 80
    n_admixed: int = 64
    admix_generations: int = 30
    admix_prop: float = 0.5
    map_id: str = "PyrhoCHB_GRCh38"
    chrom: str = "chr22"


_MAP_CACHE = {}


def get_rate_map(cfg):
    """Empirical recombination map trimmed to the requested window."""
    key = (cfg.map_id, cfg.chrom, cfg.left, cfg.right)
    if key not in _MAP_CACHE:
        sp = stdpopsim.get_species("HomSap")
        contig = sp.get_contig(cfg.chrom, genetic_map=cfg.map_id)
        rm = contig.recombination_map.slice(left=cfg.left, right=cfg.right, trim=True)
        # msprime rejects NaN spans; replace unmapped regions with the mean.
        rate = np.array(rm.rate, dtype=float)
        bad = np.isnan(rate)
        if bad.any():
            rate[bad] = np.nanmean(rate)
            rm = msprime.RateMap(position=rm.position, rate=rate)
        _MAP_CACHE[key] = rm
    return _MAP_CACHE[key]


def simulate_panels(cfg, seed):
    n_per_pop = cfg.n_ref + cfg.n_donor

    dem = msprime.Demography()
    dem.add_population(name="ANC", initial_size=cfg.ne)
    dem.add_population(name="A", initial_size=cfg.ne)
    dem.add_population(name="B", initial_size=cfg.ne)
    if cfg.migration_rate > 0:
        dem.set_symmetric_migration_rate(["A", "B"], cfg.migration_rate)
    dem.add_population_split(time=cfg.split_time, derived=["A", "B"], ancestral="ANC")
    if cfg.migration_rate > 0:
        # Gene flow only exists while both populations do.
        dem.add_symmetric_migration_rate_change(
            time=cfg.split_time, populations=["A", "B"], rate=0)

    if cfg.use_genetic_map:
        rate_map = get_rate_map(cfg)
        recomb, length = rate_map, rate_map.sequence_length
    else:
        length = cfg.right - cfg.left
        recomb = RHO_CONST

    ts = msprime.sim_ancestry(
        samples={"A": n_per_pop, "B": n_per_pop},
        demography=dem,
        sequence_length=None if cfg.use_genetic_map else length,
        recombination_rate=recomb,
        ploidy=1,
        random_seed=seed,
    )
    ts = msprime.sim_mutations(ts, rate=MU, random_seed=seed + 1)

    geno = ts.genotype_matrix().astype(np.int8)
    pos = ts.sites_position.astype(np.float64)
    keep = (geno <= 1).all(axis=1)          # drop recurrent-mutation sites
    return geno[keep], pos[keep], length


def make_admixed(cfg, donor_a, donor_b, positions, rng, rate_map=None):
    """Mosaic donors, placing breakpoints on the genetic map when supplied."""
    n_sites = len(positions)
    haps = np.zeros((n_sites, cfg.n_admixed), dtype=np.int8)
    labels = np.zeros((n_sites, cfg.n_admixed), dtype=np.int8)

    if rate_map is not None:
        # Crossovers are uniform in genetic distance, not physical distance.
        total_M = rate_map.get_cumulative_mass(rate_map.sequence_length)
        grid = np.linspace(0, rate_map.sequence_length, 20000)
        cum = rate_map.get_cumulative_mass(grid)
        expected = cfg.admix_generations * total_M
        to_phys = lambda m: np.interp(m, cum, grid)
    else:
        length = float(positions[-1])
        expected = cfg.admix_generations * RHO_CONST * length
        to_phys = lambda m: m * length / max(expected, 1e-12) if False else m

    for j in range(cfg.n_admixed):
        n_bp = rng.poisson(expected)
        if rate_map is not None:
            breaks = np.sort(to_phys(rng.uniform(0, total_M, size=n_bp)))
            hi_edge = rate_map.sequence_length
        else:
            breaks = np.sort(rng.uniform(0, length, size=n_bp))
            hi_edge = length
        edges = np.concatenate([[0.0], breaks, [hi_edge]])
        for k in range(len(edges) - 1):
            lo, hi = np.searchsorted(positions, [edges[k], edges[k + 1]])
            if hi <= lo:
                continue
            from_a = rng.random() < cfg.admix_prop
            pool = donor_a if from_a else donor_b
            haps[lo:hi, j] = pool[lo:hi, rng.integers(pool.shape[1])]
            labels[lo:hi, j] = 0 if from_a else 1
    return haps, labels


def simulate_replicate(cfg, seed):
    from lai.sim import hudson_fst

    geno, pos, length = simulate_panels(cfg, seed)
    n = cfg.n_ref + cfg.n_donor
    hap_a, hap_b = geno[:, :n], geno[:, n:]
    ref_a, donor_a = hap_a[:, : cfg.n_ref], hap_a[:, cfg.n_ref :]
    ref_b, donor_b = hap_b[:, : cfg.n_ref], hap_b[:, cfg.n_ref :]

    rate_map = get_rate_map(cfg) if cfg.use_genetic_map else None
    rng = np.random.default_rng(seed + 7919)
    admixed, labels = make_admixed(cfg, donor_a, donor_b, pos, rng, rate_map)

    return {"ref_a": ref_a, "ref_b": ref_b, "admixed": admixed, "labels": labels,
            "positions": pos, "fst": hudson_fst(ref_a, ref_b),
            "seq_length": length}

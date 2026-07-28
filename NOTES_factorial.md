# Realistic-demography factorial — not used in the manuscript

12 cells: {toy, +migration, +empirical genetic map, +both} x 3 split times,
all five methods, n=1 per cell. Results in `results/realistic_results.json`.

## Superseded

This single-seed factorial is superseded by `run_factorial.py`, whose results
are in `results/factorial_results.json` and are reported in the manuscript.

The control arm here disagreed with the replicated main sweep by ~17 sigma,
which is why nothing from this file was used. `run_control_check.py` traced
that to configuration rather than code: `lai/realistic.py` defaults to 80
reference haplotypes where `lai/sim.py` uses 100, and this script trained for
10x12 replicate-epochs where `run_external.py` uses 12x15. With both restored
the control gives +0.059 +/- 0.026 against the sweep's +0.048 +/- 0.006, a
separation of 0.7 sigma.

The replicated factorial then found that continuous migration reverses the
network's simulated advantage (-0.042 controlling for Fst, p<0.001) while an
empirical recombination map does not (+0.004, p=0.67).

## Original notes, kept for the record

## Why it is excluded

The control (`toy`) arm does not reproduce the replicated main sweep. At
Fst ~ 0.009 it gives a CNN-minus-best-tool gap of -0.055, where the
three-seed external replication gives +0.048 +/- 0.006 at Fst 0.010 under
nominally the same demography. That is a ~17 sigma discrepancy which I cannot
account for: the `geneticmap` arm, run with identical panel sizes and training
budget, reproduces the main sweep almost exactly (+0.048).

With n=1 per cell the between-condition means are also uninterpretable:
  toy         -0.0413 (n=2)
  migration   +0.0281 (n=3)
  geneticmap  +0.0409 (n=2)
  both        +0.0140 (n=3)
Within-condition spread (e.g. `both` ranges -0.040 to +0.059) exceeds the
differences between conditions.

## What would make it usable

Replicate every cell across >= 3 seeds (~36 runs, ~3 h) and compare paired
gaps, as the main external sweep does. If the toy arm then still disagrees
with the main sweep, the cause is a genuine difference between `lai/sim.py`
and `lai/realistic.py` and should be found before anything is concluded --
candidate differences are panel size (80 vs 100 haplotypes) and training
budget (10x12 vs 12x15).

## Why it is still worth keeping

The infrastructure is the seed of the follow-up the manuscript's Limitations
names: determining which demographic simplification causes the
simulation-to-reality reversal. `lai/realistic.py` supports continuous
migration and stdpopsim empirical recombination maps, and the calibration
established that at m = 5e-3 the two populations sit at migration-drift
equilibrium near Fst 0.005 -- structurally the right model for a cline such as
northern/southern Han, and not reachable by making a clean split recent.

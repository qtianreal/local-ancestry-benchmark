# A feasibility threshold for local ancestry inference

Code and derived results for a divergence-resolved benchmark of local ancestry
inference (LAI), comparing five methods across simulated and real haplotypes.

Every number in the manuscript is generated from the JSON files in `results/`
by `paper/make_numbers.py`, so the text cannot drift from the data.

## Findings

1. **A feasibility floor.** Below Hudson's *F*<sub>ST</sub> ≈ 0.0022 no method
   recovers usable local ancestry, in simulation or on real haplotypes; the
   best of five methods reaches 0.575 there. CHB/CHS (0.0004), IBS/TSI
   (0.0007) and GBR/CEU (0.0011) all fall below it. Supplying haplotype
   information lowers the floor somewhat but does not remove it.
2. **Per-site accuracy conceals tract fragmentation.** The learned method
   attains the highest per-site accuracy in simulation while producing 78.8x
   too many ancestry tracts, implying an admixture time 61.2x too old. Viterbi
   decoding of the *same* logits brings the tract ratio to 1.56x while
   changing per-site accuracy by 0.0002 — so the standard metric is blind to
   both the defect and its remedy. RFMix and FLARE recover tract structure
   correctly.
3. **The simulated lead does not survive real haplotypes, and the deficit is
   one of input more than architecture.** Across 11 real 1000 Genomes pairs
   the frequency-only network is beaten by a released tool on 10. Attention, a
   state-space layer, a 15-fold parameter range, a discriminant objective,
   self-supervised pretraining and more labelled data each move accuracy by at
   most 0.006, with inconsistent sign. Supplying the haplotype-matching
   information the released tools already receive recovers +0.031 on 8 of 8
   pairs below *F*<sub>ST</sub> = 0.04 — necessary but not sufficient, since
   the network still trails on 10 of 11. The pair it does win, CEU/TSI at
   *F*<sub>ST</sub> = 0.0027, is the least divergent one at which any method
   recovers anything; it wins there by 0.044.

**Which simulation shortcut is responsible.** A 36-run factorial crossing
continuous migration with an empirical recombination map shows that gene flow
reverses the learned method's simulated advantage (+0.033 under a clean split,
−0.013 under migration; −0.042 controlling for *F*<sub>ST</sub>, p < 0.001)
while the empirical map does nothing (+0.004, p = 0.67). Three seeds per cell,
so this establishes the sign and rough magnitude rather than a precise
estimate.

Two quantities usually held fixed mattered more than any architectural choice
tested. Match *contiguity* — the Li–Stephens statistic, as against the
agreement *rate* neural methods use — helps only where the panel is small
(+0.030 at 40 reference haplotypes, +0.003 at 100), substituting for panel
size rather than adding to it. And no method is near panel saturation:
80 → 100 haplotypes is worth +0.023 for the network, +0.028 for RFMix and
+0.011 for FLARE, without reordering them.

## Layout

    lai/                 simulation, methods, real-data loading, export
      sim.py             coalescent simulation + mosaic admixture (exact labels)
      methods.py         windowed likelihood, HMM, dilated CNN
      real.py            phased 1000 Genomes haplotype panels
      export.py          VCF / genetic map / sample map writers
      phasing.py         phase switch-error injection
      realistic.py       migration + empirical recombination maps

    run_pilot.py         main divergence sweep
    run_seeds.py         seed replication
    run_external.py      RFMix and FLARE on simulated data
    run_real.py          real 1000 Genomes pairs
    run_real_external.py RFMix and FLARE on real pairs
    run_tracts.py        tract-level structure
    run_dating.py        implied admixture times
    run_crf.py           Viterbi decoding of the trained logits
    run_phasing.py       phase switch-error sensitivity
    run_pruning_seeds.py block-group ablation
    run_lda.py           linear discriminant analysis of representations
    run_haplo.py         haplotype-feature ablation in simulation
    run_inputs.py        input-parity experiment on real pairs (panel sizes)
    run_matchlen.py      match contiguity versus agreement rate
    run_sharing.py       haplotype sharing, simulated versus real panels
    run_control_check.py does the realistic harness reproduce the main sweep?
    run_factorial.py     migration x recombination-map factorial (36 runs)

    interventions that were tried and did not close the deficit:
    run_attn.py          self-attention layer
    run_ssm.py           state-space layer
    run_ldatrain.py      discriminant training objective
    run_ssl.py           self-supervised pretraining (+ labelled-data control)
    run_scaling_hap.py   simulation pretraining
    run_norm.py          BatchNorm versus GroupNorm
    run_tune.py          width/depth/window sweep (excluded; see NOTES_tuning.md)
    run_realistic.py     demography factorial (excluded; see NOTES_factorial.md)

    make_figures.py            all manuscript figures
    make_overview_data.py      trace behind Figure 1B,C
    make_interventions.py      collects the runs above into one file
    make_submission_figures.py numbered TIFFs for submission

    results/*.json       derived results; the manuscript reads only these
    paper/make_numbers.py  generates every number quoted in the paper
    paper/numbers.tex      its output, one macro per reported number
    figures/             working figures; submission TIFFs are
                         regenerated by make_submission_figures.py

The manuscript source is not kept here; this repository carries the analysis
and the results it produced. What is kept is the link between them:
`paper/numbers.tex` defines every number quoted in the paper, and
`paper/make_numbers.py` regenerates it from `results/*.json`. Re-running it and
diffing the output verifies that no reported number was typed by hand.

## Reproducing

    python3 -m venv .venv
    ./.venv/bin/pip install msprime tskit numpy scipy scikit-learn pandas \
        matplotlib torch pysam stdpopsim

External tools, not vendored here:

- **RFMix v2** — <https://github.com/slowkoni/rfmix>. Autotools is not required;
  a `config.h` defining `VERSION` plus `gcc -I. -c cmdline-utils.c` and
  `g++ -I. -c *.cpp` (excluding `simulate.cpp`, which has its own `main`)
  links cleanly.
- **FLARE 0.6.0** — <https://faculty.washington.edu/browning/flare.jar>, needs a
  JDK.

Real data (425 MB, not versioned):

    curl -O https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/working/20220422_3202_phased_SNV_INDEL_SV/1kGP_high_coverage_Illumina.chr22.filtered.SNV_INDEL_SV_phased_panel.vcf.gz

Then, in dependency order:

    python run_pilot.py                 # main sweep
    python run_seeds.py --seed 1        # and --seed 2
    python run_external.py --seed 0     # and 1, 2
    python run_real.py --pops CHB,CHS   # and the other pairs
    python run_tracts.py && python run_dating.py
    python aggregate.py && python aggregate_external.py
    python make_figures.py
    python paper/make_numbers.py

Random seeds are fixed throughout; runs reproduce exactly on the same
platform. Neural network training uses the Metal Performance Shaders backend,
so results on other accelerators may differ in the last decimal place.

## Notes

`NOTES_factorial.md` records a demography factorial that is deliberately
excluded from the manuscript, and why.

## Licence

MIT.

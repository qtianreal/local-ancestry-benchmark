# Candidate titles

The paper now carries two results: a feasibility threshold below which no
method is informative, and a scoped method result — haplotype features recover
the learned method's real-data deficit below F_ST ~= 0.04 and harm it above.
The title should not promise a general method win, because the data do not
support one (3/11 pairs exceed FLARE overall, the same as frequency-only).

## Keeps the benchmark frame, adds the method

1. Local ancestry inference at low divergence: a feasibility threshold, and
   what a learned method needs to reach it
2. When does haplotype information matter? A divergence-resolved benchmark of
   local ancestry inference
3. Frequency is not enough: haplotype-aware segmentation for local ancestry
   inference at low divergence

## Leads with the method, scope stated

4. Haplotype-aware neural local ancestry inference below F_ST = 0.04
5. Recovering the learned-method deficit in low-divergence local ancestry
   inference
6. Input representation, not architecture, limits neural local ancestry
   inference at low divergence

## Notes

- (3) and (6) state the finding rather than the topic, which is the strongest
  form, and neither overclaims.
- (4) names a numeric bound estimated from eleven pairs with only three above
  the crossover; defensible in the text, thin for a title.
- (1) is safest if the simulation arm overturns the "helps on real, not in
  simulation" contrast, since it commits to nothing about the mechanism.
- Avoid anything of the form "outperforms RFMix and FLARE": true on 4 and 3
  of 8 in-range pairs respectively, which a reviewer will check.

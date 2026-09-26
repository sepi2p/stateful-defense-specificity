# Matched benign-optimization workloads: results

Date: 2026-09-25. Predictions: `docs/specificity_workloads_preregistration.md`
(frozen before any corpus result was inspected). Corpus:
`analysis_outputs/stateful_specificity_workloads_20260925/` (4,800 sessions =
400 images x 2 starts x 6 objectives; NES; 1,024 calls). Analysis:
`experiments/gate_trajectory_signatures/analyze_specificity_workloads.py` ->
`analysis/separability.csv`, `analysis/workload_utility.csv`. AUROCs are on the
200-image evaluation split; models and orientations fitted on the 100-image fit
split; 95% source-image bootstrap intervals.

## Workloads did what they are meant to (median over 400 sessions)

| start | objective | label kept | p(label) start -> end | end margin |
|---|---|---|---|---|
| denoise | attack | 0.24 | 0.97 -> 0.07 | -2.54 |
| denoise | restore | 1.00 | 0.97 -> 0.97 | 4.11 |
| denoise | confidence_boost | 1.00 | 0.97 -> 1.00 | 9.03 |
| denoise | boundary_probe | 1.00 | 0.97 -> 0.49 | 0.02 |
| denoise | counterfactual | 0.26 (target reached 0.75) | 0.97 -> 0.07 | -2.57 |
| deblur | attack | 0.37 | 1.00 -> 0.18 | -1.39 |
| deblur | boundary_probe | 1.00 | 1.00 -> 0.50 | 0.04 |
| deblur | counterfactual | 0.40 (target reached 0.60) | 1.00 -> 0.27 | -0.90 |

Native thresholds: Blacklight and GWAD+ alarm on 100% of sessions of every
objective, including the objective-free random walk.

## Predictions

**P1 (query-only <= 0.60 vs every benign NES workload, 1,024 calls): holds in
39 of 40 cells, fails marginally in 1.** GWAD+, Blacklight and their combination
give 0.45-0.59 everywhere except the combined query-only model on denoise
boundary_probe, 0.62 [0.58, 0.67]. That cell's 1,024-call prefix includes
post-success queries; on the pre-success prefix it is 0.50 [0.45, 0.55].

**P2 (query-only separates the random walk better): holds.** AUROC 1.00 vs the
random walk for GWAD+, Blacklight and combined, in both starts, while the same
detectors sit at 0.45-0.62 vs benign optimizers. Query-only detectors separate
optimization from non-optimization perfectly and benign from adversarial
optimization not at all.

**P3 (output trajectory >= 0.90 vs restore, confidence_boost): holds.**
0.98-1.00 at every prefix, including 128 calls.

**P4 (output trajectory <= 0.75 vs boundary_probe / counterfactual on the
pre-attack-success prefix): fails as preregistered, holds under a symmetric
prefix.**
- Preregistered prefix (calls before the attack's first label change):
  0.78-0.82. The leading feature is `out_label_changed` (single-feature AUROC
  about 0.70): benign probes sometimes cross the boundary before the attack does,
  while the attack's prefix by construction contains no crossing. This is an
  asymmetry of the truncation rule, not an intent signal (a detector using it
  would flag sessions for *not yet* having crossed).
- Post-hoc sensitivity analysis (not preregistered; stated as such): truncate
  each pair before *either* session's first label change and drop the
  label-change indicator. Output trajectory: boundary_probe 0.60 [0.56, 0.64]
  (denoise), 0.59 [0.56, 0.63] (deblur); counterfactual 0.62 [0.59, 0.66],
  0.60 [0.57, 0.64]. Same cut: restore 0.98-0.99, confidence_boost 1.00;
  Blacklight 0.48-0.51 throughout.

## Exact indistinguishability for comparison-based optimizers (Prop. 1)

`check_rank_equivalence.py`, SimBA (8/255 steps, 4,096 calls), margin attack vs
margin boundary probe, 40 evaluation images x 2 starts: all 80 attacks flip; in
all 80, queries *and* outputs are byte-identical up to and including the flip
query and diverge on the next one
(`analysis_outputs/stateful_specificity_workloads_20260925/rank_equivalence_simba.csv`).

## Reading

1. Query-only stateful detectors (Blacklight, GWAD, GWAD+) detect optimization,
   not intent: perfect against non-optimizing drift, chance against benign
   optimizers whatever the benign objective does to the model.
2. The intent signal lives in the output trajectory. It is decisive when the
   benign objective's effect on the model differs from the attack's (restore,
   confidence raising: about 1.0 from 128 calls).
3. When the benign objective moves the model the way the attack does (boundary
   probing, counterfactual explanation), output-aware detection falls to
   0.59-0.62 before any label change with a value-based optimizer (NES), and to
   exactly chance with a comparison-based one (SimBA, Prop. 1).

## Known gaps

- GWAD+ on the pre-success prefix is reported as "no decision" by the frozen rule
  because some attacks succeed before its about-258-call warm-up; report it on the
  eligible subset as a secondary analysis.
- One model (ResNet-18), one dataset (CIFAR-10), one value-based optimizer (NES).
  Needed: SimBA corpus, an ImageNet subset, a published output-aware detector
  (arXiv 2606.21592) reimplementation, adaptive attacker vs the output detector.

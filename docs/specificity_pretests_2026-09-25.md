# Pre-tests before the long specificity runs (2026-09-25)

Script: `experiments/gate_trajectory_signatures/pretest_detector_assumptions.py`
(outputs `analysis_outputs/stateful_specificity_pretests_20260925/`). Development
images and non-manifest test images only; no evaluation-split data touched.

| # | Test | Result | Verdict |
|---|---|---|---|
| 1 | Blacklight on 20 shuffled 1,024-image streams | 0/20 alarms; median max match 15 (threshold 25) | PASS |
| 2 | GWAD / GWAD+ on the same streams | 0 eligible windows, 0/20 alarms (same behaviour as the release on its independent stream) | PASS |
| 3a | Ljung-Box (p<0.025) on Lee-Fang-Chang's own benign construction: 20 streams of i.i.d. N(0,0.1^2) noise around one image | flags 0/20 at h=5,10,20; 1/20 at h=40; shuffled streams 1/20 at h=20 | PASS (implementation behaves as their paper assumes on their benign types) |
| 3b | Same test on corpus sessions (100 calibration sessions each) | attack 100%, random walk 100%, restore 85-100% depending on lag (100% at h>=20) | Supports prediction P7: the temporal statistic flags a benign optimizer |
| 3c | **Unplanned finding**: GWAD+ on the i.i.d.-noise streams | **20/20 alarm**; Blacklight 0/20 | GWAD+ flags the benign traffic type the 2026 competitor uses to set its FPR |
| 4 | LIME-style explanation traffic (SLIC, requested 40 segments -> 22-35 actual; 1,000 Bernoulli(0.5) masks; segment-mean fill) on the 20 development images | Blacklight alarms 20/20, median first alarm at query 7; **GWAD+ 0/20 alarms** despite 742 eligible windows; Ljung-Box 2/20 | PASS for Blacklight; GWAD+ does NOT flag LIME (Tier A splits by detector family) |
| 5 | LIME utility: deletion AUC of p(label), LIME order vs random order (10 random orders) | median 0.21 vs 0.67; LIME better on 20/20 images | PASS |
| 6 | Symmetric pre-flip cut implemented in `analyze_specificity_workloads.py` (`pre_either_flip`) | output trajectory 0.59-0.62 vs boundary_probe/counterfactual, 0.98-1.00 vs restore/confidence_boost; Blacklight 0.48-0.51; matches the earlier ad-hoc numbers exactly | PASS |
| 7 | Cost on non-duplicate streams | 3.1 s per shuffled session, 2.0 s per LIME session (all detectors online) | PASS (X0 4,000 shuffled sessions ~ 35 min on 6 shards) |
| 8 | GTSRB at 32 px, ResNet-18, 1 epoch (`train_gtsrb_models.py --image-size 32`) | runs; val 0.914, test 0.786 after one epoch (2.6 s/epoch) | PASS (15 epochs is minutes) |

## Consequences for the plan

- The detector ports are trustworthy on benign non-duplicate traffic (T1, T2):
  the "your port alarms on everything" objection is answered.
- X1 (Lee-Fang-Chang audit) is likely to confirm P7: their Phase-2 statistic,
  faithful on their own benign types, flags a label-preserving restoration
  optimizer. Phase 1 (their similarity stage) still has to be implemented; the
  full detector flags only sequences that pass both phases.
- Tier A is detector-specific: Blacklight flags explanation traffic within 7
  queries, GWAD+ does not. The claim becomes "similarity-hash defenses flag
  explanation traffic; update-geometry defenses flag i.i.d.-noise traffic and
  benign optimizers" rather than "every detector flags explanation traffic".
- New result to carry into X0: GWAD+ alarms on 100% of i.i.d. Gaussian-noise
  streams around one image, a benign construction used by the 2026 competitor.
- n = 20 in every pre-test; these are go/no-go checks, not paper numbers.

# X0 control corpus results (2026-09-26)

`analysis_outputs/stateful_specificity_controls_20260925/` (4,800 sessions; summary.log,
controls_predictions.json). Wilson 95% intervals.

| control | n | Blacklight | GWAD+ | GWAD | Ljung-Box h=20 |
|---|---|---|---|---|---|
| shuffled test images | 4,000 | 42 (1.05% [0.8, 1.4]) | 0 (0 windows) | 0 | 2.5% [2.0, 3.0] |
| i.i.d. N(0, 0.1^2) noise | 400 | 0 | 396 (99.0%) | 400 (100%) | 2.8% [1.5, 4.9] |
| JPEG x brightness sweep | 400 | 392 (98.0%) | 400 (100%) | 400 (100%) | 400 (100%) |

Predictions: P5a Blacklight <= 1% on shuffled FAILED narrowly (1.05%; per-query rate
42/4,096,000 = 1.0e-5, far below Blacklight's published 0.37%; alarms come late,
median query 776, i.e. rare chance collisions over long sessions). P5a GWAD family
no alarm: held. P5b GWAD+ on noise >= 90%: held (99%). P5c Blacklight on sweep
>= 90%: held (98%). P5d Ljung-Box <= 5% on shuffled and noise: held (2.5%, 2.8%,
near the nominal 2.5%).

# X1 Lee-Fang-Chang full-detector results (2026-09-26)

`analysis_outputs/lfc_workloads_20260926/analysis/` (lfc_alarm_rates.csv, lfc_predictions.json).
Regenerated corpora: 9,600/9,600 sessions SHA-256-identical to the original query streams.
Evaluation split, frozen configuration (Phase 1 online; Phase 2 Ljung-Box, lags min(10, n//5), re-run on every update).

| objective | denoise alarm (median first query) | deblur alarm (median first query) |
|---|---|---|
| attack | 1.000 (91) | 1.000 (124) |
| restore | 1.000 (98) | 0.990 (217) |
| confidence_boost | 1.000 (97) | 1.000 (128.5) |
| boundary_probe | 1.000 (88.5) | 0.995 (121) |
| counterfactual | 1.000 (86) | 1.000 (121.5) |
| random_walk | 1.000 (15.5) | 1.000 (15) |
| shuffled (clean, n=2,000) | 0.000 | |
| noise (clean, n=200) | 0.000 | |
| sweep (clean, n=200) | 0.555 (189) | |

Lag-20 sensitivity: same rates within 0.01. All P7 predictions held (P7a-d).
The detector reproduces its paper's CIFAR-10 TPR 1.00 / FPR 0.00 on the paper's
own benign types and flags 99-100% of benign optimizers at the attack's timing.

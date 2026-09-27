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

# X2 explanation-traffic results (2026-09-26)

`analysis_outputs/explanation_clients_20260926/` (explanation_summary.csv,
explanation_predictions.json). Evaluation split, n = 200 images per client; Wilson 95% CIs;
alarm rate (median first-alarm query).

| client (queries) | Blacklight | GWAD+ | GWAD | Lee-Fang-Chang | utility: random - client deletion AUC, median [95% CI] |
|---|---|---|---|---|---|
| LIME (1,000) | 1.00 (6) | 0.00 | 0.00 | 0.41 (409) | 0.198 [0.166, 0.234] |
| KernelSHAP (500) | 1.00 (3) | 0.31 (265) | 0.31 (260) | 0.67 (126) | 0.152 [0.130, 0.198] |
| Occlusion (842) | 1.00 (2) | 1.00 (259) | 1.00 (258) | 1.00 (17) | 0.258 [0.224, 0.286] |
| RISE (1,000) | 0.16 (720) | 0.00 | 0.00 | 0.00 | 0.161 [0.132, 0.189] |

Predictions: P6a (Blacklight >= 90%, median first alarm <= 50 on LIME, KernelSHAP,
occlusion) held for all three. P6b (Blacklight <= 10% on RISE) FAILED: 16% [12, 22],
alarms late (median query 720). P6c (utility, primary metric) held for all four clients.
Sensitivity (pixel-order random baseline, blurred fill): KernelSHAP, occlusion and RISE
still beat random; LIME's CI includes 0 (0.015 [-0.019, 0.060]).

Every legitimate explanation client is blocked by at least one released or 2026 detector
in most sessions, except RISE (flagged by Blacklight in 16% of sessions, by no other).

# X4 SimBA corpus results (2026-09-26)

`analysis_outputs/specificity_simba_20260926/` (4,000 sessions; pixel SimBA, 8/255 steps).
Evaluation AUROC, attack vs benign (1,024 calls | symmetric pre-flip cut):
- Blacklight: restore 0.48-0.50, confidence_boost 0.49-0.53, boundary_probe 0.56-0.63 | 0.48-0.52.
- GWAD+: restore 0.92-0.97, confidence_boost 0.74-0.87, boundary_probe 0.83-0.91,
  counterfactual 0.59-0.66 (no GWAD window before the pre-flip cut).
- Output trajectory: restore / confidence_boost 1.00 | 1.00; boundary_probe 0.98-0.99 | 0.58-0.63;
  counterfactual 0.66-0.76 | 0.58-0.60.
- Lee-Fang-Chang: 100% alarms on every objective. Native Blacklight and GWAD+: 100% alarms on
  every objective.
Predictions: P8 FAILED for GWAD+ (held for Blacklight). P9 held (0.58-0.63). P10 held. P11 held.
Mechanism of the P8 failure: SimBA's acceptance rate depends on the objective (attack
78-92% of steps accepted, restore 50-67%, confidence_boost 44-80%, boundary_probe 43-49%),
and GWAD's inter-query delta features pick this up ("acceptance-rate leak"). NES updates on
a fixed schedule and shows no such leak. At released thresholds every SimBA session is
still blocked; an attacker can remove the leak by matching a benign acceptance rate.

# X5-X6 robustness corpora (2026-09-26), scored against frozen P8-P11

score_robustness_corpus.py; analysis/robustness_predictions.json in each corpus directory.

| corpus (sessions) | P8 query-only vs Tier B (max AUROC) | P8 vs random walk | P9 output pre-flip vs Tier C (max) | P10 output vs Tier B (min) | P11 LFC on Tier B (min) |
|---|---|---|---|---|---|
| NES ResNet-18 seed 0 (main, 4,800) | 0.62 (1 cell) | 1.00 | 0.60-0.62 (post hoc) | 1.00 | 0.99 |
| SimBA ResNet-18 (4,000) | FAILED: GWAD+ 0.97 (Blacklight 0.54) | n/a | 0.63 | 1.00 | 1.00 |
| NES VGG19-BN (4,800) | 0.53 | 1.00 | 0.61 | 1.00 | 0.995 |
| NES robust ResNet-50, Engstrom (4,800) | 0.52 | 1.00 | 0.67 | 1.00 | 1.00 |
| NES ResNet-18 seed 1 (4,800) | 0.54 | 1.00 | 0.59 | 1.00 | 0.985 |
| NES ResNet-18 seed 2 (4,800) | 0.54 | 1.00 | 0.59 | 1.00 | 1.00 |
| NES GTSRB 32 px, 43 classes (4,536) | 0.50 | 1.00 | FAILED narrowly: 0.711 [0.674, 0.747] (deblur boundary_probe; other cells 0.67-0.69) | 0.97 | 0.97 |

Two frozen predictions failed across 32,536 new sessions: P8 for GWAD+ under SimBA
(acceptance-rate leak, see X4) and P9 on GTSRB by 0.011 (CI includes 0.70).

# X9 ImageNet corpus results (2026-09-27)

`analysis_outputs/specificity_imagenet_20260926/` (3,600 sessions; ResNet-50, 224 px, 100 classes,
tiled NES; score.log, analysis/). Restore is excluded as an invalid workload (0% accepted steps,
frozen before inspection); its sessions are trivially separable because the client never moves
(GWAD+ 0.999), which says nothing about specificity.

Evaluation AUROC at 1,024 calls, attack vs:
- confidence_boost (matched Tier B): GWAD+ 0.60-0.62 [0.56, 0.68], Blacklight 0.54 [0.48, 0.60],
  combined 0.60-0.62; output trajectory 1.00.
- random walk: 1.00 for every detector.
- boundary_probe: GWAD+ 0.84-0.87, Blacklight 0.59-0.60 (acceptance-rate leak: boundary_probe
  accepts 23-27% of tiled-NES steps vs 67-76% for the attack; same mechanism as SimBA).
- counterfactual: GWAD+ 0.52-0.59, Blacklight 0.52-0.53.
Native alarm rates: GWAD+ 100% of every objective; Blacklight (paper's ImageNet setting) 30-38%
of attacks vs 37-38% of confidence_boost (96-98% of random walks); Lee-Fang-Chang 98.5-100% of
confidence_boost and 99.5-100% of attacks.
Predictions (restore excluded): P8 vs confidence_boost at the 0.62 bound (GWAD+ 0.621 denoise,
CI includes the bound; Blacklight 0.54 held); P8 vs random walk held; P9 held (0.64, 0.70);
P10 held (1.00); P11 held (0.985).

# X10 ImageNet explanation-traffic results (2026-09-27)

`analysis_outputs/explanation_clients_imagenet_20260926/` (800 sessions, 0 duplicate ids; ResNet-50,
224 px, 200 evaluation images; Blacklight / Lee-Fang-Chang ImageNet settings; GWAD as released).
Alarm rate [Wilson 95% CI] (median first-alarm query); utility = random - client deletion AUC.

| client (queries) | Blacklight | GWAD+ | Lee-Fang-Chang | utility (primary) | utility (pixel/blur sensitivity) |
|---|---|---|---|---|---|
| LIME (1,000) | 1.00 [0.98, 1.00] (5) | 0.05 | 0.31 (89) | 0.268 [0.230, 0.291] | 0.000 [-0.018, 0.020] |
| KernelSHAP (500) | 1.00 [0.98, 1.00] (3) | 0.09 | 0.32 (75) | 0.162 [0.146, 0.182] | 0.024 [0.005, 0.045] |
| Occlusion (626) | 1.00 [0.98, 1.00] (2) | 0.975 (259) | 0.995 (18) | 0.233 [0.207, 0.259] | 0.001 [-0.011, 0.023] |
| RISE (1,000) | 0.035 [0.02, 0.07] (85) | 0.00 | 0.32 (140) | 0.230 [0.204, 0.244] | -0.005 [-0.023, 0.011] |

Predictions: P13a (Blacklight >= 90% on LIME, KernelSHAP, occlusion) held; P13b (utility, primary
metric) held for all four; RISE <= 10% held (3.5%). As at 32 px, the pixel-order/blur sensitivity
metric rates most clients near random at 224 px (only KernelSHAP stays significant).

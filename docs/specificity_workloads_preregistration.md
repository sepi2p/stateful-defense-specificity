# Matched benign-optimization workloads: frozen predictions and analysis

Date: 2026-09-25. Written while `run_specificity_workloads.py` was generating the
corpus and before any detector result from it was inspected.

## Question

Stateful black-box defenses are evaluated against benign traffic that does not
optimize (shuffled test sets, video frames). Do they separate adversarial
optimization from benign optimization that uses the same optimizer, budget and
perturbation ball, and differs only in the objective? Which detector-visible
signal, if any, carries the intent?

## Corpus (fixed before inspection)

- CIFAR-10, ResNet-18 seed 0, the frozen gate manifest (400 source images:
  100 fit, 100 calibration, 200 evaluation), denoise and deblur corrupted starts.
- NES, 8 antithetic pairs, sigma 2/255, sign step 0.25/255, 8/255 L-inf ball,
  1,024 classifier calls including every proposal.
- Objectives: attack (untargeted CE), restore, random_walk, confidence_boost,
  boundary_probe (margin minimization with label-preserving acceptance),
  counterfactual (CE to the start's runner-up class + 100 x MSE proximity).
- attack / restore / random_walk replay the frozen gate query streams exactly
  (SHA-256 of the float32 query bytes matched on all 12 sessions checked).
- Detectors: released GWAD and GWAD+ (pinned commit, frozen Delta-Net), official
  Blacklight fingerprint with published CIFAR-10 parameters (port verified
  query-by-query against the release), and output-trajectory features computed
  from the classifier's own logits with the first-query prediction as reference
  (no true label).

## Predictions

P1. Query-only detectors (GWAD, GWAD+, Blacklight) separate attack from each
    matched benign NES workload at evaluation AUROC <= 0.60 (1,024 calls).
P2. The same query-only detectors separate attack from the random walk better
    than from any benign NES workload.
P3. Output-trajectory detectors separate attack from label-preserving or
    confidence-raising workloads (restore, confidence_boost) at AUROC >= 0.90.
P4. Output-trajectory detectors separate attack from workloads whose effect on
    the output is adversarial (boundary_probe, counterfactual) at AUROC <= 0.75
    on the prefix before the attack's first label change.

P4 is the prediction most likely to fail. If the output detector separates
boundary_probe or counterfactual from the attack, report it and characterize
which output feature does it (e.g. targeted vs untargeted class dynamics); do not
drop the workload.

## Analysis rules

- Fit orientation / logistic models on the fit split; report AUROC on the
  evaluation split with source-image bootstrap 95% intervals (1,000 resamples).
- Prefixes 128, 256, 512, 1,024 calls, plus the attack-truncated prefix (each
  pair uses the attack session's first-success index for both sessions).
- GWAD produces no window before about 258 calls; report its early-prefix cells
  as "no decision" rather than AUROC 0.5.
- Report every objective pair and both starts. No objective, prefix or feature
  is dropped after results are seen.

## Addendum 2026-09-25: X0 non-optimizing controls (frozen before X0 runs)

Controls, all with GWAD, GWAD+ and Blacklight online, 1,024 queries each:
(a) shuffled: 1,024 distinct CIFAR-10 test images not in the manifest, 10
sessions per manifest image (4,000 sessions, split inherited from the image);
(b) noise: i.i.d. N(0, 0.1^2) around the clean manifest image (Lee-Fang-Chang's
benign construction; 400 sessions); (c) sweep: JPEG quality 95->20 x brightness
-0.2..+0.2 grid of the clean manifest image, 32 x 32 = 1,024 queries (400 sessions).

Already known from the 20-session development pre-test (disclosed, not a
prediction): on (a) Blacklight 0/20 and GWAD+ 0 windows; on (b) GWAD+ 20/20
alarms and Blacklight 0/20.

P5a. Blacklight alarms on <= 1% of (a) sessions; GWAD/GWAD+ produce no alarm on (a).
P5b. GWAD+ alarms on >= 90% of (b) sessions (pre-test replication at n = 400).
P5c. Blacklight alarms on >= 90% of (c) sessions (near-duplicate by construction).
P5d. Ljung-Box (h = 20, p < 0.025, min length 15) on p_{c1} flags <= 5% of (a) and (b).
No outcome here is used to change detector parameters.

## Addendum 2026-09-26: X1 Lee-Fang-Chang (arXiv 2606.21592) reimplementation (frozen before the corpus run)

Configuration fixed by the faithfulness check on 50 calibration images using only
the paper's attacks and benign types (`analysis_outputs/lfc_faithfulness_20260926/`):
Phase 1 with q=80, r=45, w=20, k=15, threshold 33, Bernoulli term 0 (the term as
read from the paper prevents any grouping even at scale 0.05; scales 0-0.01 give
identical TPR/FPR), window positions drawn once per detector (seed 20260926);
Phase 2 Ljung-Box p<0.025, min length 15, re-run on every update, lags
min(10, n//5) primary, 20 as sensitivity. Faithfulness: SimBA 1.00, Square 1.00,
shuffled 0.00, noise 0.00 (paper Table 4: TPR 1.00, FPR 0.00).

Disclosed prior knowledge: Phase 2 alone (h=20 on whole sessions) flagged
85-100% of restore calibration sessions in the pre-test.

On the evaluation split of the regenerated corpora (queries verified identical
to the original corpora by SHA-256):
P7a. The full detector alarms on >= 90% of restore and of confidence_boost sessions, each start.
P7b. It alarms on >= 95% of attack sessions, each start.
P7c. It alarms on <= 2% of shuffled and of noise control sessions.
P7d. It alarms on >= 90% of random_walk sessions.

## Addendum 2026-09-26: X2 explanation-traffic tier (frozen before the run)

Clients (clean manifest images, 32 px, ResNet-18 seed 0; all detectors online incl.
Lee-Fang-Chang Phase 1 with the frozen configuration; no client exceeds 1,024 queries):
- lime: SLIC (40 requested segments), 1,000 samples (first unperturbed), Bernoulli(0.5) masks, segment-mean fill.
- kernelshap: 4 x 4 grid of 8 x 8 cells, blurred fill, 500 coalitions (all-on and all-off first, sizes drawn from the Shapley kernel), weighted least squares.
- occlusion: 4 x 4 patch, stride 1 (841 queries) plus the unperturbed image, per-channel image-mean fill.
- rise: 1,000 masks, 7 x 7 Bernoulli(0.5) grid, bilinear upsampling with random shift, black fill.
Utility (computed with extra model calls NOT in the detector stream): deletion curve of
p(label) over 16 steps removing pixels in saliency order (blurred fill), AUC vs the mean of
10 random pixel orders.

Disclosed prior knowledge (20 development images, LIME only): Blacklight 20/20, median
first alarm at query 7; GWAD+ 0/20 alarms; LIME deletion AUC 0.21 vs random 0.67.

P6a. Blacklight alarms on >= 90% of lime, kernelshap and occlusion sessions, median first alarm <= 50.
P6b. Blacklight alarms on <= 10% of rise sessions (large random masks share few windows).
P6c. Utility: for each client, deletion AUC is below the random-order AUC on >= 90% of
     evaluation images; a client that fails is excluded from any legitimacy claim.
GWAD, GWAD+ and the Lee-Fang-Chang detector are reported without a prediction (exploratory).

### Amendment to P6c (2026-09-26, before the X2 main run; development images only)
A development-image check (20 images, no evaluation data) showed that pixel-order deletion
with blurred fill vs random *pixel* orders rates every client near random (better on 40-70%
of images), including KernelSHAP and occlusion: at 32 px a blurred object stays
recognizable while scattered random pixel removal is unusually destructive. Against a
spatially smooth random baseline (random 4 x 4 grid, bilinear upsampling), every client
beats random on 75-95% of development images under either fill. Revised P6c, frozen now:
primary utility = deletion AUC with per-channel image-mean fill vs the mean of 10 smooth
random orders; a client passes if the median paired difference (random - client) is > 0
with a source-image bootstrap 95% CI excluding 0 on the evaluation split. The per-image
>= 90% rule is dropped as arbitrary. The original pixel/blur variant is reported as a
sensitivity row. P6a and P6b are unchanged.

## Addendum 2026-09-26: X4-X6 robustness corpora (frozen before the runs)

Corpora (all with GWAD, GWAD+, Blacklight and Lee-Fang-Chang Phase 1 online; same
objectives, starts, eps, budget and analysis as the NES CIFAR-10 corpus):
- X4 SimBA: pixel basis, 8/255 steps, CIFAR-10 ResNet-18 seed 0, frozen manifest; five
  optimizing objectives (the objective-free walk is optimizer-independent).
- X6 second models, NES: BlackboxBench VGG19-BN; RobustBench Engstrom2019Robustness
  (L-inf-robust ResNet-50); ResNet-18 seeds 1 and 2. Each model gets its own manifest
  (same seed and split counts), since start eligibility depends on the model.
- X5 GTSRB at 32 px: ResNet-18 trained with train_gtsrb_models.py --image-size 32
  (15 epochs), 43 classes x (1 development, 2 fit, 2 calibration, 5 evaluation) images.
  Detectors run as released (CIFAR-10 parameters and Delta-Net).

P8. For every corpus, the query-only detectors (GWAD+, Blacklight, their combination)
    give evaluation AUROC <= 0.62 for attack vs each Tier-B objective (restore,
    confidence_boost) at 1,024 calls, and >= 0.90 vs the random walk where it exists.
P9. Symmetric pre-flip cut (pre_either_flip), output trajectory, attack vs boundary_probe
    and vs counterfactual: AUROC <= 0.70 on every new corpus.
P10. Output trajectory vs restore and confidence_boost at 1,024 calls: AUROC >= 0.90 on
    every new corpus.
P11. Lee-Fang-Chang (frozen configuration) alarms on >= 90% of restore and
    confidence_boost evaluation sessions on every new corpus.

## Addendum 2026-09-26: X7 output diagnostic (result) and exploratory follow-up E1

Result (analyze_output_diagnostic.py; outputs in analysis_outputs/output_diagnostic_20260926/):
the planned interpretable statistic (reference-margin drawdown) separates attacks from
restore and confidence_boost (AUROC 0.99-1.00, full sessions) but is INVERTED against
explanation clients (KernelSHAP 0.00-0.03, LIME 0.01-0.28, occlusion 0.35-0.81): masking
collapses the margin harder than an attack. With a single threshold at 1% FPR on pooled
calibration benign traffic, the two-stage detector (similarity alarm AND drawdown) detects
0.0-0.2% of attacks. The drawdown diagnostic is therefore reported as a negative result.

E1 (EXPLORATORY; declared after seeing the drawdown result, before computing E1 on any
split). Statistic: Kendall's tau between query index and reference margin within the
largest Lee-Fang-Chang Phase-1 subsequence (reference = its first query's prediction);
score = -tau (persistent decline = attack-like). Same two-stage design and pooled
calibration threshold (1% FPR, calibration split); evaluated once on the evaluation split
(early = first 256 queries; full). Success criterion: attack TPR >= 50% at that threshold
with every benign workload's FPR <= 5%. Attack-equivalent clients reported separately. If
E1 succeeds it is reported as exploratory and must be confirmed on the X4-X6 corpora
before any claim.

E1 result (2026-09-26): FAILED its success criterion. Full sessions: attack TPR 98.2% at
the pooled 1% calibration threshold; FPR <= 0.5% on restore, confidence_boost, noise,
shuffled, occlusion, KernelSHAP, RISE, LIME; random walk 4.5%; JPEG/brightness sweep
12.5% (> 5%). First 256 queries: attack TPR 5.0%. Attack-equivalent: counterfactual
98.5%, boundary_probe 56.2% (full). Reading: a persistent decline of the reference margin
is an intent signal against optimizing and explanation clients, but it matures only after
most attacks have succeeded (median first success ~480 queries), flags monotone
degradation sweeps, and cannot separate attack-equivalent clients.

## Addendum 2026-09-26: X9 ImageNet corpus (frozen before the run)

ImageNet-1k validation images (Resize 256, CenterCrop 224), torchvision ResNet-50
(IMAGENET1K_V1); 100 classes sampled with seed 20260924; manifest 1 development, 1 fit,
1 calibration, 2 evaluation images per class; this run uses fit + evaluation only (300 images),
both corrupted starts, all six objectives (3,600 sessions). Detectors: Blacklight with the
paper's ImageNet setting (window 50; S=50, T=25, q=50, p=1); Lee-Fang-Chang with the paper's
ImageNet setting (q=80, r=75, w=50, k=20, threshold 7) and the frozen Phase-2 configuration;
GWAD / GWAD+ as released (CIFAR-trained Delta-Net, reported with that caveat).
NES hyperparameters chosen on 20 DEVELOPMENT images only (attack success, detectors off),
grid tile {1,4,8} x step {0.25,1,2}/255: tile 8 (8x8 coarse directions, nearest upsampling),
step 1/255 (success 0.88 vs 0.28 for the CIFAR setting); sigma 2/255, 8 pairs, eps 8/255
unchanged. The same NES settings are used for every objective.
P12. P8-P11 as stated for X4-X6, evaluated on this corpus; for P8 the query-only family is
     reported both with and without GWAD (whose released model is CIFAR-trained).

## Addendum 2026-09-26 (22:50): X9 restoration invalid at 224 px; X10 ImageNet explanation tier (frozen before the run)

Observed during X9 (first 735 sessions; no detector result inspected): the restore objective
accepts 0% of NES steps at 224 px under the frozen tiled NES, and 0-12% (PSNR gain ~0) under
untiled NES at step 0.25/255 or 1/255 on 10 development images. Zeroth-order restoration does
not make progress at this dimensionality, so X9 restore sessions are reported as an invalid
workload (the client never moves) and excluded from P8/P10/P11 on ImageNet; confidence_boost
remains the matched Tier-B objective there. All other X9 objectives make progress (attack
success 77-85%; confidence_boost +5 margin; boundary_probe reaches the boundary).

X10: LIME (50 SLIC segments, 1,000 samples), KernelSHAP (8 x 8 grid of 28-px cells, blurred
fill, 500 coalitions), occlusion (32-px patch, stride 8: 625 + 1 queries), RISE (unchanged)
on the 200 ImageNet evaluation images, ResNet-50, Blacklight / Lee-Fang-Chang ImageNet
settings, GWAD as released. Utility metric as amended for X2.
P13a. Blacklight alarms on >= 90% of LIME, KernelSHAP and occlusion sessions.
P13b. Every client passes the P6c utility rule.
RISE, GWAD, GWAD+ and Lee-Fang-Chang: reported without prediction.

## Addendum 2026-09-27: X11 acceptance-throttled SimBA attack (frozen before the run)

Motivation: on the SimBA corpus GWAD+ separates the attack from benign clients (P8 failed),
and acceptance rates differ by objective. Whether the acceptance rate *causes* the separation,
and whether an attacker can remove it, has not been tested.
Design: CIFAR-10, ResNet-18 seed 0, frozen manifest (fit + evaluation images, both starts),
pixel SimBA at 8/255 as in X4, objective = attack (CE of the source label), with one change:
an improving step is accepted only while the session's running acceptance rate stays at or
below a cap r*. For each session r* is drawn (deterministically from the session seed) from the
acceptance rates of the SimBA *restore* sessions of the FIT split with the same start
(`throttle_rates_restore_fit.json`). Detectors online as in X4. 600 sessions.
Analysis: same logistic models and splits as the main analysis, positives = throttled attack,
negatives = the existing SimBA restore sessions (primary) and confidence_boost (secondary).
P14a. GWAD+ AUROC, throttled attack vs restore, <= 0.65 on both starts (unthrottled: 0.92-0.97).
P14b. The throttled attack still succeeds within the budget in >= 50% of evaluation sessions
      (unthrottled: 85-90%).
Reported without prediction: Blacklight, output trajectory, Lee et al. alarms, median first
success, and the comparison against confidence_boost (whose acceptance rate the attacker does
not match).

## Addendum 2026-09-27: post hoc analyses for the manuscript (declared as exploratory)

Declared after all corpora were analysed; none of these has a frozen prediction and each is
labelled exploratory or post hoc wherever it is reported.
- Acceptance-only AUROC per (corpus, start, objective) cell, next to the GWAD+ and Blacklight
  AUROCs (analyze_acceptance_leak.py, part 1).
- Prefix sweep of the exploratory trend statistic E1 (decision-time curve): attack detection
  rate at the pooled 1% calibration threshold for prefixes 64-1,021 queries, against the
  distribution of attack first-success times.
- Per-query Blacklight flag rates per objective; first-alarm distributions (Figure 1).

## Addendum 2026-09-27: X12 sensitivity of the Lee-Fang-Chang reimplementation (frozen before the run)

A close reading of arXiv 2606.21592 (Sec. 5.1) shows two departures of the frozen
reimplementation from the literal text: the salt is drawn from [0, N) rather than [0, 1], and the
randomized rounding term is switched off (as already recorded on 2026-09-26) rather than applied at
the stated rate (1/w)(x mod q)/q. The paper does not say whether the rounding randomness is redrawn
for every query. Also noted: the paper evaluates sequences of 50 queries; our sessions have 1,021.
Variants (lfc_sensitivity.py): frozen; salt_unit; literal_fixed (paper's rate, one rounding
threshold per position fixed per detector); literal_fresh (paper's rate, redrawn per query).
Part A: SimBA, Square, shuffled, noise streams on 50 calibration images. Part B: NES attack,
restore, confidence_boost sessions on 50 evaluation images, both starts (300 sessions, replayed
from the corpus seeds). Phase 2 as frozen.
P15. Every variant that reproduces the paper on Part A (alarm >= 0.95 on SimBA and Square,
     <= 0.05 on shuffled and noise) alarms on >= 90% of restore and of confidence_boost sessions.
Reported without prediction: alarm rates within the first 50 queries.

### Amendment to X12 (2026-09-27, before any X12 result was seen; the first launch was stopped after 12 minutes)
Added (i) a small-noise stream, sigma = 0.1 intensity levels on the 0-255 scale, because the paper's
noise scale is ambiguous and its own Blacklight baseline flags 97/100 noise sequences, which implies
near-duplicate queries; (ii) the JPEG/brightness sweep; (iii) the paper's evaluation protocol as a
second schedule ("sequence-50"): Phase 1 on the first 50 queries, Ljung-Box applied once to every
subsequence of >= 15 queries (lags min(10, n//5); 5 as sensitivity). The frozen online schedule
(re-test on every update) can accumulate false alarms on long benign near-duplicate streams; both
schedules are reported. P15 is unchanged and refers to the online schedule. Reported without
prediction: sequence-50 alarm rates for every stream and for attack, restore, confidence_boost.

## Addendum 2026-09-27 (evening): correction of the Blacklight decision rule

An independent check of the manuscript against the cited papers found that the project's
Blacklight port flags a query whose fingerprint shares AT LEAST 25 hashes with an earlier
one, whereas the paper and the release flag a query that shares MORE THAN 25
(`match_num > threshold`). The per-query match counts are stored in every trace, so the
published rule is applied to the stored counts (`blacklight_rule.py`); no session is
regenerated and the session logs keep the recorded value. All Blacklight-dependent analyses
are repeated (`analysis_r2` in each corpus). Predictions P1-P15 are evaluated again on the
corrected analysis, and the manuscript reports for which checks the verdict changes. This
correction was made after all results of P1-P15 were known.

## Addendum 2026-09-27 (evening): X13 sensitivity of the matched-objective result to the NES configuration

Written and committed before any X13 session is generated (the commit that adds this
addendum precedes the run). Reason for the experiment: the chance-level result of P1/P8
was obtained with one NES configuration (sign step 0.25/255, probes at 2/255, move only if
the objective improves). Section 5.6 of the draft explains the separation under SimBA and
tiled NES by the size of the accepted step. X13 tests whether the result under NES depends
on the step size and on the acceptance test.

Design: CIFAR-10, ResNet-18 seed 0, the frozen manifest, fit (100) and evaluation (200)
images, both starts, objectives attack, restore, confidence_boost, budget 1,024, eight
antithetic pairs, probes at 2/255, the same session seeds as the main corpus. Variants:

- V1: sign step 1/255, move only if the objective improves (and the label is kept, for restore).
- V2: sign step 2/255, same acceptance rule.
- V3: sign step 2/255, the client always moves to the stepped point (NES as published by
  Ilyas et al.; the label check of restore does not apply).

1,800 sessions per variant. Detectors as in the main corpus (Blacklight with the published
rule, released GWAD and GWAD+, Lee-Fang-Chang observer). Analysis as for the main corpus
(`analyze_specificity_workloads.py`, fit split for fitting, evaluation split for reporting),
plus the AUROC of the session's acceptance rate alone (orientation fixed on the fit split).

Predictions (each is checked per start and per Tier-B client, full session):

- P16a. At the released operating points Blacklight and GWAD+ raise an alarm in >= 99% of
  the evaluation sessions of every client in every variant.
- P16b. Blacklight: AUROC attack vs Tier-B client <= 0.62 in every cell of every variant.
- P16c. V1 and V2: the AUROC of GWAD+ does not exceed the AUROC of the acceptance rate alone
  by more than 0.05 in any cell.
- P16d. V3 (no acceptance test): AUROC of GWAD+ <= 0.62 in every cell.
- P16e. V2: the AUROC of GWAD+ against restore exceeds its value in the main corpus by at
  least 0.10 on at least one start. This is the prediction of the explanation given in
  Section 5.6 and the one most likely to fail; if it fails, the chance-level result does not
  depend on the step size, and the explanation of Section 5.6 has to be restricted.

Whatever the outcome, the manuscript reports all five checks and the AUROC of GWAD,
GWAD+ and Blacklight for every cell.

### Amendment to X13, 2026-09-27 15:58 (+0330), before any X13 result is analysed

The smoke test of the generator (one fit image, outputs in the scratch directory, detector
results not inspected) showed that with a sign step of 2/255 the restoration client did not
change the margin of that image at all, which suggests that it accepts no step, as on
ImageNet. A client that never moves is not a matched optimizer. Validity rule, fixed now:
in the variants with an acceptance test (V1, V2) a benign client is a valid workload only
if its median acceptance rate on the FIT split is at least 0.10. Cells of invalid workloads
are reported and marked, and they are excluded from P16b, P16c and P16e. If restoration is
invalid in V2, P16e is evaluated on V1 instead, and if it is invalid in both, P16e is
reported as not testable. V3 has no acceptance test, so every client is valid there by
construction; its restoration client moves without the label check.

## Addendum 2026-09-27 (evening): X14 explanation clients as implemented by libraries

Written and committed before any X14 session is generated. Reason: the explanation clients
of X2 and X10 are our own implementations of the query designs; a reader may ask whether
library implementations send streams that the detectors treat differently.

Design: clean evaluation images of the main CIFAR-10 manifest (200 images, 32 x 32) and one
evaluation image per class of the ImageNet manifest (100 images, 224 x 224); the models,
detectors and detector settings of X2 and X10 (Blacklight with the published rule). Clients,
with every tensor that the library passes to the model recorded in order:

- `lime_package`: lime 0.2.0.1, `lime_image` with its default settings.
- `captum_kernelshap`: Captum 0.9.0 KernelShap, SLIC superpixels (50 segments, compactness 30,
  sigma 3), 1,000 samples, hidden superpixels replaced by the mean colour of the image.
- `captum_occlusion`: Captum 0.9.0 Occlusion, window 15 x 15 at stride 8 and baseline 0 at
  224 px (the settings of the Captum tutorial), window 4 x 4 at stride 2 at 32 px.

Prior knowledge: our own implementations of the three designs are flagged by Blacklight in
100% of the sessions at a median of 2 to 6 queries, and occlusion is flagged by GWAD+ in
97.5 to 100%.

Predictions (evaluation images, each resolution separately):

- P17a. Blacklight raises an alarm in >= 90% of the sessions of each of the three clients,
  with a median first alarm at query <= 50.
- P17b. Blacklight flags >= 90% of the queries of each of the three clients.
- P17c. GWAD+ raises an alarm in >= 90% of the `captum_occlusion` sessions.
- P17d. Each client is useful by the rule of X2 (median paired difference in deletion area
  against spatially smooth random orderings > 0, bootstrap 95% interval excludes 0).

No prediction is made for GWAD+ on the other two clients or for the detector of Lee et al.;
their rates are reported.

### Amendment to X14, 2026-09-27 16:22 (+0330), before the run

The smoke test of the generator (one CIFAR-10 image, outputs in the scratch directory) showed
two properties of the library clients that the design had not considered.

- At 32 x 32 pixels the Captum occlusion client sends 226 queries, fewer than the 259 that
  GWAD+ needs before its first decision. P17c can therefore not be tested at 32 px; it is
  evaluated at 224 px only, and the 32-px cell is reported as "no decision possible".
- At 32 x 32 pixels the default segmentation of the lime package (quickshift) returned two
  superpixels for that image, so that its 1,000 samples contain at most four distinct images.
  This is what the package does with its defaults on such images; the number of superpixels
  is recorded for every session and reported. No prediction is changed.

## Addendum 2026-09-27 (evening): repetition of X12 with one more variant (not a prediction)

The independent check also found that the paper of Lee et al. lets a query join a group if
the number of shared windows "exceeds" the threshold (33 on CIFAR-10), whereas the
reimplementation joins at 33 or more. The grouping is path dependent and the queries are not
stored, so this cannot be recomputed from the traces. X12 is repeated unchanged with one added
variant, `strict_threshold` (the configuration used for every corpus, joining only above the
threshold); output in `analysis_outputs/lfc_sensitivity_r2`. No prediction is attached; the
manuscript reports the variant next to the others.

## Note added 2026-09-27 (night): times in this file

A second check of the manuscript found that the headings of two amendments above give times
that are later than the commits which contain them: the amendment to X13 is headed "15:58" and
was committed at 15:54:42 (commit b52ee9f), the amendment to X14 is headed "16:22" and was
committed at 16:20:00 (commit 3116339). The times in the headings were written as estimates
and are wrong by a few minutes; the commit times are authoritative. The X13 amendment was
written 42 seconds AFTER the X13 run had been launched (15:53:58) and before any of its
sessions had been analysed. The amendment to X12 says that the first launch was stopped "after
12 minutes"; by the log it ran for about 4 minutes. The text above is left unchanged.

Outcome of X13 and X14 (2026-09-27): P16a, P16c, P16d, P16e held; P16b failed (Blacklight
0.659 against restoration at step 1/255). P17a and P17b held at both resolutions; P17c failed
(GWAD+ alarms in 49% of the Captum occlusion sessions at 224 px); P17d held at 224 px and is
not evaluable for the lime package at 32 px, whose default segmentation returns one superpixel
for most images. The deletion-metric rule of P6c/P13b/P17d is also met by these empty
explanations, so it does not establish that an explanation is informative.

## X15 (2026-09-28): validity of the library explanations and enforcement with rejection

Reason. An external review of the draft raised two objections that the record above cannot
answer. (1) The deletion rule of P6c/P13b/P17d is met by empty explanations, so nothing in the
record shows that the flagged library clients do useful work. (2) The detectors only observed
the streams. That Blacklight "flags 98 to 100% of the queries" does not show what a client
receives when the published response, rejection of flagged queries, is applied.

Material. The 300 sessions of X14 at 224 px (lime_package, captum_kernelshap,
captum_occlusion; one evaluation image of each of 100 ImageNet classes), for which the traces
hold every model output and the match count of Blacklight for every query. Secondary, without
predictions: the 600 sessions of X14 at 32 px and the sessions of our own explanation clients
(X2, X10).

Prior knowledge when this is written. Known from X14: Blacklight flags 98.3 to 99.9% of the
queries of the three clients at 224 px, first alarm at a median of 2 to 5 queries; the lime
package segments the 224-px images into 23 to 53 superpixels and the 32-px images into one
superpixel at the median; the values of the old deletion rule. Not known: any quantity of the
rule R2 below, any explanation computed under rejection.

Rule R2 (informative explanation). Deletion area as before (sixteen steps, per-channel mean
fill, probability of the source label). Two changes remove the defect of the old rule.

- Ties in the attribution are broken by a spatially smooth random field (random 4 x 4 grid,
  bilinear upsampling), the same field for an explanation and for its reference orderings.
- The reference is a randomization of the explanation itself that keeps its spatial
  structure, 20 draws per explanation: if the map has at most 256 distinct values (maps that
  are constant on segments), the values are permuted among the level sets of the map;
  otherwise (occlusion, RISE) the map is flipped at random in both directions and shifted
  circularly by offsets drawn uniformly.
- d = mean reference area - area of the explanation, per image. An explanation client is
  informative if the median of d over the images is positive and its bootstrap 95% interval
  (2,000 resamples of images) excludes zero.

A constant map has d = 0 exactly under R2. This is a property of the construction and is
checked in the code, not predicted.

Enforcement. Every query enters the history of Blacklight, flagged or not, as in the released
code (`add_img`). The three clients choose their queries without regard to the answers, so the
set of flagged queries under enforcement is the recorded one (match count > 25). The library
is run again with the seed of the session; every query is compared with the recorded stream
(SHA-256 of the whole stream); answers are served from the recorded outputs. Handling of a
rejected query by the client:

- H1, substitution: the client puts the uniform distribution in the place of a rejected answer
  and the library code runs unchanged (all three clients).
- H2, answered queries only: the surrogate model of the lime package is fitted on the answered
  samples alone (lime package only; Captum offers no entry point for this, and none is built).

Reported per client: share of answered queries; whether the client obtains the prediction for
its unperturbed image; Spearman rank correlation between the attribution maps with and without
enforcement (pixel level; 0 if the enforced map is constant); overlap of the fifth of the image
area with the largest attributions (0.2 by chance); d under R2.

Predictions (224 px, each library client separately):

- P18a. Without enforcement at least 95% of the explanations have ten or more distinct
  attribution values, and the client is informative by R2.
- P18b. Under enforcement with H1 the median Spearman correlation with the explanation
  without enforcement is below 0.3.
- P18c. Under enforcement with H1 the median of d is below one quarter of the median of d
  without enforcement.
- P18d. The lime package at 32 px is not informative by R2 (the interval of the median of d
  includes zero).

No prediction is made for H2, for the Captum clients at 32 px, or for our own clients; these
are reported. If the stream of a session cannot be regenerated bit for bit, the session is
excluded from the enforcement analysis and counted; if more than 10% of the sessions of a
client are excluded, P18b and P18c are declared not evaluable for that client.

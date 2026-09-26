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

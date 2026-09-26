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

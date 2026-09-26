# Theory notes: when can a stateful detector tell attack from benign optimization?

Draft, 2026-09-25. To be tightened for the paper.

## Setup

A session is a transcript τ = ((q_1, o_1), …, (q_T, o_T)) of queries q_t ∈ [0,1]^d
and model outputs o_t = M(q_t). A client runs a (possibly randomized) optimizer
A with internal randomness ξ and an objective f : outputs × queries → ℝ. A
stateful detector is any (randomized) test D(τ) → {0,1}. A *query-only* detector
sees q_{1:T} only; an *output-aware* detector sees the full τ.

Attack: objective f_adv (e.g. margin or CE of the source label). Benign client:
same optimizer, same start, budget and constraint set, objective f_ben.

## Lemma 0 (Le Cam, standard)

For any detector D, TPR(D) − FPR(D) ≤ TV(P_adv(τ), P_ben(τ)). For query-only
detectors the bound uses the marginal law of q_{1:T}, which can only be smaller.
This frames every result below; it is not itself a contribution.

## Proposition 1 (rank-equivalent objectives are undetectable)

Call A *comparison-based* if, given ξ, every query it issues depends on past
outputs only through the outcomes of comparisons f(o_i, q_i) < f(o_j, q_j)
(accept/reject decisions, argmin over a candidate set). SimBA, Square attack,
(1+1)-ES, random search, and label-only / decision-based methods whose decisions
are indicator comparisons fall in this class; NES and other finite-difference
gradient estimators do not (they use objective *values*).

If f_ben = φ ∘ f_adv on the region the session can reach, for some strictly
increasing φ, then for every realisation of ξ the two clients issue identical
queries and receive identical outputs, so P_adv(τ) = P_ben(τ) and every detector,
output-aware or not, has TPR = FPR.

Proof sketch: induction on t. Identical history ⇒ identical outputs (same model)
⇒ identical comparison outcomes (φ preserves order) ⇒ identical next query.

Consequences:
- Rank-equivalence is a property of the *objectives*, not of intent. Example:
  robustness self-assessment that minimizes the source-label margin while
  rejecting label-changing moves ("boundary probe") is rank-equivalent to the
  margin attack on every step before the attack's first label change; the
  transcripts coincide exactly on that prefix.
- After the first label change the only difference is the acceptance rule, i.e.
  whether the label-flipping candidate is kept. The earliest point at which any
  detector can separate the two is therefore the attack's success time; there
  is no *early* detection with better-than-chance specificity.

Planned empirical check: SimBA with margin objective, attack vs boundary probe;
verify SHA-identical transcripts up to the attack's first success.

## Proposition 2 (query-only detectors cannot see the objective's effect)

For finite-difference (value-based) optimizers such as NES, the queries depend on
f through the accepted iterates and the sign of the gradient estimate. If the
proposal law given the current iterate is fixed (isotropic Gaussian pairs), then
q_{1:T} depends on f only through the path of accepted iterates x_{0:K}. A
query-only detector is therefore a test on that path; features such as
delta-histograms, hash collisions or similarity counts are dominated by the
proposal law (identical across objectives) and carry information about f only
through the drift of x_k. Formalize as: the law of the proposal residuals
q − x_k is objective-independent, so any detector whose statistic is a function
of the residuals has TPR = FPR.

This matches what GWAD's Delta-Net and Blacklight fingerprints compute
(inter-query deltas / near-duplicate hashes, dominated by residuals). To check
empirically: attack vs any benign NES workload at AUROC ≈ 0.5 for GWAD/GWAD+/
Blacklight, whatever the benign objective's output effect.

## What output-aware detection can and cannot do

Output-aware detectors see o_t, hence f's effect on the model. They separate
attack from benign objectives whose output footprint differs (label-preserving
restoration, confidence raising). They cannot separate objectives that are rank-
equivalent to the attack on the observed prefix (Prop. 1). Between these lie
objectives with a similar but not rank-equivalent footprint (e.g. targeted
counterfactual vs untargeted attack); separability there is an empirical
question that the corpus answers.

Takeaway for the paper: specificity is bounded by how the *objective* acts on the
model, not by the query stream. Query-only defenses are blind to intent by
construction (Prop. 2); output-aware defenses can recover intent exactly to the
extent that benign objectives are not rank-equivalent to attacks on the prefix
the defender needs to act on (Prop. 1).

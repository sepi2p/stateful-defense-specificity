# Specificity of stateful black-box defenses: code, predictions and results

Research artifact for the article

> S. Montazeri and M. Ghatee, *Optimization Is Not an Attack: Rethinking the Specificity of
> Stateful Black-Box Defenses* (submitted, 2026).

Stateful defenses (Blacklight, GWAD, GWAD+, and the detector of Lee, Fang and Chang) flag clients
of a prediction service whose queries resemble the steps of a query-based attack. The study
measures their false positives on benign clients that send many variations of one input:
explanation clients from libraries (LIME, KernelSHAP, occlusion) and *matched benign optimizers*,
which run the attacker's optimizer under the attacker's constraints with a benign objective.

## What is here

| Path | Content |
|---|---|
| `docs/specificity_workloads_preregistration.md` | The predictions, amendments and outcomes, written during the study. Its history is part of the record (see below). |
| `docs/specificity_workloads_results.md`, `docs/specificity_pretests_2026-09-25.md`, `docs/specificity_theory_notes.md` | Working records of results, pre-tests and the transcript-identity argument |
| `experiments/gate_trajectory_signatures/` | Session generators, detector adapters, the reconstruction of the detector of Lee et al. (`lfc_detector.py`), analyses, and the scripts that generate every table and figure |
| `scripts/` | The launchers with which the corpora and experiments were run |
| `analysis_outputs/` | Session logs (`sessions_shard*.jsonl`, one line per session with detector results and the SHA-256 digest of its queries), run metadata, run logs and all result files of the analyses |
| `analysis_outputs/*/detector_signals.npz` | For the sessions that the evaluation procedure reads: Blacklight's count of matching hashes per query and the scores, query indices and decisions of GWAD and GWAD+, extracted from the traces |
| `analysis_outputs/operating_profile_r1/` | Results of the evaluation procedure under benign-alarm budgets (Section 3.4 of the article), CIFAR-10 retrospective and ImageNet confirmatory |
| `analysis_outputs/operating_profile_imagenet_20260928/` | Experiment X16: fresh ImageNet calibration and confirmation images, ResNet-50 and ConvNeXt-T |
| `paper/jisa_2026/tables`, `figures`, `numbers` | The generated tables, figures and numbers of the article and its supplementary material |
| `checkpoints/` | The four models trained by the authors (three CIFAR-10 ResNet-18, one GTSRB ResNet-18) |
| `third_party/README.md`, `THIRD_PARTY_NOTICES.md` | Third-party code and models: what is needed, from where, at which commit |
| `COMMIT_MAP.tsv` | Commit identifiers of this repository against those of the private working repository |

The session logs cover the 50,236 sessions of the study, the 9,600 sessions that were generated a
second time with an additional detector, a pilot of 2,640 sessions, the 900 replays of
experiment X15, and the 1,900 sessions of experiment X16 with their replays.

## What is not here

- **Per-query traces** (model outputs and detector quantities for every query, about 20 GB).
  The session logs are computed from them. They are available from the authors on request.
- **The text of the article.** It will be added after publication.
- **Third-party code, models and datasets** (see `THIRD_PARTY_NOTICES.md`).
- The scripts with which the four models in `checkpoints/` were trained.

## The record of predictions

The study wrote numerical predictions into `docs/specificity_workloads_preregistration.md`
before results were analysed. They were not registered with a third party. The history of this
repository shows, for part of the predictions, that the commit which contains a prediction
precedes the commit which contains its result; for the others it does not, and the supplementary
material of the article says which.

    git log --format='%h %ci %s' -- docs/specificity_workloads_preregistration.md

This repository was made from the private working repository of the authors by exporting the
history of the paths listed above and nothing else (`git fast-export` with a list of paths). The
author and committer dates and the commit messages are unchanged. The commit identifiers
differ from those of the working repository, because the other files of that repository are
absent; the prediction file and some commit messages cite the original identifiers, which
`COMMIT_MAP.tsv` translates. The e-mail address of the author was replaced by the no-reply
address of the author's GitHub account. All times are local (UTC+03:30) and were set by the authors'
computer. The commit `771f230` adds the files that the working repository did not track (session
logs of all corpora, model checkpoints) and the files of the release (this file, licences,
notices, a reduced `utils/load_models.py`, the model definition in `surro_models/`).

The later changes of the working repository (its commits `42baffa` to `fe55e1c`: experiment X16,
the evaluation procedure, the final tables) were added in one commit, together with the extracts
`detector_signals.npz`. In this repository, therefore, the history does not show that the
predictions of X16 precede their results; in the working repository they were committed in
`660437a` (2026-09-28 13:50:10 +03:30), before the first session of X16 was generated.

## Reproducing the tables

From the files in this repository, without traces, third-party code or a graphics card:

    pip install -r requirements.txt
    python experiments/gate_trajectory_signatures/make_paper_tables.py
    python experiments/gate_trajectory_signatures/make_main_tables.py
    python experiments/gate_trajectory_signatures/make_prereg_table.py
    python experiments/gate_trajectory_signatures/make_paper_figures.py

The three table scripts reproduce `paper/jisa_2026/tables` and
`paper/jisa_2026/numbers/preregistration_*` byte for byte.

The evaluation procedure under benign-alarm budgets (selection on the calibration split,
one evaluation on the held-out split, intervals from resampling source images) also runs
without the traces, from the session logs and `detector_signals.npz`, and reproduces
`analysis_outputs/operating_profile_r1` byte for byte:

    python experiments/gate_trajectory_signatures/analyze_operating_profile.py --setting cifar10
    python experiments/gate_trajectory_signatures/analyze_operating_profile.py --setting imagenet
    python experiments/gate_trajectory_signatures/test_operating_profile.py

`make_paper_numbers.py`, `make_lfc_schedules.py`, `make_first_alarm_figure.py`,
`export_detector_signals.py` and the other `analyze_*.py` scripts read the traces.

## Regenerating sessions

Sessions are regenerated by the launchers in `scripts/`, after the third-party repositories
have been cloned to the paths given in `third_party/README.md` and the datasets placed under
`data/` (the paths of the authors' machine appear as defaults of some arguments). Each session
logs the SHA-256 digest of the exact bytes of its queries. Regeneration reproduces these digests
on the authors' configuration only: the random streams are drawn on the graphics card, and the
models are evaluated with the reduced-precision convolutions that PyTorch uses by default on
that hardware (NVIDIA RTX 3060 Ti; library versions in `requirements_versions.txt`).

`checkpoints/gtsrb32_resnet18/best_weights_only.pt` is the checkpoint of the study without the
states of optimizer and scheduler, which made the original file larger than the limit of
GitHub. The weights are identical. The SHA-256 digest recorded in the run metadata
(`7a6a7a07...`) is that of the original file; those of the three CIFAR-10 checkpoints match
the files here.

## Use of AI tools

As declared in the article, Claude (Anthropic) assisted in writing and running the experiment
and analysis code, including the independent implementation of the detector of Lee et al., whose
source code was unavailable, and in drafting and editing the text of the article; ChatGPT
(OpenAI) gave editorial feedback on the manuscript. The commits name Claude as co-author.

## Licences

Code: MIT (`LICENSE`). Session logs, result files, model weights and documents: CC BY 4.0
(`LICENSE-DATA.md`). Third-party material: `THIRD_PARTY_NOTICES.md`.

## Contact

Mehdi Ghatee (corresponding author), Department of Mathematics and Computer Science,
Amirkabir University of Technology, Tehran, Iran.

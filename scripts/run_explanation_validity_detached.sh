#!/usr/bin/env bash
# X15, part 2: rule R2 on the library clients (with and without enforcement) and on our own clients.
set -u
cd "$(dirname "$0")/.."
ENF=analysis_outputs/explanation_enforcement_20260927
OUT=analysis_outputs/explanation_validity_20260927
PY=/home/sepi/jupyterenv/bin/python
A=experiments/gate_trajectory_signatures/analyze_explanation_validity.py
mkdir -p "$OUT"
echo "started $(date -Is)" > "$OUT/STATUS"
$PY $A --source-dir analysis_outputs/explanation_clients_20260926 --output-dir "$OUT/own_cifar10" --dataset cifar10 --no-constant > "$OUT/own_cifar10.log" 2>&1
echo "own_cifar10 done $(date -Is) exit $?" >> "$OUT/STATUS"
until grep -q finished "$ENF/STATUS"; do sleep 20; done
$PY $A --source-dir analysis_outputs/explanation_libraries_20260927/cifar10 --enforced-dir "$ENF/cifar10" --output-dir "$OUT/libraries_cifar10" --dataset cifar10 > "$OUT/libraries_cifar10.log" 2>&1
echo "libraries_cifar10 done $(date -Is) exit $?" >> "$OUT/STATUS"
$PY $A --source-dir analysis_outputs/explanation_libraries_20260927/imagenet --enforced-dir "$ENF/imagenet" --output-dir "$OUT/libraries_imagenet" --dataset imagenet > "$OUT/libraries_imagenet.log" 2>&1
echo "libraries_imagenet done $(date -Is) exit $?" >> "$OUT/STATUS"
$PY $A --source-dir analysis_outputs/explanation_clients_imagenet_20260926 --output-dir "$OUT/own_imagenet" --dataset imagenet --no-constant > "$OUT/own_imagenet.log" 2>&1
echo "own_imagenet done $(date -Is) exit $?" >> "$OUT/STATUS"
echo "finished $(date -Is)" >> "$OUT/STATUS"

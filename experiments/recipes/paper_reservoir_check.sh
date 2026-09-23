#!/usr/bin/env bash
# Fairness check: replay baselines with reservoir sampling at an equal byte
# budget. The AO10 appendix showed that the published 'recency' default
# substantially understates ER/DER++ (feature cache P=250: ER 43.23 vs 25.14).
# This re-runs the raw-pipeline 1 MiB cells with reservoir so the equal-byte
# comparison uses the stronger, fairer baseline policy.
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/run/opengl-driver/lib"
export SSL_CERT_FILE="${SSL_CERT_FILE:-/etc/ssl/certs/ca-bundle.crt}"
PY=.venv/bin/python
mkdir -p results/equalbyte_raw

step() { echo; echo "===== $(date '+%F %T') :: $* ====="; }

step "E4-raw reservoir: CIFAR-10 ResNet-18, 1 MiB, seeds 42 1 2"
B=$((1048576 / 12296)); D=$((1048576 / 12336))
for SEED in 42 1 2; do
  $PY experiments/run_benchmark.py --config configs/cifar10_resnet18_frozen_raw.json \
    --device cuda --seed "$SEED" --buffer_sampling reservoir --methods replay \
    --buffer_size "$B" \
    --output_dir "results/equalbyte_raw/c10r18/s${SEED}_b1048576_replay_reservoir" \
    || echo "FAILED reservoir c10 replay seed${SEED}"
  $PY experiments/run_benchmark.py --config configs/cifar10_resnet18_frozen_raw.json \
    --device cuda --seed "$SEED" --buffer_sampling reservoir --methods derpp \
    --buffer_size "$D" \
    --output_dir "results/equalbyte_raw/c10r18/s${SEED}_b1048576_derpp_reservoir" \
    || echo "FAILED reservoir c10 derpp seed${SEED}"
done

step "E4-raw reservoir: CIFAR-100 ResNet-18, 1 MiB, seed 42"
B=$((1048576 / 12296)); D=$((1048576 / 12696))
$PY experiments/run_benchmark.py --config configs/cifar100_resnet18_frozen_raw.json \
  --device cuda --seed 42 --buffer_sampling reservoir --methods replay \
  --buffer_size "$B" \
  --output_dir "results/equalbyte_raw/c100r18/s42_b1048576_replay_reservoir" \
  || echo "FAILED reservoir c100 replay"
$PY experiments/run_benchmark.py --config configs/cifar100_resnet18_frozen_raw.json \
  --device cuda --seed 42 --buffer_sampling reservoir --methods derpp \
  --buffer_size "$D" \
  --output_dir "results/equalbyte_raw/c100r18/s42_b1048576_derpp_reservoir" \
  || echo "FAILED reservoir c100 derpp"

step "RESERVOIR_CHECK_DONE $(date '+%F %T')"

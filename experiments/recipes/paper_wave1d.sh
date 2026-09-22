#!/usr/bin/env bash
# Paper wave 1d: resume after the server restart.
#
# E9 already has its seed-42 cells; this finishes seeds 1/2 plus the E3-fast
# regenerations, then wave 2 runs (repair, raw sweep, MIR, Tiny-ImageNet, slow
# regenerations, latency, domain-shift, drift/appendix).
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/run/opengl-driver/lib"
export SSL_CERT_FILE="${SSL_CERT_FILE:-/etc/ssl/certs/ca-bundle.crt}"
PY=.venv/bin/python

step() { echo; echo "===== $(date '+%F %T') :: $* ====="; }

step "E9 capacity sweep, seeds 1 2 (seed 42 already done)"
for SEED in 1 2; do
  for MAXE in 2 4 6 20; do
    $PY experiments/run_benchmark.py --config configs/cifar100_resnet18_frozen.json \
      --device cuda --seed "$SEED" --methods palmoe --max_experts "$MAXE" --track_routing \
      --output_dir "results/capacity/maxe${MAXE}/seed${SEED}" \
      || echo "FAILED capacity maxe${MAXE} seed${SEED}"
  done
  $PY experiments/run_benchmark.py --config configs/cifar100_resnet18_frozen.json \
    --device cuda --seed "$SEED" --methods replay250,derpp --expert_hidden 4600 \
    --output_dir "results/capacity/parammatched/seed${SEED}" \
    || echo "FAILED capacity parammatched seed${SEED}"
done

step "E3 MNIST 5-seed regeneration with byte accounting"
$PY experiments/run_benchmark_multi.py --seeds "42 1 2 3 4" --device cuda \
  --config configs/mnist_default.json \
  --output_dir results/final_mnist_multiseed || echo "FAILED E3 mnist"

step "E3 CIFAR-10 ResNet-18 3-seed regeneration with byte accounting"
$PY experiments/run_benchmark_multi.py --seeds "42 1 2" --device cuda \
  --config configs/cifar10_resnet18_frozen.json \
  --output_dir results/final_c10r18_multiseed || echo "FAILED E3 c10r18"

step "PAPER_WAVE1D_DONE $(date '+%F %T')"

#!/usr/bin/env bash
# 3-seed error bars for the CIFAR tables (run overnight / on a free GPU).
# Note: the multi-seed driver forwards --pretrain_cache, so seed 42 reuses the
# cached SimCLR weights and only seeds 1/2 pay the pretraining cost.
#
# Expected total runtime: ~1.5-2 h on the 8 GB laptop GPU.
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/run/opengl-driver/lib"
export SSL_CERT_FILE="${SSL_CERT_FILE:-/etc/ssl/certs/ca-bundle.crt}"

echo "=== 1/3: CIFAR-10 ResNet-18 (frozen ImageNet), seeds 42 1 2 ==="
.venv/bin/python experiments/run_benchmark_multi.py \
  --seeds "42 1 2" --device cuda \
  --config configs/cifar10_resnet18_frozen.json \
  --output_dir results/cifar10_resnet18_multiseed

echo "=== 2/3: CIFAR-10 conv frozen, seeds 42 1 2 ==="
.venv/bin/python experiments/run_benchmark_multi.py \
  --seeds "42 1 2" --device cuda --pretrain_cache \
  --config configs/cifar10_big_frozen_full.json \
  --methods palmoe,hybrid,derpp,replay250,icarl \
  --output_dir results/cifar10_conv_multiseed

echo "=== 3/3: CIFAR-100 20-task, seeds 42 1 2 ==="
.venv/bin/python experiments/run_benchmark_multi.py \
  --seeds "42 1 2" --device cuda --pretrain_cache \
  --config configs/cifar100_big_frozen.json \
  --methods palmoe,hybrid,derpp,icarl \
  --output_dir results/cifar100_multiseed

echo "MULTISEED_CIFAR_DONE"

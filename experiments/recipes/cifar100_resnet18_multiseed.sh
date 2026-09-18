#!/usr/bin/env bash
# 3-seed error bars for the strong-backbone 20-task Split-CIFAR-100 recipe
# (frozen ImageNet ResNet-18, feature cache, relative validation gate).
#
# The single-seed evidence (results/cifar100_resnet18/) is the strongest
# 20-task result so far (pure 16.01 / hybrid 18.96); this run adds the
# mean +/- std over seeds 42 1 2 for the README table.
#
# Expected runtime: ~35-45 min on the 8 GB laptop GPU (no pretraining, one
# feature-cache build per seed).
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/run/opengl-driver/lib"
export SSL_CERT_FILE="${SSL_CERT_FILE:-/etc/ssl/certs/ca-bundle.crt}"

.venv/bin/python experiments/run_benchmark_multi.py \
  --seeds "42 1 2" --device cuda \
  --config configs/cifar100_resnet18_frozen.json \
  --methods naive,ewc,replay250,derpp,icarl,palmoe,hybrid \
  --output_dir results/cifar100_resnet18_multiseed

echo "CIFAR100_RESNET18_MULTISEED_DONE"

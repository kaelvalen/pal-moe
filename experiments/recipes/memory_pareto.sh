#!/usr/bin/env bash
# Memory-budget Pareto curve for PAL-MoE v2 (CIFAR-100 + frozen ViT-B/16).
# Pure (zero-raw-replay) and hybrid at three latent-memory budgets; the
# result JSONs carry total/trainable/active params and the prototype footprint,
# so the figure is accuracy (and forgetting) versus stored bytes.
#
# Single seed 42; expected runtime ~15-25 min per budget after the feature
# cache exists (run vit_cifar_quick.sh or vit_cifar_multiseed.sh first, or let
# the first budget build the cache).
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/run/opengl-driver/lib"
export SSL_CERT_FILE="${SSL_CERT_FILE:-/etc/ssl/certs/ca-bundle.crt}"

for SIZE in 250 1000 4000; do
  echo "=== CIFAR-100 ViT-B/16, prototype budget ${SIZE} ==="
  .venv/bin/python experiments/run_benchmark.py \
    --config configs/cifar100_vit.json \
    --proto_size "${SIZE}" \
    --methods palmoe,hybrid \
    --output_dir "results/cifar100_vit_pareto/proto${SIZE}" \
    --device cuda
done

echo "MEMORY_PARETO_DONE"

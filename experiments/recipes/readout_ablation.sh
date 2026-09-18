#!/usr/bin/env bash
# Read-out ablation on the frozen ViT-B/16 stack: learned MoE head (moe) vs
# nearest-class-mean over stored latents (ncm) vs bias-corrected logits (bias),
# for the pure and hybrid variants. Cheap: reuses the persistent feature cache.
#
# Expected runtime: ~10-15 min per head after the CIFAR-10 ViT feature cache
# exists (run vit_cifar_quick.sh first).
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/run/opengl-driver/lib"
export SSL_CERT_FILE="${SSL_CERT_FILE:-/etc/ssl/certs/ca-bundle.crt}"

for HEAD in moe ncm bias; do
  echo "=== CIFAR-10 ViT-B/16, read-out: ${HEAD} ==="
  .venv/bin/python experiments/run_benchmark.py \
    --config configs/cifar10_vit.json \
    --eval_head "${HEAD}" \
    --methods palmoe,hybrid \
    --output_dir "results/cifar10_vit_readout/${HEAD}" \
    --device cuda
done

echo "READOUT_ABLATION_DONE"

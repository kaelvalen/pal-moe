#!/usr/bin/env bash
# S3 driver: extract one projected feature cache per backbone condition, then
# run the S2 ladder on each. The ladder code is shared with S2 on purpose, so
# the two stages cannot drift apart.
set -u
cd "$(dirname "$0")/../.."
source /tmp/opencode/cuda_env.sh 2>/dev/null || true

LEVELS="L0_ncm L1_ridge L1_cosine L2a_shared_joint L2b_shared_seq L3_per_task L4_oracle"
CONDITIONS="random mlp conv resnet18_random resnet18_imagenet vit_b16_imagenet"

.venv/bin/python -u experiments/s3_backbones.py --transfer --device cuda --out results/s3

for name in $CONDITIONS; do
  echo "=== S3 ladder: $name ==="
  .venv/bin/python -u experiments/s2_ladder.py \
    --cache "results/s3/cache_${name}_cifar100/feature_cache.pt" \
    --levels $LEVELS \
    --seeds 42,1,2 --epochs 10 --device cuda \
    --out "results/s3/${name}" --tag "_${name}"
done

echo "S3_DONE"

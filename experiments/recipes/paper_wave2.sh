#!/usr/bin/env bash
# Paper wave 2: repair, the raw-pipeline equal-byte sweep (the honest H2 test),
# external baselines, the serious benchmark and the slow regenerations.
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/run/opengl-driver/lib"
export SSL_CERT_FILE="${SSL_CERT_FILE:-/etc/ssl/certs/ca-bundle.crt}"
PY=.venv/bin/python
mkdir -p results/repair results/equalbyte_raw results/hybrid_raw results/mir \
  results/tinyimagenet_multiseed results/latency results/mnist_domainshift_multiseed

step() { echo; echo "===== $(date '+%F %T') :: $* ====="; }

# ---------------------------------------------------------------- repair
# E1a/E1b ran before the hybrid naming/result-key fixes. Under --feature_cache
# the variant is "pure + latent replay" (no raw store), and it is re-run here
# per seed, merged into the existing per-seed JSONs and re-aggregated. Any
# stale "Hybrid" rows from the intermediate naming are dropped first.
step "REPAIR: latent-replay rows for E1a (CIFAR-10 ViT), seeds 42 1 2"
for SEED in 42 1 2; do
  $PY experiments/run_benchmark.py --config configs/cifar10_vit.json \
    --device cuda --methods hybrid --seed "$SEED" \
    --output_dir "results/repair/c10vit_hybrid/seed${SEED}" || echo "FAILED repair c10vit ${SEED}"
done
$PY experiments/repair_missing_rows.py --target_dir results/cifar10_vit_multiseed \
  --source_dir results/repair/c10vit_hybrid --drop_pattern "PAL-MoE + Replay (Hybrid" --write
$PY experiments/run_benchmark_multi.py --seeds "42 1 2" \
  --output_dir results/cifar10_vit_multiseed --aggregate_only

step "REPAIR: latent-replay rows for E1b (CIFAR-100 ViT), seeds 42 1 2"
for SEED in 42 1 2; do
  $PY experiments/run_benchmark.py --config configs/cifar100_vit.json \
    --device cuda --methods hybrid --seed "$SEED" \
    --output_dir "results/repair/c100vit_hybrid/seed${SEED}" || echo "FAILED repair c100vit ${SEED}"
done
$PY experiments/repair_missing_rows.py --target_dir results/cifar100_vit_multiseed \
  --source_dir results/repair/c100vit_hybrid --drop_pattern "PAL-MoE + Replay (Hybrid" --write
$PY experiments/run_benchmark_multi.py --seeds "42 1 2" \
  --output_dir results/cifar100_vit_multiseed --aggregate_only

# ---------------------------------------------------------------- E4-raw
# Raw-pipeline equal-byte sweep (no feature cache, frozen encoder). This is the
# honest raw-vs-latent storage test: replay stores raw 32x32 images (12296 B),
# latent replay 256-d features (1032 B), PAL pure ~2184 B/prototype, PAL hybrid
# ~14472 B/prototype (latent anchors + one raw image). CIFAR-10 ResNet-18.
run_raw_c10() { # seed budget method extra...
  local SEED=$1 BUDGET=$2 METHOD=$3; shift 3
  $PY experiments/run_benchmark.py --config configs/cifar10_resnet18_frozen_raw.json \
    --device cuda --track_routing --methods "$METHOD" \
    --output_dir "results/equalbyte_raw/c10r18/s${SEED}_b${BUDGET}_${METHOD}" "$@" \
    || echo "FAILED raw c10 s${SEED} b${BUDGET} ${METHOD}"
}
step "E4-raw equal-byte CIFAR-10 ResNet-18, raw pipeline (seed 42, 1 and 4 MiB)"
for BUDGET in 1048576 4194304; do
  B=$((BUDGET / 12296)); D=$((BUDGET / 12336)); K=$((BUDGET / 122960))
  L=$((BUDGET / 1032)); P=$((BUDGET / 2184)); H=$((BUDGET / 14472))
  PS=$((P / 5 * 2)); [ "$PS" -lt 256 ] && PS=256; [ "$PS" -gt 5000 ] && PS=5000
  echo "--- budget ${BUDGET} B: replay=${B} derpp=${D} icarl_k=${K} latent=${L} pure=${P} hybrid=${H} ps=${PS}"
  run_raw_c10 42 "$BUDGET" replay --buffer_size "$B"
  run_raw_c10 42 "$BUDGET" derpp --buffer_size "$D"
  [ "$K" -ge 1 ] && run_raw_c10 42 "$BUDGET" icarl --icarl_k "$K"
  run_raw_c10 42 "$BUDGET" latent_replay --buffer_size "$L"
  run_raw_c10 42 "$BUDGET" palmoe --proto_size "$P" --proto_samples "$PS"
  run_raw_c10 42 "$BUDGET" hybrid --proto_size "$H" --proto_samples "$PS"
done
step "E4-raw equal-byte CIFAR-10 ResNet-18, raw pipeline (seeds 1 2, 1 MiB)"
for SEED in 1 2; do
  BUDGET=1048576
  B=$((BUDGET / 12296)); D=$((BUDGET / 12336)); K=$((BUDGET / 122960))
  L=$((BUDGET / 1032)); P=$((BUDGET / 2184)); H=$((BUDGET / 14472))
  PS=$((P / 5 * 2)); [ "$PS" -lt 256 ] && PS=256
  run_raw_c10 "$SEED" "$BUDGET" replay --buffer_size "$B"
  run_raw_c10 "$SEED" "$BUDGET" derpp --buffer_size "$D"
  [ "$K" -ge 1 ] && run_raw_c10 "$SEED" "$BUDGET" icarl --icarl_k "$K"
  run_raw_c10 "$SEED" "$BUDGET" latent_replay --buffer_size "$L"
  run_raw_c10 "$SEED" "$BUDGET" palmoe --proto_size "$P" --proto_samples "$PS"
  run_raw_c10 "$SEED" "$BUDGET" hybrid --proto_size "$H" --proto_samples "$PS"
done

# CIFAR-100 raw pipeline, seed 42 only (20 tasks, ~3-4 min per method).
run_raw_c100() { # budget method extra...
  local BUDGET=$1 METHOD=$2; shift 2
  $PY experiments/run_benchmark.py --config configs/cifar100_resnet18_frozen_raw.json \
    --device cuda --track_routing --methods "$METHOD" \
    --output_dir "results/equalbyte_raw/c100r18/s42_b${BUDGET}_${METHOD}" "$@" \
    || echo "FAILED raw c100 b${BUDGET} ${METHOD}"
}
step "E4-raw equal-byte CIFAR-100 ResNet-18, raw pipeline (seed 42, 1 MiB)"
BUDGET=1048576
B=$((BUDGET / 12296)); D=$((BUDGET / 12696)); K=$((BUDGET / 1229600))
L=$((BUDGET / 1032)); P=$((BUDGET / 4175)); H=$((BUDGET / 14472))
PS=$((P / 20 * 2)); [ "$PS" -lt 128 ] && PS=128
run_raw_c100 "$BUDGET" replay --buffer_size "$B"
run_raw_c100 "$BUDGET" derpp --buffer_size "$D"
[ "$K" -ge 1 ] && run_raw_c100 "$BUDGET" icarl --icarl_k "$K"
run_raw_c100 "$BUDGET" latent_replay --buffer_size "$L"
run_raw_c100 "$BUDGET" palmoe --proto_size "$P" --proto_samples "$PS"
run_raw_c100 "$BUDGET" hybrid --proto_size "$H" --proto_samples "$PS"

# ---------------------------------------------------------------- E4-hybrid
# True hybrid vs pure in the raw pipeline (default proto_size), the rows the
# feature-cached tables cannot provide. CIFAR-10 + CIFAR-100, seeds 42 1 2.
step "True hybrid vs pure, raw pipeline, CIFAR-10/100 ResNet-18, seeds 42 1 2"
for SEED in 42 1 2; do
  $PY experiments/run_benchmark.py --config configs/cifar10_resnet18_frozen_raw.json \
    --device cuda --methods palmoe,hybrid --track_routing --seed "$SEED" \
    --output_dir "results/hybrid_raw/c10r18/seed${SEED}" || echo "FAILED hybrid_raw c10 ${SEED}"
  $PY experiments/run_benchmark.py --config configs/cifar100_resnet18_frozen_raw.json \
    --device cuda --methods palmoe,hybrid --track_routing --seed "$SEED" \
    --output_dir "results/hybrid_raw/c100r18/seed${SEED}" || echo "FAILED hybrid_raw c100 ${SEED}"
done

# ---------------------------------------------------------------- E12
step "E12 MIR (Maximally Interfered Retrieval), CIFAR-10/100 ResNet-18, seeds 42 1 2"
for SEED in 42 1 2; do
  $PY experiments/run_benchmark.py --config configs/cifar10_resnet18_frozen.json \
    --device cuda --methods mir --seed "$SEED" \
    --output_dir "results/mir/c10r18/seed${SEED}" || echo "FAILED mir c10 ${SEED}"
  $PY experiments/run_benchmark.py --config configs/cifar100_resnet18_frozen.json \
    --device cuda --methods mir --seed "$SEED" \
    --output_dir "results/mir/c100r18/seed${SEED}" || echo "FAILED mir c100 ${SEED}"
done

# ---------------------------------------------------------------- E10
step "E10 Tiny-ImageNet 20x10 class-IL (frozen ImageNet ResNet-18, 64px, shared cache)"
for SEED in 42 1 2; do
  $PY experiments/run_benchmark.py \
    --dataset folder --data_dir data/tiny-imagenet-200/train_flat \
    --classes_per_task 10 --image_size 64 \
    --encoder_arch resnet18 --encoder_weights imagenet --freeze_encoder \
    --feature_cache --feature_cache_dir results/feature_cache/tinyimagenet_r18 \
    --feature_dim 512 --expert_hidden 512 --epochs 5 --num_workers 8 \
    --methods naive,ewc,replay250,derpp,icarl,mir,palmoe,hybrid --track_routing \
    --seed "$SEED" --output_dir "results/tinyimagenet_multiseed/seed${SEED}" \
    --device cuda || echo "FAILED tiny seed ${SEED}"
done

# ---------------------------------------------------------------- E3 (slow)
step "E3 CIFAR-10 conv 3-seed regeneration with byte accounting"
for SEED in 42 1 2; do
  $PY experiments/run_benchmark.py --config configs/cifar10_big_frozen_full.json \
    --device cuda --pretrain_cache \
    --methods palmoe,hybrid,derpp,replay250,icarl --seed "$SEED" \
    --output_dir "results/final_c10conv_multiseed/seed${SEED}" || echo "FAILED c10conv ${SEED}"
done

step "E3 CIFAR-100 conv 3-seed regeneration with byte accounting"
for SEED in 42 1 2; do
  $PY experiments/run_benchmark.py --config configs/cifar100_big_frozen.json \
    --device cuda --pretrain_cache \
    --methods palmoe,hybrid,derpp,icarl --seed "$SEED" \
    --output_dir "results/final_c100conv_multiseed/seed${SEED}" || echo "FAILED c100conv ${SEED}"
done

# ---------------------------------------------------------------- M4
step "M4 latency measurements (ResNet-18 and ViT-B/16 geometries)"
$PY experiments/measure_latency.py --dataset cifar10 --encoder_arch resnet18 \
  --encoder_weights imagenet --feature_dim 256 --expert_hidden 512 \
  --num_experts 1,2,4,6 --batch_sizes 1,128 --device cuda \
  --output results/latency/cifar10_resnet18.json || echo "FAILED latency c10"
$PY experiments/measure_latency.py --dataset cifar100 --encoder_arch resnet18 \
  --encoder_weights imagenet --feature_dim 256 --expert_hidden 512 \
  --num_experts 1,2,4,6 --batch_sizes 1,128 --device cuda \
  --output results/latency/cifar100_resnet18.json || echo "FAILED latency c100"
$PY experiments/measure_latency.py --dataset cifar10 --encoder_arch vit_b_16 \
  --encoder_weights imagenet --feature_dim 768 --expert_hidden 512 \
  --num_experts 1,5 --batch_sizes 1,128 --device cuda \
  --output results/latency/cifar10_vit.json || echo "FAILED latency vit"

# ---------------------------------------------------------------- E11 pilot
step "E11 domain-shift MNIST, 5 seeds (rotate, shared expert, task-free)"
for SEED in 42 1 2 3 4; do
  $PY experiments/run_benchmark.py --config configs/mnist_default.json \
    --device cuda --methods palmoe --domain_shift rotate --shared_expert \
    --freeze_shared_after 2 --task_free_eval --seed "$SEED" \
    --output_dir "results/mnist_domainshift_multiseed/seed${SEED}" || echo "FAILED domainshift ${SEED}"
done

# ---------------------------------------------------------------- E8b
step "E8b trainable-encoder drift cell, CIFAR-10 ResNet-18, seed 42"
$PY experiments/run_benchmark.py --config configs/cifar10_resnet18_trainable.json \
  --device cuda --methods palmoe,hybrid --track_routing --seed 42 \
  --output_dir results/drift/trainable_c10r18/seed42 || echo "FAILED E8b"

# ---------------------------------------------------------------- AO10
step "AO10 buffer-sampling appendix (reservoir vs recency), seed 42"
$PY experiments/run_benchmark.py --config configs/cifar10_resnet18_frozen.json \
  --device cuda --methods replay250,derpp --buffer_sampling reservoir \
  --output_dir results/appendix/reservoir/seed42 || echo "FAILED AO10"
$PY experiments/run_benchmark.py --config configs/cifar10_resnet18_frozen.json \
  --device cuda --methods ewc --ewc_online \
  --output_dir results/appendix/ewc_online/seed42 || echo "FAILED AO10 ewc"

step "PAPER_WAVE2_DONE $(date '+%F %T')"

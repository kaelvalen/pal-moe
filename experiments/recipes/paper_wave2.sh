#!/usr/bin/env bash
# Paper wave 2: external baselines, the serious benchmark and the slow
# regenerations, run after paper_wave1.sh.
#
# E12 MIR baselines -> E10 Tiny-ImageNet -> E3 slow regenerations ->
# M4 latency -> E11 domain-shift pilot (5 seeds).
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/run/opengl-driver/lib"
export SSL_CERT_FILE="${SSL_CERT_FILE:-/etc/ssl/certs/ca-bundle.crt}"
PY=.venv/bin/python
mkdir -p results/mir results/tinyimagenet_multiseed results/latency results/mnist_domainshift_multiseed

step() { echo; echo "===== $(date '+%F %T') :: $* ====="; }

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

# ---------------------------------------------------------------- E4c
# Equal-byte latent replay (single head): one stored latent = 256*4 + 8 label
# bytes = 1032 B, so at equal bytes it can store ~12x more items than raw ER.
run_latent() { # dataset seed budget extra...
  local DATASET=$1 SEED=$2 BUDGET=$3; shift 3
  local CFG="configs/cifar10_resnet18_frozen.json"
  [ "$DATASET" = "c100r18" ] && CFG="configs/cifar100_resnet18_frozen.json"
  local B=$((BUDGET / 1032))
  $PY experiments/run_benchmark.py --config "$CFG" --device cuda \
    --methods latent_replay --buffer_size "$B" \
    --output_dir "results/equalbyte/${DATASET}/s${SEED}_b${BUDGET}_latent_replay" \
    || echo "FAILED latent ${DATASET} s${SEED} b${BUDGET}"
}
step "E4c equal-byte latent replay, CIFAR-10 ResNet-18 (seed 42, four budgets)"
for BUDGET in 262144 1048576 4194304 16777216; do
  run_latent c10r18 42 "$BUDGET"
done
step "E4c equal-byte latent replay, CIFAR-100 ResNet-18 (seed 42, three budgets)"
for BUDGET in 262144 1048576 4194304; do
  run_latent c100r18 42 "$BUDGET"
done
step "E4c equal-byte latent replay, seeds 1 2 at 1 MiB"
for SEED in 1 2; do
  run_latent c10r18 "$SEED" 1048576
  run_latent c100r18 "$SEED" 1048576
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

step "PAPER_WAVE2_DONE $(date '+%F %T')"

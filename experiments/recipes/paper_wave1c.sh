#!/usr/bin/env bash
# Paper wave 1c: resume after the seed-argument fix.
#
# Re-runs the equal-byte cells whose "seeds 1/2" were actually seed-42 repeats,
# then the remaining consolidation runs (E7/E5/E8/E9/E3-fast) with real seeds.
# Wave 1b remains as the from-scratch reference and now passes --seed too.
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/run/opengl-driver/lib"
export SSL_CERT_FILE="${SSL_CERT_FILE:-/etc/ssl/certs/ca-bundle.crt}"
PY=.venv/bin/python

step() { echo; echo "===== $(date '+%F %T') :: $* ====="; }

# ---------------------------------------------------------------- equal-byte
# The s1/s2 dirs hold seed-42 repeats from the buggy run; drop them so the
# re-runs write seed1/seed2 result files.
rm -rf results/equalbyte/c10r18/s1_* results/equalbyte/c10r18/s2_* \
       results/equalbyte/c100r18/s1_* results/equalbyte/c100r18/s2_*
mkdir -p results/equalbyte

run_c10() { # seed budget method extra...
  local SEED=$1 BUDGET=$2 METHOD=$3; shift 3
  $PY experiments/run_benchmark.py --config configs/cifar10_resnet18_frozen.json \
    --device cuda --seed "$SEED" --track_routing --methods "$METHOD" \
    --output_dir "results/equalbyte/c10r18/s${SEED}_b${BUDGET}_${METHOD}" "$@" \
    || echo "FAILED c10 s${SEED} b${BUDGET} ${METHOD}"
}
run_c100() { # seed budget method extra...
  local SEED=$1 BUDGET=$2 METHOD=$3; shift 3
  $PY experiments/run_benchmark.py --config configs/cifar100_resnet18_frozen.json \
    --device cuda --seed "$SEED" --track_routing --methods "$METHOD" \
    --output_dir "results/equalbyte/c100r18/s${SEED}_b${BUDGET}_${METHOD}" "$@" \
    || echo "FAILED c100 s${SEED} b${BUDGET} ${METHOD}"
}

step "E4a real seeds 1 2, CIFAR-10 ResNet-18 feature cache (1 MiB and 4 MiB)"
for SEED in 1 2; do
  for BUDGET in 1048576 4194304; do
    B=$((BUDGET / 1032)); D=$((BUDGET / 1072)); K=$((BUDGET / 10320))
    P=$((BUDGET / 2184))
    PS=$((P / 5 * 2)); [ "$PS" -lt 256 ] && PS=256; [ "$PS" -gt 5000 ] && PS=5000
    run_c10 "$SEED" "$BUDGET" replay --buffer_size "$B"
    run_c10 "$SEED" "$BUDGET" derpp --buffer_size "$D"
    [ "$K" -ge 1 ] && run_c10 "$SEED" "$BUDGET" icarl --icarl_k "$K"
    run_c10 "$SEED" "$BUDGET" palmoe --proto_size "$P" --proto_samples "$PS"
  done
done

step "E4b real seeds 1 2, CIFAR-100 ResNet-18 feature cache (1 MiB)"
for SEED in 1 2; do
  BUDGET=1048576
  B=$((BUDGET / 1032)); D=$((BUDGET / 1432)); K=$((BUDGET / 103200))
  P=$((BUDGET / 4175))
  PS=$((P / 20 * 2)); [ "$PS" -lt 128 ] && PS=128
  run_c100 "$SEED" "$BUDGET" replay --buffer_size "$B"
  run_c100 "$SEED" "$BUDGET" derpp --buffer_size "$D"
  [ "$K" -ge 1 ] && run_c100 "$SEED" "$BUDGET" icarl --icarl_k "$K"
  run_c100 "$SEED" "$BUDGET" palmoe --proto_size "$P" --proto_samples "$PS"
done

# ---------------------------------------------------------------- E7
# Gated vs forced expansion with the same expert cap (20), so the only
# difference is the allocation policy. Re-run all six cells: the old gated
# cells used max_experts 6 and hit the (very slow) prune/merge path.
rm -rf results/growth
run_growth() { # protocol seed extra...
  local PROTO=$1 SEED=$2; shift 2
  $PY experiments/run_benchmark.py --config configs/cifar100_resnet18_frozen.json \
    --device cuda --seed "$SEED" --max_experts 20 --methods palmoe,hybrid --track_routing \
    --output_dir "results/growth/cifar100r18_${PROTO}/seed${SEED}" "$@" \
    || echo "FAILED growth ${PROTO} seed${SEED}"
}
step "E7 gated vs forced expansion, CIFAR-100 ResNet-18, seeds 42 1 2 (max_experts 20)"
for SEED in 42 1 2; do
  run_growth gated "$SEED"
  run_growth forced "$SEED" --expand_every_task
done

# ---------------------------------------------------------------- E5
rm -rf results/ablation_final
run_ablation_cell() { # variant seed extra...
  local VARIANT=$1 SEED=$2; shift 2
  $PY experiments/run_benchmark.py --config configs/cifar10_resnet18_frozen.json \
    --device cuda --seed "$SEED" --track_routing \
    --output_dir "results/ablation_final/${VARIANT}/seed${SEED}" "$@" \
    || echo "FAILED ablation ${VARIANT} seed${SEED}"
}
step "E5 component ablation, CIFAR-10 ResNet-18, seeds 42 1 2"
for SEED in 42 1 2; do
  run_ablation_cell single_nn "$SEED" --methods naive
  run_ablation_cell er_raw "$SEED" --methods replay250
  run_ablation_cell latent_replay "$SEED" --methods latent_replay --buffer_size 250
  run_ablation_cell static_moe "$SEED" --methods palmoe \
    --expand_every_task --max_experts 5 --lambda_r 0 --lambda_e 0 --lambda_ood 0 \
    --router_anchor_steps 0 --joint_calib_epochs 0 --proto_routing_alpha 0
  run_ablation_cell proto_anchors "$SEED" --methods palmoe \
    --expand_every_task --max_experts 5 --lambda_r 0.5 --lambda_e 2.5 --lambda_ood 0 \
    --router_anchor_steps 300 --joint_calib_epochs 5 --proto_routing_alpha 0
  run_ablation_cell gated_moe "$SEED" --methods palmoe \
    --max_experts 5 --lambda_r 0.5 --lambda_e 2.5 --lambda_ood 0 \
    --router_anchor_steps 300 --joint_calib_epochs 5 --proto_routing_alpha 0
  run_ablation_cell full "$SEED" --methods palmoe
done

# ---------------------------------------------------------------- E8
rm -rf results/drift
run_drift() { # cell seed extra...
  local CELL=$1 SEED=$2; shift 2
  $PY experiments/run_benchmark.py --config configs/cifar100_resnet18_frozen.json \
    --device cuda --seed "$SEED" --methods palmoe,hybrid \
    --output_dir "results/drift/${CELL}/seed${SEED}" "$@" \
    || echo "FAILED drift ${CELL} seed${SEED}"
}
step "E8 anchor refresh on/off, CIFAR-100 ResNet-18, seeds 42 1 2"
for SEED in 42 1 2; do
  run_drift refresh_off "$SEED"
  run_drift refresh_on "$SEED" --refresh_anchors_after_calib
done
step "E8 inference anchoring on top (seed 42)"
run_drift refresh_off_alpha 42 --proto_routing_alpha 0.5
run_drift refresh_on_alpha 42 --refresh_anchors_after_calib --proto_routing_alpha 0.5

# ---------------------------------------------------------------- E9
rm -rf results/capacity
step "E9 capacity sweep (max_experts) and param-matched baselines"
for SEED in 42 1 2; do
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

# ---------------------------------------------------------------- E3 (fast)
step "E3 MNIST 5-seed regeneration with byte accounting"
$PY experiments/run_benchmark_multi.py --seeds "42 1 2 3 4" --device cuda \
  --config configs/mnist_default.json \
  --output_dir results/final_mnist_multiseed || echo "FAILED E3 mnist"

step "E3 CIFAR-10 ResNet-18 3-seed regeneration with byte accounting"
$PY experiments/run_benchmark_multi.py --seeds "42 1 2" --device cuda \
  --config configs/cifar10_resnet18_frozen.json \
  --output_dir results/final_c10r18_multiseed || echo "FAILED E3 c10r18"

step "PAPER_WAVE1C_DONE $(date '+%F %T')"

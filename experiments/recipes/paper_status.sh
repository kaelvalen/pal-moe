#!/usr/bin/env bash
# Quick status of the paper run queue (read-only).
#
#     bash experiments/recipes/paper_status.sh
set -u
cd "$(dirname "$0")/../.."

echo "=== running processes ==="
pgrep -af "paper_all|paper_wave|run_benchmark.py|measure_latency" | grep -v pgrep || echo "(none)"

for log in results/paper_run.log results/paper_wave1b.log results/paper_wave2.log; do
  [ -f "$log" ] || continue
  echo
  echo "=== $log ==="
  grep "::" "$log" | tail -6
  echo "FAILED lines: $(grep -c FAILED "$log" 2>/dev/null || echo 0)"
done

echo
echo "=== completed result files per experiment group ==="
for d in equalbyte equalbyte_raw hybrid_raw growth ablation_final drift capacity \
         mir tinyimagenet_multiseed latency mnist_domainshift_multiseed; do
  if [ -d "results/$d" ]; then
    echo "results/$d: $(find "results/$d" -name 'benchmark_results_seed*.json' | wc -l) result files"
  fi
done

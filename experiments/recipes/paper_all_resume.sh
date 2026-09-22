#!/usr/bin/env bash
# Resume the full paper queue after a restart: wave 1d (E9 seeds 1/2 + E3-fast)
# followed by wave 2 (repair, raw equal-byte, MIR, Tiny-ImageNet, slow
# regenerations, latency, domain-shift pilot, drift/appendix).
#
#     nohup bash experiments/recipes/paper_all_resume.sh > results/paper_run.log 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/run/opengl-driver/lib"
export SSL_CERT_FILE="${SSL_CERT_FILE:-/etc/ssl/certs/ca-bundle.crt}"

echo "PAPER_RESUME_START $(date '+%F %T')"
bash experiments/recipes/paper_wave1d.sh
echo "WAVE1D_EXIT=$? $(date '+%F %T')"
bash experiments/recipes/paper_wave2.sh
echo "WAVE2_EXIT=$? $(date '+%F %T')"
echo "PAPER_RESUME_DONE $(date '+%F %T')"

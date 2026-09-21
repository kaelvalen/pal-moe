#!/usr/bin/env bash
# Full paper queue: wave 1b (corrected equal-byte + consolidation) followed by
# wave 2 (repair, raw-pipeline equal-byte, MIR, Tiny-ImageNet, slow
# regenerations, latency, domain-shift pilot, drift/appendix cells).
#
# Run (detached, survives the terminal):
#
#     nohup bash experiments/recipes/paper_all.sh > results/paper_run.log 2>&1 &
#
# Follow progress:
#
#     tail -f results/paper_run.log
#     grep "::" results/paper_wave1b.log results/paper_wave2.log
#
# Expected runtime on the RTX 5060 laptop GPU: ~2.5-3 h (wave 1b) + ~5-6 h
# (wave 2). Failures do not stop the queue; each step logs "FAILED ..." and the
# run continues. Afterwards, regenerate the tables/figures with:
#
#     python experiments/paper_report.py
#
# The run is results-neutral w.r.t. the published defaults: it uses the
# corrected item sizes for the equal-byte sweeps and the raw-pipeline configs
# for the true raw-vs-latent comparison.
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/run/opengl-driver/lib"
export SSL_CERT_FILE="${SSL_CERT_FILE:-/etc/ssl/certs/ca-bundle.crt}"

echo "PAPER_ALL_START $(date '+%F %T')"
bash experiments/recipes/paper_wave1b.sh
echo "WAVE1B_EXIT=$? $(date '+%F %T')"
bash experiments/recipes/paper_wave2.sh
echo "WAVE2_EXIT=$? $(date '+%F %T')"
echo "PAPER_ALL_DONE $(date '+%F %T')"

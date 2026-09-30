#!/usr/bin/env bash
# End-to-end runner for the ICASSP 2027 AEC Agent Track submission.
#
# Prereqs (one-time, on the AutoDL instance):
#   1. Run deploy_autodl.sh from tools/VibeVoice/ to install dependencies.
#   2. Run hk_download/download_all.sh on HK to pull HF models.
#   3. Pull the MMAE benchmark:
#         git clone https://github.com/ddlBoJack/MMAE.git
#         hf download BoJack/MMAE --repo-type dataset --include 'wav/**' \
#             --local-dir ./MMAE
#
# Usage:
#   bash tools/run_agent.sh --meta /path/to/MMAE-meta.json --output outputs/
#
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

python -m agent.router "$@"
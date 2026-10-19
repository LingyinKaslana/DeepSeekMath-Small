#!/usr/bin/env bash
set -euo pipefail
cd .
source activate.sh
exec python -u runs/deepseek_sft_gsm_math_20261015_setup/pipeline.py > runs/deepseek_sft_gsm_math_20261015_setup/pipeline.log 2>&1

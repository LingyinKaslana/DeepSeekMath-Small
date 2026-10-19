#!/usr/bin/env bash
set -euo pipefail
cd .
source activate.sh
python runs/deepseek_coder_1p3b_10b_20261012_setup/control.py --phase training --status running
set +e
python -u runs/deepseek_coder_1p3b_10b_20261012_setup/source/train_cpt.py \
  --backend-config runs/deepseek_coder_1p3b_10b_20261012_setup/backend_config.json \
  --data data/processed/deepseek_coder_1p3b_5b_s42 \
  --output runs/deepseek_coder_1p3b_10b_20261012 \
  --resume ./runs/deepseek_coder_1p3b_500m_20261012/final_checkpoint \
  --training-extension-file runs/deepseek_coder_1p3b_10b_20261012_setup/training_extension.json \
  --steps 305176 --epochs 2 --lr 3e-6 --warmup-steps 763 --lr-schedule constant \
  --seed 42 --log-every 100 --eval-every 1000 --train-probe-blocks 256 \
  --checkpoint-every 10000 --keep-checkpoints 3 \
  > runs/deepseek_coder_1p3b_10b_20261012_setup/training.log 2>&1
extension_train_exit=$?
set -e
if [ "$extension_train_exit" -ne 0 ];then
 python runs/deepseek_coder_1p3b_10b_20261012_setup/control.py --phase training --status failed --exit-code "$extension_train_exit"
 exit "$extension_train_exit"
fi
python runs/deepseek_coder_1p3b_10b_20261012_setup/control.py --phase checkpoint_verification --status running
set +e
python runs/deepseek_coder_1p3b_10b_20261012_setup/source/verify_cpt_checkpoint.py \
 runs/deepseek_coder_1p3b_10b_20261012/final_checkpoint \
 --output runs/deepseek_coder_1p3b_10b_20261012_setup/final_checkpoint_verification.json \
 > runs/deepseek_coder_1p3b_10b_20261012_setup/checkpoint_verification.log 2>&1
extension_verify_exit=$?
set -e
if [ "$extension_verify_exit" -ne 0 ];then
 python runs/deepseek_coder_1p3b_10b_20261012_setup/control.py --phase checkpoint_verification --status failed --exit-code "$extension_verify_exit"
 exit "$extension_verify_exit"
fi
python runs/deepseek_coder_1p3b_10b_20261012_setup/control.py --phase curve_report --status running
set +e
.venv-plot/bin/python runs/deepseek_coder_1p3b_10b_20261012_setup/source/plot_cpt_train_validation.py \
 --run runs/deepseek_coder_1p3b_10b_20261012 \
 --output runs/deepseek_coder_1p3b_10b_20261012_setup/report --batch-window 1000 \
 > runs/deepseek_coder_1p3b_10b_20261012_setup/plot.log 2>&1
extension_plot_exit=$?
set -e
if [ "$extension_plot_exit" -ne 0 ];then
 python runs/deepseek_coder_1p3b_10b_20261012_setup/control.py --phase curve_report --status failed --exit-code "$extension_plot_exit"
 exit "$extension_plot_exit"
fi
python runs/deepseek_coder_1p3b_10b_20261012_setup/control.py --phase finished --status complete_verified --exit-code 0

#!/bin/bash
# Doplnkový experiment M2b (CBAM s identitnou inicializáciou), spustí sa po run_all.sh
cd "$(dirname "$0")"
export HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONWARNINGS=ignore
while pgrep -x run_all.sh >/dev/null; do sleep 30; done
for s in 1 2 3; do nice -n 19 ./venv/bin/python src/train.py --model M2b_ResNet50_CBAM_id --seed $s --exp main; done
echo "EXTRA HOTOVÉ"

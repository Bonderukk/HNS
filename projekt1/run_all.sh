#!/bin/bash
# Všetky experimenty projektu 1 (hotové behy sa preskočia, dá sa spustiť znova po prerušení)
cd "$(dirname "$0")"
export HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONWARNINGS=ignore
PY="nice -n 19 ./venv/bin/python"
# 1) Porovnanie modelov M1–M5, každý 3x (seed 1–3), s augmentáciou, celá sieť dotrénovaná
for s in 1 2 3; do
  for m in M1_ResNet50 M2_ResNet50_CBAM M3_ConvNeXt_T M4_Swin_T M5_CoAtNet_0; do
    $PY src/train.py --model $m --seed $s --exp main
  done
done
# 2) Vplyv augmentácie (ResNet50 bez augmentácie; s augmentáciou = beh z bodu 1)
for s in 1 2 3; do $PY src/train.py --model M1_ResNet50 --seed $s --exp no_aug --no-aug; done
# 3) Rozmrazovanie vrstiev ResNet50: len hlava, postupné rozmrazovanie, trénovanie od nuly
for mode in head gradual scratch; do
  for s in 1 2 3; do $PY src/train.py --model M1_ResNet50 --seed $s --exp unfreeze_$mode --mode $mode; done
done
echo "VŠETKO HOTOVÉ"

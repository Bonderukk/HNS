"""Zadanie 2: detekcia defektov podloziek - YOLO11 (bez / s augmentaciou) a RT-DETR.

    python Zadanie2.py search   hladanie hyperparametrov YOLO (mAP50 na validacnych datach)
    python Zadanie2.py train    finalne trenovanie troch detektorov
    python Zadanie2.py eval     P, R, mAP50 a PR krivky na validacnych a testovacich datach
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import RTDETR, YOLO

ROOT = Path(__file__).parent
DATA = ROOT / "Dataset_Podlozky_Detekcia/TP26_detection-1/data.yaml"
OWN = ROOT / "Testovacie_obrazy_podloziek/data.yaml"   # vlastne nafotene obrazy (ak existuju)
RUNS = ROOT / "runs"

# spolocne hyperparametre vsetkych detektorov (vybrane cez "search")
COMMON = dict(data=str(DATA), imgsz=512, epochs=100, batch=8, optimizer="AdamW", lr0=0.0003,
              seed=0, exist_ok=True, workers=2)   # workers=2: na Windowse sa procesy spustaju pomaly
# augmentacia: predvolena z ultralytics (HSV, posun, mierka, vodorovne preklopenie, mozaika)
# + zvisle preklopenie; NO_AUG vsetko vypina (augmentations=[] vypne aj albumentations)
AUG = dict(flipud=0.5)
NO_AUG = dict(hsv_h=0, hsv_s=0, hsv_v=0, translate=0, scale=0, fliplr=0, mosaic=0, augmentations=[])

MODELS = {  # nazov behu: (trieda, predtrenovane vahy, augmentacia)
    "yolo_noaug": (YOLO, "yolo11s.pt", NO_AUG),
    "yolo_aug": (YOLO, "yolo11s.pt", AUG),
    "rtdetr": (RTDETR, "rtdetr-l.pt", AUG),
}

# hladanie (50 epoch): oproti vychodziemu nastaveniu (yolo11s, AdamW, lr0=0.001, batch=8)
# sa meni vzdy jeden parameter; batch=16 je len pre porovnanie - RT-DETR-l sa s nim
# do 8 GB VRAM nezmesti
SEARCH = {
    "vychodzie": dict(lr0=0.001),
    "model_n": dict(lr0=0.001, model="yolo11n.pt"),
    "model_m": dict(lr0=0.001, model="yolo11m.pt"),
    "SGD_lr0.01": dict(optimizer="SGD", lr0=0.01),
    "lr0.0003": dict(lr0=0.0003),
    "lr0.003": dict(lr0=0.003),
    "batch4": dict(lr0=0.001, batch=4),
    "batch16": dict(lr0=0.001, batch=16),
}


def metrics(m):
    """P, R, mAP50 (priemer cez triedy) a AP50 jednotlivych tried."""
    return dict(P=m.box.mp, R=m.box.mr, mAP50=m.box.map50, mAP50_95=m.box.map,
                **{m.names[c]: ap for c, ap in zip(m.box.ap_class_index, m.box.ap50)})


def search():
    rows = []
    for name, kw in SEARCH.items():
        kw = dict(model="yolo11s.pt", epochs=50) | kw
        m = YOLO(kw.pop("model")).train(**(COMMON | AUG | kw), project=RUNS / "search", name=name)
        rows.append(dict(variant=name) | metrics(m))
        pd.DataFrame(rows).round(3).to_csv(ROOT / "search.csv", index=False)


def train():
    for name, (cls, weights, aug) in MODELS.items():
        cls(weights).train(**COMMON, **aug, project=RUNS, name=name)


def evaluate():
    sets = {"val": dict(data=str(DATA), split="val"), "test": dict(data=str(DATA), split="test")}
    if OWN.exists():
        sets["vlastne"] = dict(data=str(OWN), split="test")
    rows = []
    for name, (cls, _, _) in MODELS.items():
        model = cls(RUNS / name / "weights/best.pt")
        for s, kw in sets.items():   # val() uklada aj BoxPR_curve.png a ukazky predikcii
            m = model.val(**kw, imgsz=512, project=RUNS / "eval", name=f"{name}_{s}", exist_ok=True)
            rows.append(dict(model=name, data=s) | metrics(m))
    df = pd.DataFrame(rows).round(3)
    df.to_csv(ROOT / "results.csv", index=False)
    print(df.to_string(index=False))
    plt.figure(figsize=(8, 4.5))   # priebeh validacneho mAP50 pocas ucenia
    for name in MODELS:
        h = pd.read_csv(RUNS / name / "results.csv")
        plt.plot(h["epoch"], h["metrics/mAP50(B)"], label=name)
    plt.xlabel("epocha"), plt.ylabel("mAP50 (validacne data)"), plt.legend(), plt.grid(alpha=.3)
    plt.savefig(ROOT / "map50_priebeh.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    {"search": search, "train": train, "eval": evaluate}[sys.argv[1]]()

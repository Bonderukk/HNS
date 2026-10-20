"""Trénovanie jedného modelu (transfer learning) + vyhodnotenie na testovacích dátach.

Príklad: python src/train.py --model M1_ResNet50 --seed 1 --exp main
"""
import argparse
import subprocess
import json
import math
import time
from pathlib import Path

import numpy as np
import timm.optim
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import PROJ, GarbageDataset, get_transforms, load_meta
from models import MODELS, build_model, head_and_stages

# Rovnaké parametre trénovania pre všetky modely
EPOCHS = 10
BATCH = 32
LR = 1e-4
WEIGHT_DECAY = 0.05
WARMUP_EPOCHS = 1
LABEL_SMOOTHING = 0.1

# Postupné rozmrazovanie ResNet50: epocha (od 0) -> počet rozmrazených blokov od konca
GRADUAL = {0: 0, 2: 1, 4: 2, 6: 5}  # najprv hlava, potom layer4, layer3, nakoniec celá sieť


def set_trainable(model, mode, epoch):
    """Vráti zoznam zmrazených modulov (tie bežia v eval režime, aby sa nemenili BN štatistiky)."""
    if mode in ("full", "scratch"):
        return []
    head, stages = head_and_stages(model)
    if mode == "head":
        n_open = 0
    else:  # gradual
        n_open = max(v for k, v in GRADUAL.items() if epoch >= k)
    frozen = stages[:len(stages) - n_open]
    for m in stages:
        for p in m.parameters():
            p.requires_grad = m not in frozen
    for m in head:
        for p in m.parameters():
            p.requires_grad = True
    return frozen


def gpu_temp():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,power.draw", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return "?"


def temp_c():
    try:
        return int(gpu_temp().split(",")[0])
    except ValueError:
        return 0


def cool_down(hot=83, cool=75):
    """Ochrana notebooku pred prehriatím: pri vysokej teplote GPU počká, kým nevychladne."""
    if temp_c() < hot:
        return 0.0
    t0 = time.time()
    while temp_c() > cool and time.time() - t0 < 600:
        time.sleep(5)
    return time.time() - t0


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_sum, logits_all, y_all = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(x)
        # rovnaká loss ako pri trénovaní (s label smoothing), aby boli krivky porovnateľné
        loss_sum += F.cross_entropy(out.float(), y, reduction="sum", label_smoothing=LABEL_SMOOTHING).item()
        logits_all.append(out.float().cpu())
        y_all.append(y.cpu())
    logits, y = torch.cat(logits_all), torch.cat(y_all)
    return loss_sum / len(y), (logits.argmax(1) == y).float().mean().item(), logits, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--exp", default="main")
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--mode", default="full", choices=["full", "head", "gradual", "scratch"])
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()

    out = PROJ / "runs" / args.exp / f"{args.model}_s{args.seed}"
    if (out / "result.json").exists():
        print("hotové, preskakujem", out)
        return
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(2)
    device = "cuda"

    meta = load_meta()
    n_cls = len(meta["classes"])
    train_ds = GarbageDataset("train", get_transforms(True, not args.no_aug))
    val_ds = GarbageDataset("val", get_transforms(False, False))
    test_ds = GarbageDataset("test", get_transforms(False, False))
    g = torch.Generator().manual_seed(args.seed)
    train_dl = DataLoader(train_ds, BATCH, shuffle=True, num_workers=4, pin_memory=True,
                          drop_last=True, persistent_workers=True, generator=g)
    val_dl = DataLoader(val_ds, 64, num_workers=4, pin_memory=True, persistent_workers=True)
    test_dl = DataLoader(test_ds, 64, num_workers=4, pin_memory=True)

    model = build_model(args.model, n_cls, pretrained=args.mode != "scratch").to(device)
    opt = timm.optim.create_optimizer_v2(model, "adamw", lr=LR, weight_decay=WEIGHT_DECAY)
    steps = args.epochs * len(train_dl)
    warm = WARMUP_EPOCHS * len(train_dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    scaler = torch.amp.GradScaler()

    hist = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc", "lr", "time"]}
    best_acc, best_ep = -1.0, -1
    t0 = time.time()
    paused = 0.0  # čas čakania na vychladnutie GPU (odpočíta sa z času trénovania)
    for ep in range(args.epochs):
        frozen = set_trainable(model, args.mode, ep)
        model.train()
        for m in frozen:
            m.eval()
        tl, tc, tn = 0.0, 0, 0
        for step, (x, y) in enumerate(train_dl):
            if step % 25 == 0:
                paused += cool_down()
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                outp = model(x)
                loss = F.cross_entropy(outp, y, label_smoothing=LABEL_SMOOTHING)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tl += loss.item() * len(y)
            tc += (outp.argmax(1) == y).sum().item()
            tn += len(y)
        vl, va, _, _ = evaluate(model, val_dl, device)
        hist["train_loss"].append(tl / tn)
        hist["train_acc"].append(tc / tn)
        hist["val_loss"].append(vl)
        hist["val_acc"].append(va)
        hist["lr"].append(opt.param_groups[0]["lr"])
        hist["time"].append(time.time() - t0 - paused)
        if va > best_acc:
            best_acc, best_ep = va, ep
            torch.save(model.state_dict(), out / "best.pt")
        print(f"[{args.exp} {args.model} s{args.seed}] ep {ep + 1}/{args.epochs} "
              f"train {tl / tn:.3f}/{tc / tn:.3f} val {vl:.3f}/{va:.3f} {time.time() - t0:.0f}s (pauza {paused:.0f}s) GPU {gpu_temp()}", flush=True)

    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    tl_, ta, logits, y = evaluate(model, test_dl, device)
    np.savez(out / "test_preds.npz", logits=logits.numpy(), y=y.numpy())
    res = {"model": args.model, "seed": args.seed, "exp": args.exp, "mode": args.mode,
           "augment": not args.no_aug, "epochs": args.epochs, "best_epoch": best_ep + 1,
           "best_val_acc": best_acc, "test_acc": ta, "test_loss": tl_,
           "train_time_s": hist["time"][-1], "history": hist,
           "params_m": sum(p.numel() for p in model.parameters()) / 1e6}
    (out / "result.json").write_text(json.dumps(res, indent=1))
    print(f"TEST acc {ta:.4f} (best val {best_acc:.4f} @ ep {best_ep + 1})", flush=True)


if __name__ == "__main__":
    main()

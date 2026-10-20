"""Štatistické vyhodnotenie všetkých behov, grafy, kontingenčné tabuľky a Grad-CAM.

Výstupy idú do results/ (CSV, JSON a PNG obrázky pre prezentáciu).
"""
import json
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from data import PROJ, GarbageDataset, get_transforms, load_meta, MEAN, STD
from models import MODELS, build_model, gradcam_target

RUNS, OUT = PROJ / "runs", PROJ / "results"
OUT.mkdir(exist_ok=True)
META = load_meta()
CLASSES = META["classes"]
SHORT = {"M1_ResNet50": "M1 ResNet50", "M2_ResNet50_CBAM": "M2 ResNet50+CBAM",
         "M2b_ResNet50_CBAM_id": "M2b CBAM (identita)", "M3_ConvNeXt_T": "M3 ConvNeXt-T",
         "M4_Swin_T": "M4 Swin-T", "M5_CoAtNet_0": "M5 CoAtNet-0"}
COLORS = {"M1_ResNet50": "#2a78d6", "M2_ResNet50_CBAM": "#eb6834", "M2b_ResNet50_CBAM_id": "#6250d6",
          "M3_ConvNeXt_T": "#1baf7a", "M4_Swin_T": "#eda100", "M5_CoAtNet_0": "#e87ba4"}
INK, INK2, GRID, SURF = "#1d2433", "#52514e", "#e4e2dc", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.size": 13,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "text.color": INK, "axes.titlesize": 15, "axes.titleweight": "bold",
    "legend.frameon": False, "lines.linewidth": 2,
})


def load_runs(exp):
    rows = []
    for f in sorted((RUNS / exp).glob("*/result.json")):
        r = json.loads(f.read_text())
        d = np.load(f.parent / "test_preds.npz")
        pred, y = d["logits"].argmax(1), d["y"]
        p, rc, f1, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
        r.update(dir=f.parent, precision=p, recall=rc, f1=f1, pred=pred, y=y)
        rows.append(r)
    return rows


def summarize(rows, key="model"):
    df = pd.DataFrame([{k: r[k] for k in [key, "seed", "test_acc", "precision", "recall", "f1", "best_val_acc",
                                         "best_epoch", "train_time_s", "params_m"]} for r in rows])
    agg = df.groupby(key).agg(["mean", "std"]).drop(columns=["seed"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return df, agg


def agg_dict(d, key):
    a = d.drop(columns=[c for c in ["seed", "label"] if c in d]).groupby(key).agg(["mean", "std"]).round(4)
    a.columns = [f"{x}_{y}" for x, y in a.columns]
    return {str(k): v for k, v in a.to_dict("index").items()}


def best_run(rows, model):
    """Najlepší beh = najvyššia validačná presnosť (testovacie dáta sa pri výbere nepoužívajú)."""
    return max((r for r in rows if r["model"] == model), key=lambda r: (r["best_val_acc"], -r["test_loss"]))


def plot_history(runs, labels, path, title, colors=None):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    colors = colors or [COLORS[r["model"]] for r in runs]
    for r, lab, c in zip(runs, labels, colors):
        h = r["history"]
        ep = np.arange(1, len(h["train_loss"]) + 1)
        l1, = ax[0].plot(ep, h["train_loss"], "--", color=c, label=f"{lab} – trénovanie")
        ax[0].plot(ep, h["val_loss"], "-", color=l1.get_color(), marker="o", ms=5, label=f"{lab} – validácia")
        ax[1].plot(ep, h["train_acc"], "--", color=l1.get_color())
        ax[1].plot(ep, h["val_acc"], "-", color=l1.get_color(), marker="o", ms=5)
    ax[0].set(title="Loss", xlabel="Epocha")
    ax[1].set(title="Accuracy", xlabel="Epocha")
    ax[0].legend(fontsize=11)
    fig.suptitle(title, fontsize=16, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_cm(r, path, title):
    cm = confusion_matrix(r["y"], r["pred"], labels=range(len(CLASSES)))
    fig, ax = plt.subplots(figsize=(8.2, 7))
    ax.grid(False)
    norm = cm / cm.sum(1, keepdims=True)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=11,
                        color="white" if norm[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set(xlabel="Predikovaná trieda", ylabel="Skutočná trieda", title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_model_bars(df, path):
    """Bodový graf: priemer ± smerodajná odchýlka a jednotlivé behy (os nemusí začínať od nuly)."""
    models = [m for m in MODELS if (df.model == m).any()][::-1]
    fig, ax = plt.subplots(figsize=(12, 0.95 * len(models) + 1.2))
    for i, m in enumerate(models):
        v = df[df.model == m].test_acc.values * 100
        ax.errorbar(v.mean(), i, xerr=v.std(ddof=1), fmt="o", ms=11, color=COLORS[m], ecolor=COLORS[m],
                    elinewidth=3, capsize=7, zorder=3)
        ax.scatter(v, np.full(len(v), i) + 0.22, color=INK2, s=20, zorder=3)
        ax.text(v.mean(), i - 0.36, f"{v.mean():.2f} ± {v.std(ddof=1):.2f} %", ha="center", fontsize=12, color=INK)
    ax.set_yticks(range(len(models)), [SHORT[m] for m in models], fontsize=13)
    ax.set_ylim(-0.7, len(models) - 0.4)
    ax.set_xlabel("Test accuracy [%]  (bod = priemer, úsečka = ±1 SD, sivé body = jednotlivé behy)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def per_class(r):
    p, rc, f1, sup = precision_recall_fscore_support(r["y"], r["pred"], labels=range(len(CLASSES)), zero_division=0)
    return pd.DataFrame({"precision": p, "recall": rc, "f1": f1, "support": sup}, index=CLASSES)


def model_stats(df):
    groups = {m: df[df.model == m].test_acc.values for m in MODELS if (df.model == m).sum() >= 2}
    out = {}
    if len(groups) >= 2:
        f, p = stats.f_oneway(*groups.values())
        out["anova"] = {"F": f, "p": p}
        out["pairwise_welch"] = []
        for a, b in combinations(groups, 2):
            t, pv = stats.ttest_ind(groups[a], groups[b], equal_var=False)
            out["pairwise_welch"].append({"a": a, "b": b, "diff_pp": (groups[a].mean() - groups[b].mean()) * 100,
                                          "t": t, "p": pv})
    return out


# ------------------------------------------------------------------ Grad-CAM
def gradcam_figures(main_rows):
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    ds = GarbageDataset("test", get_transforms(False, False))
    models_present = [m for m in MODELS if any(r["model"] == m for r in main_rows)]
    best = {m: best_run(main_rows, m) for m in models_present}
    preds = {m: best[m]["pred"] for m in models_present}
    y = best[models_present[0]]["y"]
    n_wrong = sum((preds[m] != y).astype(int) for m in models_present)
    rng = np.random.default_rng(0)
    # spoločný výber: 3 obrázky správne u všetkých modelov (rôzne triedy) + 3 zle u väčšiny modelov
    easy, used = [], set()
    for i in rng.permutation(np.where(n_wrong == 0)[0]):
        if y[i] not in used:
            easy.append(i)
            used.add(y[i])
        if len(easy) == 3:
            break
    hard = [i for i in np.argsort(-n_wrong, kind="stable") if n_wrong[i] > 0][:3]
    chosen = easy + hard

    def denorm(x):
        return np.clip(x.permute(1, 2, 0).numpy() * np.array(STD) + np.array(MEAN), 0, 1).astype(np.float32)

    cams = {}
    for m in models_present:
        model = build_model(m, len(CLASSES), pretrained=False)
        model.load_state_dict(torch.load(best[m]["dir"] / "best.pt", map_location="cpu"))
        model = model.cuda().eval()
        layer, reshape = gradcam_target(m, model)
        with GradCAM(model=model, target_layers=[layer], reshape_transform=reshape) as cam:
            # vlastný výber pre model: 4 správne a 4 nesprávne klasifikované
            ok = rng.permutation(np.where(preds[m] == y)[0])[:4].tolist()
            bad = rng.permutation(np.where(preds[m] != y)[0])[:4].tolist()
            idx = list(dict.fromkeys(chosen + ok + bad))
            x = torch.stack([ds[i][0] for i in idx]).cuda()
            g = cam(input_tensor=x, targets=[ClassifierOutputTarget(int(preds[m][i])) for i in idx])
        cams[m] = {i: show_cam_on_image(denorm(ds[i][0]), g[k], use_rgb=True) for k, i in enumerate(idx)}
        fig, axes = plt.subplots(2, 4, figsize=(14, 8.4))
        for ax, i in zip(axes.flat, ok + bad):
            ax.imshow(cams[m][i])
            ax.axis("off")
            good = preds[m][i] == y[i]
            ax.set_title(f"skut.: {CLASSES[y[i]]}\npred.: {CLASSES[preds[m][i]]}", fontsize=13,
                         color="#1a7f4b" if good else "#c0392b", fontweight="bold")
        fig.suptitle(f"Grad-CAM – {SHORT[m]}  (horný riadok správne, dolný nesprávne)", fontsize=16,
                     fontweight="bold")
        fig.tight_layout(h_pad=2.5)
        fig.savefig(OUT / f"gradcam_{m}.png", dpi=110)
        plt.close(fig)
        del model
        torch.cuda.empty_cache()

    fig, axes = plt.subplots(len(chosen), len(models_present) + 1, figsize=(2.6 * (len(models_present) + 1),
                                                                          2.75 * len(chosen)))
    for r, i in enumerate(chosen):
        axes[r, 0].imshow(denorm(ds[i][0]))
        axes[r, 0].set_title(f"{CLASSES[y[i]]}", fontsize=13, fontweight="bold")
        for c, m in enumerate(models_present, 1):
            axes[r, c].imshow(cams[m][i])
            good = preds[m][i] == y[i]
            axes[r, c].set_title(("✓ " if good else "✗ ") + CLASSES[preds[m][i]], fontsize=12,
                                 color="#1a7f4b" if good else "#c0392b")
            if r == 0:
                axes[r, c].text(0.5, 1.28, SHORT[m].split(" ", 1)[1], transform=axes[r, c].transAxes,
                                ha="center", fontsize=13, fontweight="bold")
        for a in axes[r]:
            a.axis("off")
    axes[0, 0].text(0.5, 1.28, "Obrázok", transform=axes[0, 0].transAxes, ha="center", fontsize=13,
                    fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "gradcam_compare.png", dpi=110)
    plt.close(fig)


def main():
    report = {}
    main_rows = load_runs("main")
    df, agg = summarize(main_rows)
    df.to_csv(OUT / "main_runs.csv", index=False)
    agg.to_csv(OUT / "main_summary.csv")
    print(agg[["test_acc_mean", "test_acc_std", "precision_mean", "recall_mean", "f1_mean", "train_time_s_mean"]])
    report["main_runs"] = df.to_dict("records")
    report["main_stats"] = model_stats(df)
    plot_model_bars(df, OUT / "model_comparison.png")
    report["best_runs"] = {}
    for m in MODELS:
        if not any(r["model"] == m for r in main_rows):
            continue
        b = best_run(main_rows, m)
        plot_history([b], [SHORT[m]], OUT / f"history_{m}.png", f"{SHORT[m]} – najlepší beh (seed {b['seed']})")
        plot_cm(b, OUT / f"cm_{m}.png", f"{SHORT[m]} – test (seed {b['seed']})")
        pc = per_class(b)
        pc.to_csv(OUT / f"per_class_{m}.csv")
        report["best_runs"][m] = {"seed": b["seed"], "test_acc": b["test_acc"], "precision": b["precision"],
                                  "recall": b["recall"], "f1": b["f1"], "best_epoch": b["best_epoch"],
                                  "per_class": pc.round(4).to_dict("index")}

    # CBAM: štandardná vs. identitná inicializácia
    cb = [m for m in ["M1_ResNet50", "M2_ResNet50_CBAM", "M2b_ResNet50_CBAM_id"] if any(r["model"] == m for r in main_rows)]
    if len(cb) == 3:
        plot_history([best_run(main_rows, m) for m in cb], [SHORT[m] for m in cb], OUT / "cbam_history.png",
                     "CBAM – vplyv inicializácie pozornosti")

    # augmentácia
    na = load_runs("no_aug")
    if na:
        aug = [r for r in main_rows if r["model"] == "M1_ResNet50"]
        d = pd.DataFrame([{"augment": a, "seed": r["seed"], "test_acc": r["test_acc"], "f1": r["f1"],
                           "final_train_acc": r["history"]["train_acc"][-1], "final_val_loss": r["history"]["val_loss"][-1],
                           "min_val_loss": min(r["history"]["val_loss"])}
                          for a, rs in [(True, aug), (False, na)] for r in rs])
        d.to_csv(OUT / "augmentation_runs.csv", index=False)
        report["augmentation"] = agg_dict(d, "augment")
        if len(na) >= 2 and len(aug) >= 2:
            t, p = stats.ttest_ind([r["test_acc"] for r in aug], [r["test_acc"] for r in na], equal_var=False)
            report["augmentation_ttest"] = {"t": t, "p": p}
        ba, bn = best_run(aug, "M1_ResNet50"), best_run(na, "M1_ResNet50")
        plot_history([ba, bn], ["s augmentáciou", "bez augmentácie"], OUT / "augmentation_history.png",
                     "ResNet50 – vplyv augmentácie", colors=["#2a78d6", "#eb6834"])

    # rozmrazovanie vrstiev
    modes = {"head": "Len hlava (zmrazený backbone)", "gradual": "Postupné rozmrazovanie",
             "full": "Celá sieť od začiatku (M1)", "scratch": "Od nuly (bez predtrénovania)"}
    uf = {"full": [r for r in main_rows if r["model"] == "M1_ResNet50"]}
    for mo in ["head", "gradual", "scratch"]:
        uf[mo] = load_runs(f"unfreeze_{mo}")
    if any(uf[mo] for mo in ["head", "gradual", "scratch"]):
        rows = [{"mode": mo, "label": modes[mo], "seed": r["seed"], "test_acc": r["test_acc"], "f1": r["f1"],
                 "train_time_s": r["train_time_s"]} for mo in modes for r in uf[mo]]
        d = pd.DataFrame(rows)
        d.to_csv(OUT / "unfreeze_runs.csv", index=False)
        report["unfreeze"] = agg_dict(d, "mode")
        fig, ax = plt.subplots(figsize=(11, 5.2))
        pal = {"head": "#6250d6", "gradual": "#1baf7a", "full": "#2a78d6", "scratch": "#eb6834"}
        for mo in modes:
            if not uf[mo]:
                continue
            va = np.array([r["history"]["val_acc"] for r in uf[mo]]) * 100
            ep = np.arange(1, va.shape[1] + 1)
            ax.plot(ep, va.mean(0), color=pal[mo], marker="o", ms=5, label=modes[mo])
            ax.fill_between(ep, va.min(0), va.max(0), color=pal[mo], alpha=0.15, lw=0)
        for e, txt in [(3, "+layer4"), (5, "+layer3"), (7, "celá sieť")]:
            ax.axvline(e - 0.5, color="#1baf7a", ls=":", lw=1.2)
            ax.text(e - 0.4, 0.03, txt, color="#137a55", fontsize=12, transform=ax.get_xaxis_transform(), va="bottom")
        ax.set(xlabel="Epocha", ylabel="Validačná accuracy [%]", title="ResNet50 – stratégie rozmrazovania (priemer, rozsah 3 behov)")
        ax.legend(loc="center right", bbox_to_anchor=(1, 0.4))
        fig.tight_layout()
        fig.savefig(OUT / "unfreeze_val_acc.png", dpi=130)
        plt.close(fig)

    (OUT / "report.json").write_text(json.dumps(report, indent=1, default=float))
    if main_rows:
        gradcam_figures(main_rows)
    print("hotovo ->", OUT)


if __name__ == "__main__":
    main()

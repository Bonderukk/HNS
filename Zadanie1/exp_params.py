"""Uloha 1 zadania: vplyv trenovacich parametrov na priebeh ucenia MLP.

Prva struktura siete (128-64) sa trenuje s roznymi nastaveniami optimalizatora,
kroku ucenia a velkosti davky. Priebehy chyby a uspesnosti sa priemeruju cez
niekolko rozdeleni dat, aby zavery neboli o nahode jedneho behu.

Funkcie run_sweep a plot_sweep pouziva aj uloha 2 (exp_structures.py).
"""
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from models import MLP, fit_seeds, summarize

FIGS = Path(__file__).parent / "figs"
BASE = dict(hidden=(128, 64), dropout=0.3, lr=1e-3, batch=32, optimizer="adam", epochs=40)
SEEDS, MARKS = range(3), (10, 15, 20, 40)

SWEEPS = {
    "optimalizator": {"AdamW lr=0.001": {},
                      "SGD+momentum lr=0.001": dict(optimizer="sgdm"),
                      "SGD+momentum lr=0.01": dict(optimizer="sgdm", lr=1e-2)},
    "krok ucenia":   {f"lr={v}": dict(lr=v) for v in (1e-2, 1e-3, 1e-4)},
    "velkost davky": {f"batch={v}": dict(batch=v) for v in (16, 32, 64, 128)},
}


def mean_history(models):
    """Priemerny priebeh cez opakovania: {'tr_loss': pole cez epochy, ...}."""
    return {k: np.mean([[e[k] for e in m.history] for m in models], axis=0)
            for k in models[0].history[0]}


def run_sweep(cls, variants, seeds=SEEDS, **base):
    """Kazdy variant (base | variant) natrenuje na vsetkych rozdeleniach dat.

    Vrati {variant: {'runs': metriky behov, 'summary': min/max/priemer}};
    neuronove siete maju navyse 'history' (priemerny priebeh) a 'params'.
    """
    results = {}
    for label, kw in variants.items():
        models, runs = fit_seeds(partial(cls, **(base | kw)), seeds)
        results[label] = dict(runs=runs, summary=summarize(runs))
        if hasattr(models[0], "net"):
            results[label] |= dict(history=mean_history(models),
                                   params=sum(p.numel() for p in models[0].net.parameters()))
    return results


def plot_sweep(results, title, name, marks=()):
    """Dva grafy: chyba a uspesnost. Plna ciara = validacia, ciarkovana = trening.

    marks - epochy vyznacene zvislou bodkovanou ciarou.
    """
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for label, r in results.items():
        h = r["history"]
        x = np.arange(1, len(h["va_loss"]) + 1)          # epochy cislovane od 1
        color = axs[0].plot(x, h["va_loss"], label=label)[0].get_color()
        axs[0].plot(x, h["tr_loss"], "--", color=color, alpha=.4, linewidth=1)
        axs[1].plot(x, h["va_acc"] * 100, color=color, label=label)
        axs[1].plot(x, h["tr_acc"] * 100, "--", color=color, alpha=.4, linewidth=1)
    for ax, ylab in zip(axs, ("chyba (cross-entropy)", "uspesnost [%]")):
        ax.set(xlabel="epocha", ylabel=ylab)
        ax.grid(alpha=.3)
        for m in marks:
            ax.axvline(m, color="gray", linestyle=":")
    axs[1].set_ylim(0, 100)
    axs[1].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / f"{name}.png", dpi=120)
    plt.close(fig)


def print_epochs(title, results):
    """Validacna uspesnost po 10/15/20/40 epochach - podklad pre volbu poctu epoch."""
    head = f"{title:24s}" + "".join(f"{f'{e} ep.':>10s}" for e in MARKS)
    print(f"\n{head}\n{'-' * len(head)}")
    for label, r in results.items():
        print(f"{label:24s}" + "".join(f"{r['history']['va_acc'][e - 1] * 100:9.1f}%"
                                       for e in MARKS))


if __name__ == "__main__":
    for title, variants in SWEEPS.items():
        results = run_sweep(MLP, variants, **BASE)
        plot_sweep(results, f"MLP 128-64: vplyv parametra '{title}' "
                            f"(priemer z {len(SEEDS)} rozdeleni)",
                   f"param_{title.replace(' ', '_')}")
        print_epochs(title, results)
    print(f"\ngrafy ulozene v {FIGS}")

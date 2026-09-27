"""Uloha 3 a 4 zadania: finalne porovnanie modelov a kontingencne matice.

Kazdy model s konfiguraciou z ulohy 2 (a poctom epoch z exp_epochs.py) sa
natrenuje na 5 rozdeleniach dat. Vypise sa min / priemer / max uspesnosti,
presnosti a senzitivity a nakreslia sa kontingencne matice na testovacich
datach, scitane cez vsetky opakovania.
"""
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from exp_epochs import CHOSEN_EPOCHS
from exp_params import FIGS
from exp_structures import CHOSEN, SEEDS
from models import METRICS, N_CLASSES, fit_seeds, load_split, summarize

LABELS = ["beton", "trava", "strk", "piesok", "dlazba", "tartan"]
FINAL = {cls: cfg | CHOSEN_EPOCHS.get(cls, {}) for cls, cfg in CHOSEN.items()}


def test_confusion(model, seed):
    """Kontingencna matica jedneho modelu na testovacich datach jeho rozdelenia."""
    X, y = load_split(seed, model.source)["test"]
    return confusion_matrix(y, model.predict(X), labels=range(N_CLASSES))


def print_minmax(results):
    """Min / priemer / max kazdej metriky cez opakovania, v %."""
    print(f"{'':13s}" + "".join(f"{n:^21s}" for n in ("uspesnost", "presnost", "senzitivita")))
    print(f"{'model  data':13s}" + f"{'min':>7s}{'priem.':>7s}{'max':>7s}" * 3)
    for name, s in results.items():
        for part in ("val", "test"):
            values = [s[part][m][i] * 100 for m in METRICS for i in (0, 2, 1)]
            print(f"{name:6s} {part:6s}" + "".join(f"{v:7.1f}" for v in values))


def show_confusions(cms):
    """Kontingencne matice ako text do logu a ako obrazok 2x2."""
    fig, axs = plt.subplots(2, 2, figsize=(10, 9))
    for ax, (name, cm) in zip(axs.flat, cms.items()):
        print(f"\n{name} (riadky = skutocny teren, stlpce = predikcia)")
        print(f"{'':8s}" + "".join(f"{label:>8s}" for label in LABELS))
        for label, row in zip(LABELS, cm):
            print(f"{label:8s}" + "".join(f"{v:8d}" for v in row))

        ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(ax=ax, cmap="Blues",
                                                               colorbar=False)
        ax.set(title=f"{name}: uspesnost {np.trace(cm) / cm.sum() * 100:.1f} %",
               xlabel="predikovany teren", ylabel="skutocny teren")
        ax.tick_params(axis="x", labelrotation=45)
    fig.suptitle(f"Kontingencne matice na testovacich datach "
                 f"(sucet {len(SEEDS)} opakovani, {cm.sum(1)[0]} okien na teren)")
    fig.tight_layout()
    fig.savefig(FIGS / "kontingencne_matice.png", dpi=120)
    plt.close(fig)


def main():
    results, cms = {}, {}
    for cls, cfg in FINAL.items():
        print(f"{cls.__name__:5s} {cfg}")
        models, runs = fit_seeds(partial(cls, **cfg), SEEDS)
        results[cls.__name__] = summarize(runs)
        cms[cls.__name__] = sum(test_confusion(m, s) for m, s in zip(models, SEEDS))

    print(f"\nUloha 3: metriky cez {len(SEEDS)} opakovani (v %)")
    print_minmax(results)
    print("(val je nadhodnotena - podla nej sa vyberala struktura aj epocha; "
          "nezavisly odhad je test)")
    show_confusions(cms)


if __name__ == "__main__":
    main()

"""Doplnok k ulohe 2: predlzeny trening vybranej CNN a LSTM.

Krivky z ulohy 2 ukazali, ze CNN (40 epoch) a LSTM (60 epoch) sa este
zlepsovali. Kazdy model sa preto natrenuje raz, na vyrazne viac epoch.
Kratsie treningy netreba spustat zvlast: pri rovnakom seede a konstantnom
kroku ucenia je prvych N epoch dlhsieho behu presne N-epochovy beh.

Pocet epoch sa voli podla vyhladenej validacnej CHYBY, nie podla najlepsej
uspesnosti - najlepsia epocha z dlhsieho useku je lepsia uz len preto,
ze je z coho vyberat.
"""
from functools import partial

import numpy as np

from exp_params import mean_history, plot_sweep
from exp_structures import CHOSEN, SEEDS
from models import CNN, LSTM, fit_seeds

EPOCHS = {CNN: (40, 160), LSTM: (60, 200)}     # (povodny pocet epoch, predlzeny)
STEP, WINDOW = 20, 10                           # krok tabulky, vyhladenie krivky
TOLERANCE = 1.01                                # "na minime" = do 1 % od minima

# Vysledok main() (logs/epochy.txt) zapisany pre ulohu 3.
CHOSEN_EPOCHS = {CNN: dict(epochs=100), LSTM: dict(epochs=80)}


def best_val_upto(models, n):
    """Val uspesnost checkpointu, ktory by vybral n-epochovy trening (po rozdeleniach)."""
    return [max(e["va_acc"] for e in m.history[:n]) for m in models]


def main():
    chosen = {}
    for cls, (old, new) in EPOCHS.items():
        name, cfg = cls.__name__, CHOSEN[cls]
        models, _ = fit_seeds(partial(cls, epochs=new, **cfg), SEEDS)
        h = mean_history(models)
        label = ", ".join(f"{k}={v}" for k, v in cfg.items())
        plot_sweep({label: {"history": h}},
                   f"{name}: predlzeny trening (priemer z {len(SEEDS)} rozdeleni, "
                   f"bodkovane = povodnych {old} epoch)", f"epochy_{name}", marks=(old,))

        marks = range(old, new + 1, STEP)
        smooth = {k: {n: h[k][n - WINDOW:n].mean() for n in marks} for k in h}
        print(f"\n{name} {label} - hodnoty kriviek priemerovane cez poslednych {WINDOW} epoch")
        print(f"{'epochy':>7s}{'tren. chyba':>13s}{'val chyba':>11s}{'val acc':>9s}"
              f"{'najlepsia val do N':>21s}")
        for n in marks:
            print(f"{n:7d}{smooth['tr_loss'][n]:13.3f}{smooth['va_loss'][n]:11.3f}"
                  f"{smooth['va_acc'][n] * 100:8.1f}%"
                  f"{np.mean(best_val_upto(models, n)) * 100:20.1f}%")
        print(f"kontrola s ulohou 2 - najlepsia val do {old}. epochy po rozdeleniach:",
              " ".join(f"{v * 100:.1f}" for v in best_val_upto(models, old)))

        va = smooth["va_loss"]
        best = min(n for n in marks if va[n] <= TOLERANCE * min(va.values()))
        print(f"-> zvolene: {best} epoch (najmensi pocet s vyhladenou val chybou na minime)")
        chosen[cls] = dict(epochs=best)
    assert chosen == CHOSEN_EPOCHS, "vyber sa lisi od konstanty CHOSEN_EPOCHS - aktualizuj ju"


if __name__ == "__main__":
    main()

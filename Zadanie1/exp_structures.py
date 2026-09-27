"""Uloha 2 zadania: porovnanie struktur modelov a vplyvu Dropout vrstvy.

Pre kazdy typ modelu sa porovnaju tri struktury (zakladna + 2 dalsie), na
vybranej strukture potom tri urovne dropoutu. Vsetko je priemer z 5 rozdeleni
dat - tych istych ako v ulohe 3, takze vysledky su priamo porovnatelne.

Vyber sa robi IBA podla validacnej uspesnosti. Testovacie data sa pri
rozhodovani nepouzivaju, inak by testovacia uspesnost bola nadhodnotena.
"""
from exp_params import plot_sweep, run_sweep
from models import CNN, LSTM, MLP, PARTS, SVM, print_table

SEEDS = range(5)

SVM_KERNELS = {"linear":    dict(kernel="linear"),
               "poly st.3": dict(kernel="poly", degree=3, coef0=1),
               "rbf":       dict(kernel="rbf")}

STRUCTURES = {                     # vzdy od najjednoduchsej po najzlozitejsiu
    MLP:  {"64":            dict(hidden=(64,)),
           "128-64":        dict(hidden=(128, 64)),
           "256-128-64":    dict(hidden=(256, 128, 64))},
    CNN:  {"16-32":         dict(channels=(16, 32)),
           "32-64-128":     dict(channels=(32, 64, 128)),
           "32-64-128-256": dict(channels=(32, 64, 128, 256))},
    LSTM: {"64":            dict(hidden=64),
           "128":           dict(hidden=128),
           "2x128":         dict(hidden=128, layers=2)},
}
DROPOUTS = {f"dropout={p}": dict(dropout=p) for p in (0.0, 0.3, 0.5)}

# Vysledok main() (logs/uloha2.txt) zapisany pre dalsie ulohy.
CHOSEN = {SVM:  dict(kernel="rbf"),
          MLP:  dict(hidden=(256, 128, 64), dropout=0.3),
          CNN:  dict(channels=(16, 32), dropout=0.0),
          LSTM: dict(hidden=128, layers=2, dropout=0.0)}


def report(results, title):
    """Tabulka uspesnosti + validacia po rozdeleniach; vrati variant s najlepsou validaciou.

    Pri zhode vyhra variant uvedeny skor, t.j. jednoduchsi.
    """
    names = {k: f"{k} ({r['params']:,} vah)" if "params" in r else k
             for k, r in results.items()}
    print_table({names[k]: r["summary"] for k, r in results.items()}, title,
                parts=PARTS, measures=("acc",))
    w = max(map(len, names.values())) + 2
    print("val acc po rozdeleniach dat:")
    for k, r in results.items():
        print(f"{names[k]:{w}s}" + "".join(f"{run['val']['acc'] * 100:7.1f}" for run in r["runs"]))
    best = max(results, key=lambda k: results[k]["summary"]["val"]["acc"][2])
    print(f"-> vybrane podla validacie: {names[best]}")
    return best


def main():
    chosen = {SVM: SVM_KERNELS[report(run_sweep(SVM, SVM_KERNELS, SEEDS),
                                      "SVM: porovnanie jadier")]}
    for cls, variants in STRUCTURES.items():
        name, n = cls.__name__, len(SEEDS)
        results = run_sweep(cls, variants, SEEDS)
        best = report(results, f"{name}: porovnanie struktur")
        plot_sweep(results, f"{name}: porovnanie struktur (priemer z {n} rozdeleni)",
                   f"struktura_{name}")

        results = run_sweep(cls, DROPOUTS, SEEDS, **variants[best])
        best_dropout = report(results, f"{name} {best}: vplyv dropoutu")
        plot_sweep(results, f"{name} {best}: vplyv dropoutu (priemer z {n} rozdeleni)",
                   f"dropout_{name}")
        chosen[cls] = variants[best] | DROPOUTS[best_dropout]

    print("\n(tabulky v %, priemer z opakovani +- polovica rozpatia min-max)")
    print("\nVybrane konfiguracie pre ulohu 3:")
    for cls, cfg in chosen.items():
        print(f"  {cls.__name__:5s} {cfg}")
    assert chosen == CHOSEN, "vyber sa lisi od konstanty CHOSEN - aktualizuj ju"


if __name__ == "__main__":
    main()

"""Priprava dat pre klasifikaciu terenu (HNS, zadanie 1).

Nacita merania ACC+GYR, nareze ich posuvnym oknom 1 s s posunom 0,5 s,
rozdeli merania na trenovacie/validacne/testovacie a vypocita casove
priznaky pre SVM a MLP. Vysledok ulozi do .npz.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "utils" / "utils"))
from prepare_data import create_sliding_windows      # noqa: E402
from feature_extraction import get_tdf               # noqa: E402

ROOT = Path(__file__).parent / "outdoor_terrain_classification_database"
ROOT = ROOT / "outdoor_terrain_classification_database" / "Measurements"

TERRAINS = ["concrete", "grass", "pebbles", "sand", "paving stone",
            "synthetic running track"]
SENSORS = ("ACC", "GYR")
SPEEDS, SESSIONS = (2, 3), range(1, 8)
FS, WIN, HOP = 400, 400, 200          # 400 Hz -> okno 1 s, posun 0,5 s
SPLIT = (10, 2, 2)                    # meranie na teren: trening/validacia/test

CHANNELS = [f"{s}_{a}" for s in SENSORS for a in "XYZ"]
FEATURES = ["MAV", "NZC", "NSSC", "RMS", "WL", "WAMP", "MAX", "MIN", "PTP"]
FEATURE_NAMES = [f"{f}_{c}" for f in FEATURES for c in CHANNELS]


def load_measurement(terrain, speed, session):
    """Jedno meranie ako pole (vzorky, kanaly); kanaly su osi XYZ snimacov."""
    return np.hstack([np.loadtxt(ROOT / f"{s}_{terrain}_{speed}_{session}.csv",
                                 delimiter=",") for s in SENSORS])


def split_measurements(rng):
    """Nahodne rozdeli 14 merani kazdeho terenu na 10/2/2 podla SPLIT.

    Deli sa na urovni merani, nie okien - susedne okna sa prekryvaju,
    takze delenie az po narezani by prepustilo trenovacie data do testu.
    """
    a, b = SPLIT[0], SPLIT[0] + SPLIT[1]
    parts = ([], [], [])
    for t in range(1, len(TERRAINS) + 1):
        m = np.array([(t, sp, se) for sp in SPEEDS for se in SESSIONS])
        rng.shuffle(m)
        for part, chunk in zip(parts, (m[:a], m[a:b], m[b:])):
            part += chunk.tolist()
    return parts


def build(measurements):
    """Okna a znacky (0-5) pre zoznam merani."""
    X, y = [], []
    for terrain, speed, session in measurements:
        w = create_sliding_windows(load_measurement(terrain, speed, session),
                                   window_size=WIN, step_size=HOP)
        X.append(w)
        y.append(np.full(len(w), terrain - 1))
    return np.concatenate(X), np.concatenate(y)


def extract_features(X):
    """9 casovych priznakov na kanal -> (okna, 9*kanaly).

    Pocita sa funkciou get_tdf z utils, zvlast pre kazde okno a kanal.
    MAV sa dodatocne deli poctom vzoriek - v utils toto delenie chyba,
    vzorec v zadani ho vyzaduje.
    """
    n_win, n_samples, n_ch = X.shape
    F = np.array([[get_tdf(w[:, c]) for c in range(n_ch)] for w in X])
    F[:, :, FEATURES.index("MAV")] /= n_samples
    return F.transpose(0, 2, 1).reshape(n_win, -1)   # poradie: priznak-major


def prepare(seed=0, out_dir="data"):
    """Pripravi jedno rozdelenie dat a ulozi ho do out_dir/terrain_seed<seed>.npz."""
    rng = np.random.default_rng(seed)
    Path(out_dir).mkdir(exist_ok=True)
    arrays = {"feature_names": np.array(FEATURE_NAMES), "classes": np.array(TERRAINS)}

    for name, measurements in zip(("train", "val", "test"), split_measurements(rng)):
        X, y = build(measurements)
        arrays |= {f"X_{name}": X, f"y_{name}": y, f"F_{name}": extract_features(X)}
        print(f"{name:5s}: {len(measurements):3d} merani -> okna {X.shape}, "
              f"priznaky {arrays[f'F_{name}'].shape}, "
              f"na triedu {np.bincount(y, minlength=len(TERRAINS))}")

    path = Path(out_dir) / f"terrain_seed{seed}.npz"
    np.savez_compressed(path, **arrays)
    print(f"ulozene: {path}")
    return path


if __name__ == "__main__":
    import sys
    for seed in (int(s) for s in sys.argv[1:]) if len(sys.argv) > 1 else [0]:
        print(f"--- seed {seed} ---")
        prepare(seed)

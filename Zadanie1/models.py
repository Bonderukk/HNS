"""Modely pre klasifikaciu terenu: SVM, MLP, CNN a LSTM (HNS, zadanie 1).

Vsetky modely maju rovnake rozhranie fit(data)/predict(X), takze metriky,
opakovania aj vyhodnotenie su spolocny kod. Tri neuronove siete zdielaju
jednu trenovaciu slucku (TorchModel) a lisia sa iba metodou _build.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent / "utils" / "utils"))
from train_func import test_step, train_step          # noqa: E402

DATA = Path(__file__).parent / "data"
PARTS = ("train", "val", "test")
METRICS = ("acc", "prec", "rec")
N_CLASSES = 6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------- data ------

def load_split(seed, source="F"):
    """Jedno rozdelenie dat: {'train': (X, y), 'val': ..., 'test': ...}.

    source='F' -> priznaky (SVM, MLP), source='X' -> signal (CNN, LSTM).
    """
    d = np.load(DATA / f"terrain_seed{seed}.npz", allow_pickle=True)
    return {p: (d[f"{source}_{p}"].astype(np.float32), d[f"y_{p}"]) for p in PARTS}


# ------------------------------------------------------------ metriky ------

def metrics(y_true, y_pred):
    """Uspesnost, presnost a senzitivita (makro priemer cez triedy)."""
    kw = dict(average="macro", zero_division=0)
    return {"acc": accuracy_score(y_true, y_pred),
            "prec": precision_score(y_true, y_pred, **kw),
            "rec": recall_score(y_true, y_pred, **kw)}


def evaluate(model, data, parts=("val", "test")):
    return {p: metrics(data[p][1], model.predict(data[p][0])) for p in parts}


# ----------------------------------------------------------------- SVM -----

class SVM:
    """SVM klasifikator nad 54 casovymi priznakmi."""

    source = "F"

    def __init__(self, **kw):
        self.kw = dict(kernel="rbf", C=10, gamma="scale") | kw

    def fit(self, data):
        X, y = data["train"]
        self.scaler = StandardScaler().fit(X)
        self.svc = SVC(**self.kw).fit(self.scaler.transform(X), y)
        return self

    def predict(self, X):
        return self.svc.predict(self.scaler.transform(X))


# -------------------------------------------------- neuronove siete --------

class Transpose(nn.Module):
    """(N, dlzka, kanaly) <-> (N, kanaly, dlzka) - 1D konvolucia chce kanaly druhe."""

    def forward(self, x):
        return x.transpose(1, 2)


class MeanState(nn.Module):
    """Priemer vystupov LSTM cez vsetky casove kroky."""

    def forward(self, x):
        return x[0].mean(1)


OPTIMIZERS = {"adam": torch.optim.AdamW,
              "sgdm": lambda p, lr: torch.optim.SGD(p, lr=lr, momentum=0.9)}

TRAINING = dict(dropout=0.3, lr=1e-3, epochs=40, batch=32, optimizer="adam")


class TorchModel:
    """Spolocna trenovacia slucka. Potomok dodava iba _build(shape).

    Trenovacie kroky su prevzate z utils/train_func.py.
    """

    source = "F"
    defaults = {}          # prepisy trenovacich parametrov a parametre architektury

    def __init__(self, **kw):
        cfg = TRAINING | self.defaults | kw
        unknown = cfg.keys() - TRAINING.keys() - self.defaults.keys()
        if unknown:
            raise TypeError(f"{type(self).__name__}: neznamy parameter {sorted(unknown)}")
        self.arch = {k: v for k, v in cfg.items() if k not in TRAINING}
        self.__dict__.update({k: cfg[k] for k in TRAINING})

    def _build(self, shape):
        raise NotImplementedError

    def _prep(self, X):
        """Standardizacia; pre 3D signal sa normalizuje kazdy kanal zvlast."""
        flat = self.scaler.transform(X.reshape(-1, X.shape[-1]))
        return torch.tensor(flat.reshape(X.shape), dtype=torch.float32)

    def _loader(self, X, y, shuffle):
        return DataLoader(TensorDataset(self._prep(X), torch.tensor(y).long()),
                          batch_size=self.batch, shuffle=shuffle)

    def fit(self, data):
        (Xtr, ytr), (Xva, yva) = data["train"], data["val"]
        self.scaler = StandardScaler().fit(Xtr.reshape(-1, Xtr.shape[-1]))
        tr, va = self._loader(Xtr, ytr, True), self._loader(Xva, yva, False)

        self.net = self._build(Xtr.shape[1:]).to(DEVICE)
        opt = OPTIMIZERS[self.optimizer](self.net.parameters(), lr=self.lr)
        crit = nn.CrossEntropyLoss()

        self.history, best = [], (-1.0, None)
        for e in range(self.epochs):
            tr_acc, tr_loss = train_step(self.net, tr, opt, crit, e, DEVICE)
            va_acc, va_loss, *_ = test_step(self.net, va, opt, crit, e, DEVICE)
            self.history.append(dict(tr_loss=tr_loss, tr_acc=tr_acc,
                                     va_loss=va_loss, va_acc=va_acc))
            if va_acc > best[0]:      # necha si vahy s najlepsou validaciou
                best = (va_acc, {k: v.clone() for k, v in self.net.state_dict().items()})
        self.net.load_state_dict(best[1])
        return self

    def predict(self, X):
        self.net.eval()
        with torch.no_grad():
            return self.net(self._prep(X).to(DEVICE)).argmax(1).cpu().numpy()


class MLP(TorchModel):
    """Plne prepojena siet nad 54 priznakmi."""

    source = "F"
    defaults = dict(hidden=(128, 64))

    def _build(self, shape):
        layers, prev = [], shape[0]
        for h in self.arch["hidden"]:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(self.dropout)]
            prev = h
        return nn.Sequential(*layers, nn.Linear(prev, N_CLASSES))


class CNN(TorchModel):
    """1D konvolucna siet nad surovym signalom (400 vzoriek x 6 kanalov)."""

    source = "X"
    defaults = dict(channels=(32, 64, 128), kernel=7)

    def _build(self, shape):
        k, prev = self.arch["kernel"], shape[1]
        layers = [Transpose()]
        for c in self.arch["channels"]:
            layers += [nn.Conv1d(prev, c, k, padding=k // 2), nn.BatchNorm1d(c),
                       nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(self.dropout)]
            prev = c
        # globalny priemer cez cas -> vystup nezavisi na dlzke okna
        return nn.Sequential(*layers, nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                             nn.Linear(prev, N_CLASSES))


class LSTM(TorchModel):
    """Rekurentna siet nad surovym signalom.

    Signal sa najprv skrati L2 normou po blokoch (RMS obalka vibracii).
    Priemerovanie by tu nefungovalo - vibracia ma nulovu strednu hodnotu,
    takze priemer susednych vzoriek ju iba vyrusi.
    """

    source = "X"
    defaults = dict(hidden=128, layers=1, pool=4, lr=3e-3, epochs=60)

    def _build(self, shape):
        h, layers = self.arch["hidden"], self.arch["layers"]
        return nn.Sequential(
            Transpose(), nn.LPPool1d(2, self.arch["pool"]), Transpose(),
            nn.LSTM(shape[1], h, layers, batch_first=True,
                    dropout=self.dropout if layers > 1 else 0.0),
            MeanState(), nn.Dropout(self.dropout), nn.Linear(h, N_CLASSES))


# ---------------------------------------------------------- opakovania -----

def fit_seeds(factory, seeds=range(5)):
    """Natrenuje model na kazdom rozdeleni dat; vrati (modely, metriky).

    Zdroj dat (priznaky/signal) si urcuje sam model atributom source.
    """
    models, runs = [], []
    for s in seeds:
        torch.manual_seed(s)
        model = factory()
        data = load_split(s, model.source)
        models.append(model.fit(data))
        runs.append(evaluate(model, data, PARTS))
    return models, runs


def run_repeats(factory, seeds=range(5)):
    """Iba metriky opakovanych behov - krok 6 zadania."""
    return fit_seeds(factory, seeds)[1]


def summarize(runs):
    """min / max / priemer kazdej metriky cez opakovania."""
    return {part: {m: (min(r[part][m] for r in runs),
                       max(r[part][m] for r in runs),
                       float(np.mean([r[part][m] for r in runs])))
                   for m in METRICS}
            for part in runs[0]}


def print_table(results, title="", parts=("val", "test"), measures=METRICS):
    """results: {nazov: summarize(...)}. Bunka = priemer +- polovica rozpatia min-max, v %."""
    w = max(map(len, results)) + 2
    cols = [(p, m) for p in parts for m in measures]
    head = f"{'model':{w}s}" + "".join(f"{p + ' ' + m:>16s}" for p, m in cols)
    print(f"\n{title}\n{head}\n{'-' * len(head)}")
    for name, s in results.items():
        print(f"{name:{w}s}" + "".join(
            f"{s[p][m][2] * 100:9.1f} +-{(s[p][m][1] - s[p][m][0]) * 50:4.1f}" for p, m in cols))


if __name__ == "__main__":
    import time
    zoo = {"SVM": SVM, "MLP": MLP, "CNN": CNN, "LSTM": LSTM}
    print(f"zariadenie: {DEVICE}")
    results = {}
    for name, cls in zoo.items():
        t = time.time()
        results[name] = summarize(run_repeats(cls))
        print(f"{name:5s} hotovy za {time.time() - t:5.1f} s")
    print_table(results, "Zakladne struktury, priemer z 5 rozdeleni dat")
    print("(v %, priemer z opakovani +- polovica rozpatia min-max)")

"""Príprava datasetu Garbage Classification v2: stratifikované rozdelenie 80/10/10 a cache v RAM."""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T

DATA_DIR = Path.home() / ".cache/kagglehub/datasets/sumn2u/garbage-classification-v2/versions/12/standardized_256"
PROJ = Path(__file__).resolve().parent.parent
CACHE = PROJ / "cache"
SPLIT_SEED = 42  # rozdelenie je rovnaké pre všetky modely a behy
IMG_SIZE = 224
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def build_cache():
    """Načíta všetky obrázky (256x256) do jedného uint8 poľa a uloží rozdelenie train/val/test."""
    CACHE.mkdir(exist_ok=True)
    classes = sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir())
    files, labels = [], []
    for ci, c in enumerate(classes):
        for f in sorted((DATA_DIR / c).iterdir()):
            files.append(f)
            labels.append(ci)
    labels = np.array(labels)
    imgs = np.lib.format.open_memmap(CACHE / "images.npy", mode="w+", dtype=np.uint8,
                                     shape=(len(files), 256, 256, 3))
    for i, f in enumerate(files):
        im = Image.open(f).convert("RGB")
        if im.size != (256, 256):
            im = im.resize((256, 256), Image.BICUBIC)
        imgs[i] = np.asarray(im)
    imgs.flush()
    idx = np.arange(len(files))
    tr, rest = train_test_split(idx, test_size=0.2, stratify=labels, random_state=SPLIT_SEED)
    va, te = train_test_split(rest, test_size=0.5, stratify=labels[rest], random_state=SPLIT_SEED)
    np.save(CACHE / "labels.npy", labels)
    meta = {"classes": classes, "files": [str(f.relative_to(DATA_DIR)) for f in files],
            "train": tr.tolist(), "val": va.tolist(), "test": te.tolist()}
    (CACHE / "split.json").write_text(json.dumps(meta))
    print(f"{len(files)} obrázkov, train {len(tr)} / val {len(va)} / test {len(te)}")


def load_meta():
    if not (CACHE / "split.json").exists():
        build_cache()
    return json.loads((CACHE / "split.json").read_text())


def get_transforms(train: bool, augment: bool):
    norm = [T.ToDtype(torch.float32, scale=True), T.Normalize(MEAN, STD)]
    if train and augment:
        return T.Compose([
            T.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0), antialias=True),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(p=0.2),
            T.RandomRotation(15),
            T.ColorJitter(0.2, 0.2, 0.2, 0.02),
            *norm,
        ])
    return T.Compose([T.Resize(IMG_SIZE, antialias=True), *norm])


class GarbageDataset(Dataset):
    def __init__(self, split: str, transform):
        meta = load_meta()
        self.idx = np.array(meta[split])
        self.labels = np.load(CACHE / "labels.npy")
        self.images = None  # memmap sa otvára lenivo v každom workeri
        self.transform = transform

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        if self.images is None:
            self.images = np.load(CACHE / "images.npy", mmap_mode="r")
        j = self.idx[i]
        x = torch.from_numpy(np.array(self.images[j])).permute(2, 0, 1)
        return self.transform(x), int(self.labels[j])


if __name__ == "__main__":
    build_cache()

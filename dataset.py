"""Datasets for the two-task pipeline, read straight from the released dataset.

$DATA_ROOT/split.csv (series_uid, patient_id, dataset, split) assigns every
series to train / test / excluded. The validation set is carved out of the
train rows at patient level (`data.val_fraction`, `data.split_seed`), so no
patient is in two of train / val / test; excluded rows are never used.

NoduleFineCropDataset                        [task: nodule]
    CT volume + nodule mask cropped to the series' lung bbox (a CSV written
    by scripts/make_bboxes.py); the transform pipeline then draws patches.

Roi2DDataset                                 [task: roi]
    One sample per axial slice, at native resolution. Keeps every
    ct_2d/'<series_uid>_<NNNN>.npz' whose series is in the split (only series
    with a lung annotation have 2D slices).
"""

import csv
import os
import random
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset


# ── split ─────────────────────────────────────────────────────────────────────

def read_split(cfg):
    """{series_uid: 'train' | 'val' | 'test'} from $DATA_ROOT/split.csv."""
    root = Path(cfg["data"]["root"])
    with open(root / "split.csv", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] in ("train", "test")]

    # NLST patients appear in both NLST label streams: group them as one cohort.
    def patient(r):
        return ("nlst" if r["dataset"].startswith("nlst") else r["dataset"], r["patient_id"])

    train_patients = sorted({patient(r) for r in rows if r["split"] == "train"})
    random.Random(int(cfg["data"].get("split_seed", 42))).shuffle(train_patients)
    n_val = round(len(train_patients) * float(cfg["data"].get("val_fraction", 0.15)))
    val_patients = set(train_patients[:n_val])

    return {r["series_uid"]: ("val" if r["split"] == "train" and patient(r) in val_patients
                              else r["split"])
            for r in rows}


def read_bboxes(path):
    """{series_uid: (h0, h1, w0, w1, d0, d1)} from a make_bboxes.py CSV."""
    with open(path, newline="") as f:
        return {r["series_uid"]: tuple(int(r[k]) for k in ("h0", "h1", "w0", "w1", "d0", "d1"))
                for r in csv.DictReader(f)}


# ── 3D nodule segmentation ────────────────────────────────────────────────────

class NoduleFineCropDataset(Dataset):
    def __init__(self, root, series_uids, bboxes, transform=None):
        """
        Parameters
        ----------
        root        : dataset root with ct_3d/ and nodule_sem_seg_3d/
        series_uids : series to include (from the split)
        bboxes      : {series_uid: (h0, h1, w0, w1, d0, d1)} lung bboxes
        transform   : callable applied to {"image", "label"} dict
        """
        root = Path(root)
        wanted = set(series_uids)
        self.bboxes = bboxes
        self.transform = transform
        self.samples = []
        missing = sorted(wanted - bboxes.keys())
        if missing:
            raise ValueError(f"{len(missing)} series have no lung bbox (e.g. {missing[0]}); "
                             f"regenerate the bbox CSV with scripts/make_bboxes.py")
        for fname in sorted(f for f in os.listdir(root / "ct_3d") if f.endswith(".npz")):
            uid = fname[:-len(".npz")]
            if uid not in wanted:
                continue
            self.samples.append({
                "ct_path":  str(root / "ct_3d" / fname),
                "seg_path": str(root / "nodule_sem_seg_3d" / fname),
                "uid":      uid,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        ct  = np.load(s["ct_path"])["data"].astype(np.float32)[:1]        # (1,H,W,D)
        seg = np.load(s["seg_path"])["data"].astype(np.float32)[:1]       # (1,H,W,D)
        h0, h1, w0, w1, d0, d1 = self.bboxes[s["uid"]]
        sample = {
            "image":    ct [:, h0:h1, w0:w1, d0:d1],
            "label":    seg[:, h0:h1, w0:w1, d0:d1],
            "filename": s["uid"],
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


# ── 2D ROI segmentation ───────────────────────────────────────────────────────

class Roi2DDataset(Dataset):
    """One sample per axial slice, at native resolution. Walks ct_2d/ once at
    construction and keeps every '<series_uid>_<NNNN>.npz' whose series UID is
    in `series_uids`."""

    def __init__(self, root, series_uids, transform=None):
        root = Path(root)
        self.ct_dir  = root / "ct_2d"
        self.roi_dir = root / "roi_sem_seg_2d"
        self.transform = transform
        wanted = set(series_uids)
        # Filename: '<series_uid>_<NNNN>.npz'. Split on the last '_'.
        self.files = [
            p.name for p in sorted(self.ct_dir.glob("*.npz"))
            if p.name.rsplit("_", 1)[0] in wanted
        ]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        ct  = np.load(self.ct_dir  / fname)["data"].astype(np.float32)  # (1, H, W)
        roi = np.load(self.roi_dir / fname)["data"].astype(np.float32)  # (1, H, W)
        sample = {
            "image":    ct,
            "label":    roi,
            "filename": fname,
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


# ── task-aware factory ────────────────────────────────────────────────────────

def build_eval_dataset(task, cfg, val_transform, split_name="val"):
    """Return a single dataset for the requested split ('train' | 'val' | 'test')."""
    if split_name not in ("train", "val", "test"):
        raise ValueError(f"unknown split {split_name!r}")
    root = cfg["data"]["root"]
    uids = [u for u, s in read_split(cfg).items() if s == split_name]
    if task == "nodule":
        return NoduleFineCropDataset(root, uids, read_bboxes(cfg["data"]["bbox_csv"]),
                                     transform=val_transform)
    if task == "roi":
        return Roi2DDataset(root, uids, transform=val_transform)
    raise ValueError(f"unknown task: {task!r}")


def build_datasets(task, cfg, train_transform, val_transform):
    """Return (train_ds, val_ds) for the given task ('roi' or 'nodule')."""
    return (build_eval_dataset(task, cfg, train_transform, "train"),
            build_eval_dataset(task, cfg, val_transform,   "val"))

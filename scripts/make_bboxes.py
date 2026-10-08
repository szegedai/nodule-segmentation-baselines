"""Per-series 3D lung bounding boxes for the nodule models.

The bbox is the extent of the per-slice lung mask (half-open, voxels),
padded by 20 voxels and clipped to the CT. The lung mask comes from:

  --source gt     the released 2D lung masks (roi_sem_seg_2d/) where they
                  exist (NSCLC-Radiomics, NLST), the ROI model otherwise
                  (LIDC-IDRI has no lung masks). Use this for training and
                  for the paper's evaluation protocol.
  --source model  the ROI model for every series — the composite
                  ROI → bbox → nodule pipeline, as on a new scan.

The ROI model runs per axial slice of ct_3d/ with the same sliding-window
inference as eval_roi.py (sigmoid > 0.5).

Usage:
    export DATA_ROOT=/path/to/dataset
    python scripts/make_bboxes.py --config configs/roi_sw.yaml --ckpt <roi ckpt> \\
        --source gt --out outputs/bboxes_gt.csv
    python scripts/make_bboxes.py --config configs/roi_swin_sw.yaml --ckpt <roi ckpt> \\
        --source model --splits test --out outputs/bboxes_roi_swin_sw_test.csv
"""

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataset import read_split                                              # noqa: E402
from eval_metrics_nodule import build_predictor, load_config, load_weights  # noqa: E402
from model import build_model                                               # noqa: E402

PAD = 20
FIELDS = ["series_uid", "source", "h0", "h1", "w0", "w1", "d0", "d1"]


@torch.no_grad()
def predict_lung(predictor, ct, device, amp, batch=16):
    """ct (H, W, D) → per-slice lung mask (D, H, W) bool."""
    slices = torch.from_numpy(ct).permute(2, 0, 1).unsqueeze(1)          # (D, 1, H, W)
    out = []
    for i in range(0, slices.shape[0], batch):
        x = slices[i:i + batch].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
            logits = predictor(x)
        out.append((torch.sigmoid(logits.float()) > 0.5)[:, 0].cpu())
    return torch.cat(out).numpy()


def gt_lung(roi_dir, slice_names):
    """Released per-slice lung masks → (D, H, W) bool."""
    return np.stack([np.load(roi_dir / n)["data"][0] > 0.5 for n in slice_names])


def bbox(mask, shape):
    """(D, H, W) mask, CT shape (H, W, D) → padded half-open (h0, h1, w0, w1, d0, d1)."""
    z = np.flatnonzero(mask.any(axis=(1, 2)))
    h = np.flatnonzero(mask.any(axis=(0, 2)))
    w = np.flatnonzero(mask.any(axis=(0, 1)))
    if not len(z):
        return None
    H, W, D = shape
    return (max(0, int(h[0]) - PAD), min(H, int(h[-1]) + 1 + PAD),
            max(0, int(w[0]) - PAD), min(W, int(w[-1]) + 1 + PAD),
            max(0, int(z[0]) - PAD), min(D, int(z[-1]) + 1 + PAD))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="ROI model config")
    ap.add_argument("--ckpt",   required=True, help="ROI model checkpoint")
    ap.add_argument("--source", required=True, choices=["gt", "model"])
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--out",    required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = Path(cfg["data"]["root"])
    splits = set(args.splits.split(","))
    uids = sorted(u for u, s in read_split(cfg).items() if s in splits)

    out = Path(args.out)
    done = {}
    if out.exists():                                   # resume
        with open(out, newline="") as f:
            done = {r["series_uid"]: r for r in csv.DictReader(f)}
    todo = [u for u in uids if u not in done]
    print(f"{len(uids)} series in {sorted(splits)}, {len(todo)} to do", flush=True)

    slices = defaultdict(list)
    if args.source == "gt":
        for p in sorted((root / "roi_sem_seg_2d").glob("*.npz")):
            slices[p.name.rsplit("_", 1)[0]].append(p.name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and bool(cfg.get("training", {}).get("amp", True))
    model = build_model(cfg["model"]).to(device).eval()
    load_weights(model, args.ckpt)
    predictor = build_predictor(model, cfg)

    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for n, uid in enumerate(todo, 1):
        ct = np.load(root / "ct_3d" / f"{uid}.npz")["data"][0].astype(np.float32)   # (H, W, D)
        if args.source == "gt" and slices[uid]:
            mask, source = gt_lung(root / "roi_sem_seg_2d", slices[uid]), "gt"
        else:
            mask, source = predict_lung(predictor, ct, device, amp), "model"
        bb = bbox(mask, ct.shape)
        if bb is None:
            sys.exit(f"{uid}: no lung found ({source}) — cannot derive a bbox")
        done[uid] = dict(zip(FIELDS, (uid, source, *bb)))
        if n % 10 == 0 or n == len(todo):
            with open(out, "w", newline="") as f:
                w = csv.DictWriter(f, FIELDS, lineterminator="\n")
                w.writeheader()
                w.writerows(done[u] for u in sorted(done))
            print(f"  {n}/{len(todo)}  ({time.time() - t0:.0f} s)", flush=True)
    print(f"wrote {out}  ({len(done)} bboxes)")


if __name__ == "__main__":
    main()

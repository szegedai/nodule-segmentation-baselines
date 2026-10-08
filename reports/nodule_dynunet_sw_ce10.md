# Lung nodule segmentation — DynUNet (3D U-Net) — sliding window 128³, CE weight 10

Paper baseline (Technical Validation, 3D nodule segmentation). Evaluated on the
test split of the released `split.csv`.

## Model and weights

| | |
|---|---|
| Architecture | DynUNet (3D U-Net), 3D, 1 → 2 channels (softmax: background, nodule) |
| Config | `configs/dynunet_sw_ce10.yaml` |
| Weights | [HalmosiL/nodule-dynunet-3d-sw](https://huggingface.co/HalmosiL/nodule-dynunet-3d-sw) @ `0b4fadc90e` |

## Test data

All 308 `test` series of `split.csv` (306 patients,
821 ground-truth nodule components):
LIDC-IDRI 195, NLST (radiologist-corrected) 21, NSCLC-Radiomics 92.

Each CT is cropped to its lung bounding box (+20 voxels) at native 1 mm
resolution and segmented with 128³ sliding windows, 50 % overlap per axis,
Gaussian blending and a per-voxel arg-max. Two bounding-box sources:

- **Paper protocol** — released lung masks for NSCLC-Radiomics and NLST, the
  `roi_sw` SegResNet ROI model for LIDC-IDRI (no lung masks)
  (`results_bboxes/bboxes_gt_test.csv`).
- **Composite** — the `roi_swin_sw` SwinUNETR ROI model for every series, as on
  a new scan (`results_bboxes/bboxes_roi_swin_sw_test.csv`). Nodules outside
  the predicted box are not seen by the nodule model.

Metrics: micro-averaged voxel Dice / precision / recall over the test set,
mean and median per-case Dice, and instance recall (IR) / precision (IP) over
26-connected components, without any minimum-size filtering.

## Results

| Bounding boxes | Dice (micro) | Precision | Recall | Per-case Dice mean | Per-case Dice median | IR | IP | GT components | Predicted components |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Paper protocol | **0.5753** | 0.4422 | 0.8232 | 0.4570 | 0.4969 | 0.8855 | 0.1086 | 821 | 6,876 |
| Composite (ROI → nodule) | **0.5223** | 0.3850 | 0.8116 | 0.4542 | 0.4834 | 0.8867 | 0.0844 | 821 | 8,879 |

Per-case Dice distribution (308 cases):

| | mean | std | min | p05 | p25 | median | p75 | p95 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Paper protocol | 0.4570 | 0.2698 | 0.0000 | 0.0077 | 0.2484 | 0.4969 | 0.6907 | 0.8229 | 0.9371 |
| Composite | 0.4542 | 0.2681 | 0.0000 | 0.0073 | 0.2361 | 0.4834 | 0.6893 | 0.8289 | 0.9371 |

Instance metrics by source (paper protocol):

| Source | Series | GT components | IR | Predicted components | IP |
|---|---:|---:|---:|---:|---:|
| LIDC-IDRI | 195 | 560 | 0.8964 | 3,986 | 0.1264 |
| NLST (radiologist-corrected) | 21 | 162 | 0.8086 | 508 | 0.2579 |
| NSCLC-Radiomics | 92 | 99 | 0.9495 | 2,382 | 0.0470 |

Instance precision is low because every isolated false-positive component
counts, however small; the micro Dice is dominated by large nodules, the
per-case Dice weights every patient equally.

**Training data.** The models were trained on the `train` rows of `split.csv`, with a
patient-level validation subset of the training pool for checkpoint selection; the test
series were used for nothing but this evaluation.

## Training recipe

| Setting | Value |
|---|---|
| Loss | Focal Tversky + weighted CE (α=0.3, β=0.7, γ=2.0, λ_CE=0.1, CE weights bg/nodule = 1.0/10.0) |
| Optimizer | Adam, lr 1e-05, weight decay 1e-05 |
| LR schedule | cosine annealing, T_max = 400, η_min = 1e-06 |
| Patches | 128×128×128, 4 per volume, pos:neg = 2:1 |
| Batch | 2 volumes × 4 patches |
| Epochs | 400 (best validation Dice kept) |
| Seed / precision | 42 / bf16 autocast |

## Reproduce

```bash
bash scripts/download_checkpoints.sh
python scripts/make_bboxes.py --config configs/roi_sw.yaml --ckpt checkpoints/roi_sw/model.pth \
    --source gt --splits test --out outputs/bboxes_gt_test.csv
python eval_metrics_nodule.py      --config configs/dynunet_sw_ce10.yaml --ckpt checkpoints/dynunet_sw_ce10/model.pth \
    --split test --bboxes outputs/bboxes_gt_test.csv
python scripts/instance_metrics.py --config configs/dynunet_sw_ce10.yaml --ckpt checkpoints/dynunet_sw_ce10/model.pth \
    --split test --bboxes outputs/bboxes_gt_test.csv
```
Composite: the same with `make_bboxes.py --config configs/roi_swin_sw.yaml --ckpt checkpoints/roi_swin_sw/model.pth --source model`.
Raw metrics:
`results_eval/dynunet_sw_ce10{,_composite}_test.json`, `results_instance/dynunet_sw_ce10{,_composite}_test.json`.

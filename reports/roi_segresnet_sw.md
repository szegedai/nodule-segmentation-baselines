# Lung ROI segmentation — SegResNet (2D) — sliding window 256², native resolution

Paper baseline (Technical Validation, 2D ROI segmentation). Evaluated on the
test split of the released `split.csv`.

## Model and weights

| | |
|---|---|
| Architecture | SegResNet (MONAI `SegResNet`), 2D, 1 → 1 channel (sigmoid) |
| Config | `configs/roi_sw.yaml` |
| Weights | [HalmosiL/roi-segresnet-2d-sw](https://huggingface.co/HalmosiL/roi-segresnet-2d-sw) @ `33e3e95dfd` |

## Test data

Series of the `test` split that carry a lung annotation — LIDC-IDRI has none,
so its 195 test series are not part of this evaluation:

| Source | Series | Slices |
|---|---:|---:|
| NLST (radiologist-corrected) | 21 | 6,858 |
| NSCLC-Radiomics | 92 | 33,647 |
| **Total** | **113** | **40,505** |

Every axial slice at native 1 mm resolution; 256² sliding windows, 50 % overlap
per axis, Gaussian blending, sigmoid > 0.5 (`eval_roi.py --split test`).

## Results

| Dice (micro) | Precision | Recall | Per-slice Dice mean | Per-slice Dice p05 |
|---:|---:|---:|---:|---:|
| **0.9805** | 0.9816 | 0.9794 | 0.9475 | 0.8268 |

Per-slice Dice distribution (40,505 slices, every slice weighted equally):

| | mean | std | min | p05 | median | p95 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dice | 0.9475 | 0.1650 | 0.0000 | 0.8268 | 0.9881 | 1.0000 | 1.0000 |

Low per-slice scores come from slices at the top and bottom of the lungs, where
the lung is absent or tiny and a few pixels decide the score.

**Training data.** The models were trained on the `train` rows of `split.csv`, with a
patient-level validation subset of the training pool for checkpoint selection; the test
series were used for nothing but this evaluation.

## Training recipe

| Setting | Value |
|---|---|
| Loss | Soft Dice on the sigmoid output (MONAI `DiceLoss`, squared_pred) |
| Optimizer | Adam, lr 0.001, weight decay 1e-05 |
| LR schedule | cosine annealing, T_max = 100, η_min = 1e-06 |
| Patches | 256×256, 4 per slice, pos:neg = 1.0:1, 20,000 sampled slices / epoch |
| Batch | 4 slices × 4 patches |
| Epochs | 100 (best validation Dice kept) |
| Seed / precision | 42 / bf16 autocast |

## Reproduce

```bash
bash scripts/download_checkpoints.sh
python eval_roi.py --config configs/roi_sw.yaml --ckpt checkpoints/roi_sw/model.pth --split test
```
Raw metrics: `results_eval/roi_sw_test.json`.

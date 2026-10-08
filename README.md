# Lung nodule segmentation — paper baselines

Training and evaluation code for the five baseline models of the dataset
paper: two 2D lung (ROI) models and three 3D nodule models. All are trained on
patches at native 1 mm resolution and applied with sliding-window inference.

**Model weights:** [Hugging Face collection *Medical-Image-Segmentation*](https://huggingface.co/collections/HalmosiL/medical-image-segmentation)
— the five models below (`scripts/download_checkpoints.sh` fetches them).

| Task        | Model             | Config                                 | Weights (Hugging Face) |
|-------------|-------------------|----------------------------------------|------------------------|
| ROI (2D)    | SegResNet         | `configs/roi_sw.yaml`                  | [roi-segresnet-2d-sw](https://huggingface.co/HalmosiL/roi-segresnet-2d-sw) |
| ROI (2D)    | SwinUNETR (small) | `configs/roi_swin_sw.yaml`             | [roi-swinunetr-2d-sw](https://huggingface.co/HalmosiL/roi-swinunetr-2d-sw) |
| Nodule (3D) | SegResNet (small) | `configs/segresnet_small_sw_ce10.yaml` | [nodule-segresnet-3d-small-sw](https://huggingface.co/HalmosiL/nodule-segresnet-3d-small-sw) |
| Nodule (3D) | SegResNet (wide)  | `configs/segresnet_wide_sw_ce10.yaml`  | [nodule-segresnet-3d-wide-sw](https://huggingface.co/HalmosiL/nodule-segresnet-3d-wide-sw) |
| Nodule (3D) | DynUNet           | `configs/dynunet_sw_ce10.yaml`         | [nodule-dynunet-3d-sw](https://huggingface.co/HalmosiL/nodule-dynunet-3d-sw) |

## Dependencies

Python 3.10+ and a CUDA GPU (training and full evaluation are impractical on CPU).

```bash
pip install -r requirements.txt     # torch>=2.0, monai==1.4.0, einops, numpy, scipy, pyyaml, tensorboard
```

MONAI must be 1.4.x: later versions removed an argument the SwinUNETR ROI
model uses.

## Data

`DATA_ROOT` is the released dataset; the code reads it directly:

```
$DATA_ROOT/
  split.csv                              series_uid, patient_id, dataset, split (train | test | excluded)
  ct_3d/<series_uid>.npz                 CT volume (1, H, W, D), [0, 1]
  nodule_sem_seg_3d/<series_uid>.npz     nodule mask (1, H, W, D)
  ct_2d/, roi_sem_seg_2d/                axial slices and lung masks (ROI models)
```

- **Test:** the `test` rows of `split.csv`. `excluded` rows are never used.
- **Validation:** 15 % of the `train` patients (seed 42), the same for every
  model, so no patient is in two of train / val / test.
- **Lung bounding boxes:** the nodule models see each CT cropped to its lung
  (+20 voxels). `scripts/make_bboxes.py` derives the box from the released lung
  masks (NSCLC-Radiomics, NLST) or from an ROI model (LIDC-IDRI has no lung masks).

## Reproduce the paper results

Evaluate the published weights on the test split:

```bash
export DATA_ROOT=/path/to/dataset
bash scripts/download_checkpoints.sh      # 5 models → checkpoints/<config>/model.pth (no login needed)

# ROI models
python eval_roi.py --config configs/roi_sw.yaml      --ckpt checkpoints/roi_sw/model.pth      --split test
python eval_roi.py --config configs/roi_swin_sw.yaml --ckpt checkpoints/roi_swin_sw/model.pth --split test

# lung boxes of the test series: released lung masks, roi_sw for LIDC-IDRI
python scripts/make_bboxes.py --config configs/roi_sw.yaml --ckpt checkpoints/roi_sw/model.pth \
    --source gt --splits test --out outputs/bboxes_gt_test.csv

# nodule models (repeat for segresnet_small_sw_ce10 and dynunet_sw_ce10)
M=segresnet_wide_sw_ce10
python eval_metrics_nodule.py      --config configs/$M.yaml --ckpt checkpoints/$M/model.pth --split test --bboxes outputs/bboxes_gt_test.csv
python scripts/instance_metrics.py --config configs/$M.yaml --ckpt checkpoints/$M/model.pth --split test --bboxes outputs/bboxes_gt_test.csv
```

Composite ROI → nodule pipeline: build the boxes with the SwinUNETR ROI model
(`make_bboxes.py --config configs/roi_swin_sw.yaml --ckpt checkpoints/roi_swin_sw/model.pth
--source model --splits test --out outputs/bboxes_roi_swin_sw_test.csv`) and pass
that file to `--bboxes`.

Each script prints its metrics and writes them as JSON next to the checkpoint.
The numbers for the current release are in `reports/`. The published weights
were trained on the `train` rows of `split.csv`; the test series were used only
for evaluation.

## Train from scratch

ROI models first (the nodule bounding boxes of LIDC-IDRI come from one), then
the boxes, then the nodule models:

```bash
python train.py --config configs/roi_sw.yaml
python train.py --config configs/roi_swin_sw.yaml
python scripts/make_bboxes.py --config configs/roi_sw.yaml --ckpt checkpoints/roi_sw/best_model.pth \
    --source gt --out outputs/bboxes_gt.csv
python train.py --config configs/segresnet_small_sw_ce10.yaml     # likewise segresnet_wide_sw_ce10, dynunet_sw_ce10
```

Each run writes `checkpoints/<config>/best_model.pth` (best validation Dice),
`last.pth` (continue with `--resume`) and TensorBoard logs in `runs/<config>/`.
Evaluate them as above with `--ckpt checkpoints/<config>/best_model.pth`.

## Code

| File | Purpose |
|---|---|
| `train.py` | training loop: Adam + cosine LR, bf16, sliding-window validation |
| `dataset.py` | reads `split.csv`; 3D nodule crops and 2D ROI slices |
| `transforms.py` | patch sampling and augmentation |
| `loss.py` | Focal Tversky + CE (nodule), Dice (ROI) |
| `model.py` | SegResNet / SwinUNETR / DynUNet builder |
| `eval_roi.py`, `eval_metrics_nodule.py` | Dice, precision, recall, per-case / per-slice Dice |
| `scripts/instance_metrics.py` | per-nodule recall and precision (26-connected components) |
| `scripts/make_bboxes.py` | lung bounding boxes (released masks or ROI model) |
| `scripts/download_checkpoints.sh` | the five published models, pinned revisions |
| `configs/` | one config per model |
| `reports/`, `results_*/` | test-set results of the published models on the current release |

## License

Apache License 2.0 — see [`LICENSE`](LICENSE).

"""Transforms per task (sliding-window recipe).

Nodule (3D, 2-class softmax):
    Train : RandCropByPosNegLabeld draws `training.patches_per_volume`
            patches per volume, ratio `pos:neg = training.pos_neg_ratio:1`,
            then spatial + intensity augmentation.
    Val   : full volume, no resize; the training / eval loop runs
            `sliding_window_inference`.

ROI (2D):
    Train : positive/negative-balanced `preprocessing.patch_size` crops of the
            native-resolution slice, padded to the patch size.
    Val   : full native-resolution slice.
"""

from monai.transforms import (
    Compose,
    EnsureTyped,
    RandAdjustContrastd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotate90d,
    RandRotated,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandZoomd,
    ResizeWithPadOrCropd,
)

KEYS = ["image", "label"]
_DEFAULT_PATCH_SIZE = (128, 128, 128)


def _patch(preproc_cfg):
    return tuple((preproc_cfg or {}).get("patch_size", list(_DEFAULT_PATCH_SIZE)))


def _rand_pos_neg_crop(patch_size, num_samples, pos, neg):
    return RandCropByPosNegLabeld(
        keys=KEYS, label_key="label",
        spatial_size=patch_size,
        pos=pos, neg=neg, num_samples=num_samples,
        image_key="image", image_threshold=0.0, allow_smaller=True,
    )


def _spatial_augs():
    return [
        RandFlipd(keys=KEYS, prob=0.5, spatial_axis=0),
        RandFlipd(keys=KEYS, prob=0.5, spatial_axis=1),
        RandFlipd(keys=KEYS, prob=0.5, spatial_axis=2),
        RandRotate90d(keys=KEYS, prob=0.5, max_k=3, spatial_axes=(0, 1)),
        RandRotate90d(keys=KEYS, prob=0.5, max_k=3, spatial_axes=(0, 2)),
        RandRotate90d(keys=KEYS, prob=0.5, max_k=3, spatial_axes=(1, 2)),
        RandRotated(keys=KEYS, range_x=0.3, range_y=0.3, range_z=0.3,
                    prob=0.5, mode=["bilinear", "nearest"], padding_mode="zeros"),
        RandZoomd(keys=KEYS, prob=0.4, min_zoom=0.85, max_zoom=1.15,
                  mode=["trilinear", "nearest"], padding_mode="constant", keep_size=True),
    ]


def _intensity_augs():
    return [
        RandScaleIntensityd(keys=["image"], factors=0.2,        prob=0.5),
        RandShiftIntensityd(keys=["image"], offsets=0.15,       prob=0.5),
        RandGaussianNoised( keys=["image"], prob=0.3, mean=0.0, std=0.05),
        RandGaussianSmoothd(keys=["image"], prob=0.2,
                            sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.5, 1.0)),
        RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.7, 1.5), retain_stats=True),
    ]


# ── nodule (3D) ──────────────────────────────────────────────────────────────

def get_nodule_val_transforms(preproc_cfg=None, training_cfg=None):
    return Compose([EnsureTyped(keys=KEYS, dtype="float32", track_meta=False)])


def get_nodule_train_transforms(preproc_cfg=None, training_cfg=None):
    tcfg = training_cfg or {}
    return Compose([
        _rand_pos_neg_crop(
            patch_size  = _patch(preproc_cfg),
            num_samples = int(tcfg.get("patches_per_volume", 4)),
            pos         = float(tcfg.get("pos_neg_ratio", 2.0)),
            neg         = 1.0,
        ),
        *_spatial_augs(),
        *_intensity_augs(),
        EnsureTyped(keys=KEYS, dtype="float32", track_meta=False),
    ])


# ── ROI (2D) ─────────────────────────────────────────────────────────────────

def get_roi_val_transforms(preproc_cfg=None, training_cfg=None):
    return Compose([
        EnsureTyped(keys=KEYS, dtype="float32", track_meta=False),
    ])


def get_roi_train_transforms(preproc_cfg=None, training_cfg=None):
    tcfg = training_cfg or {}
    patch = tuple((preproc_cfg or {}).get("patch_size", [256, 256]))
    return Compose([
        _rand_pos_neg_crop(
            patch_size  = patch,
            num_samples = int(tcfg.get("patches_per_volume", 4)),
            pos         = float(tcfg.get("pos_neg_ratio", 1.0)),
            neg         = 1.0,
        ),
        # some series have slices smaller than the patch (e.g. 254x254);
        # pad them so every patch in the batch is the same size
        ResizeWithPadOrCropd(keys=KEYS, spatial_size=patch),
        EnsureTyped(keys=KEYS, dtype="float32", track_meta=False),
    ])


# ── task-aware factory ───────────────────────────────────────────────────────

def build_transforms(task, cfg):
    preproc  = cfg.get("preprocessing", {}) or {}
    training = cfg.get("training",      {}) or {}
    if task == "nodule":
        return (get_nodule_train_transforms(preproc, training),
                get_nodule_val_transforms(  preproc, training))
    if task == "roi":
        return (get_roi_train_transforms(preproc, training),
                get_roi_val_transforms(  preproc, training))
    raise ValueError(f"unknown task: {task!r}")

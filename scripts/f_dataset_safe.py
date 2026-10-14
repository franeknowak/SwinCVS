"""
Dataset utilities for running SwinCVS inference on, and fine-tuning on, the SAFE dataset.

Kept separate from `scripts/f_dataset.py` on purpose: that module is the
Endoscapes-specific pipeline (fixed stride-5 sequences, `IMAGES_PATH` +
"<vid>_<frame>.jpg" naming) and is left untouched. The SAFE CSV instead
marks the keyframe explicitly via `is_ds_keyframe`, and frames live under a
per-video folder (`VIDEO_ROOT/<vid>/<frame>.jpg`), so sequence construction
and image path resolution both need their own logic.
"""

import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from torchvision import transforms
import torchvision.transforms.functional as TF
import pandas as pd


def build_image_path(video_root, vid, frame):
    """
    SAFE dataset convention: one sub-folder per video, frames named
    "<vid>_<frame>.jpg" inside it (frame is already zero-padded in the csv,
    e.g. "0005") -> VIDEO_ROOT/<vid>/<vid>_<frame>.jpg
    """
    return str(Path(video_root) / str(vid) / f"{vid}_{frame}.jpg")


def build_safe_sequences(df, seq_len=5):
    """
    Shared 5-frame window construction (used by inference and fine-tuning).

    For every `is_ds_keyframe == True` row, take that row and the `seq_len-1`
    rows preceding it *within the same video* (sorted by frame number), and
    keep it only if the frame numbers are exactly contiguous. Keyframes
    without a full contiguous window are skipped with a warning.

    Returns
    -------
    sequences : list of {"vid": str, "frames": [str x seq_len]}
    meta_rows : list of dict, the keyframe csv row (all columns) per sequence
    """
    df = df.copy()
    df["frame"] = df["frame"].astype(str).str.zfill(4)
    df["_frame_int"] = df["frame"].astype(int)

    sequences = []
    meta_rows = []
    skipped = 0

    for vid, vid_df in df.groupby("vid", sort=False):
        vid_df = vid_df.sort_values("_frame_int").reset_index(drop=True)
        keyframe_positions = vid_df.index[vid_df["is_ds_keyframe"] == True].tolist()

        for pos in keyframe_positions:
            if pos < seq_len - 1:
                skipped += 1
                continue

            window = vid_df.iloc[pos - (seq_len - 1) : pos + 1]
            frame_ints = window["_frame_int"].tolist()
            # Require exactly consecutive frame numbers for a valid sequence
            if frame_ints != list(range(frame_ints[0], frame_ints[0] + seq_len)):
                skipped += 1
                continue

            frames = window["frame"].tolist()
            sequences.append({"vid": vid, "frames": frames})
            meta_rows.append(vid_df.iloc[pos].drop(labels=["_frame_int"]).to_dict())

    if skipped:
        print(f"Warning: skipped {skipped} keyframe(s) without a full {seq_len}-frame preceding window.")

    return sequences, meta_rows


def build_safe_transform(resize, center_crop, mean, std):
    """SAFE preprocessing: Resize -> CenterCrop -> ToTensor -> Normalize (min-max applied per image afterwards)."""
    return transforms.Compose(
        [
            transforms.Resize(tuple(resize)),
            transforms.CenterCrop(center_crop),
            transforms.ToTensor(),
            transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
        ]
    )


def minmax_scale(image):
    """Per-image min-max scaling applied after Normalize (as in the original pipeline)."""
    return (image - torch.min(image)) / (-torch.min(image) + torch.max(image))


def get_safe_inference_dataset(config):
    """
    Build 5-frame sequences ending on each `is_ds_keyframe == True` row.

    Unlike the Endoscapes pipeline (fixed stride of 5), the SAFE csv marks
    keyframes explicitly, so for every keyframe row we look back at the 4
    preceding rows *within the same video* and require the frame numbers to
    be contiguous (idx-4 .. idx). Keyframes without 4 valid preceding frames
    (e.g. at the very start of a video, or after a gap in the frame numbers)
    are skipped with a warning and dropped from the outputs.

    Returns
    -------
    dataset : SafeSwinCVS_Inference_Dataset
    meta_df : pd.DataFrame
        One row per usable sequence, carrying every original csv column for
        the keyframe row (so results can be re-merged 1:1), in the same
        order the dataset yields sequences.
    """
    csv_path = Path(config.CSV_PATH)
    if not (str(csv_path).endswith(".csv") and csv_path.is_file()):
        raise FileNotFoundError(f"Invalid CSV Path: {config.CSV_PATH}")

    df = pd.read_csv(csv_path)
    sequences, meta_rows = build_safe_sequences(df)
    meta_df = pd.DataFrame(meta_rows).reset_index(drop=True)

    transform_sequence = build_safe_transform(
        config.RESIZE, config.CENTER_CROP, config.ENDOSCAPES_MEAN, config.ENDOSCAPES_STD
    )

    dataset = SafeSwinCVS_Inference_Dataset(
        sequences, transform_sequence, video_root=config.VIDEO_ROOT
    )

    return dataset, meta_df


class SafeSwinCVS_Inference_Dataset(Dataset):
    """5-frame sequence dataset for SAFE-format inference (no labels required)."""

    def __init__(self, sequences, transform_sequence, video_root):
        self.sequences = sequences
        self.transforms = transform_sequence
        self.video_root = video_root

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        vid = seq["vid"]
        paths = [build_image_path(self.video_root, vid, frame) for frame in seq["frames"]]

        image_list = []
        for path in paths:
            image = Image.open(path).convert("RGB")
            image = self.transforms(image)
            image = minmax_scale(image)
            image_list.append(image)

        images = torch.stack(image_list)
        return images


def get_safe_inference_dataloader(config, dataset, batch_size=None, num_workers=0):
    """
    num_workers defaults to 0 so that preprocessing runs synchronously on the
    main thread - required for accurate end-to-end (pre-to-post-process)
    latency measurement in inference_safe.py.
    """
    bs = batch_size if batch_size is not None else config.BATCH_SIZE
    return DataLoader(
        dataset,
        batch_size=bs,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
    )

##############################################################################################
# SAFE fine-tuning (labelled) datasets
##############################################################################################

SAFE_TARGET_COLUMNS = ["C1", "C2", "C3"]


def get_safe_split_csv(config, split):
    """labels/<FPS>/splits/<split>.csv under DATA.ROOT (overridable per split via DATA.<SPLIT>_CSV)."""
    override = config.DATA.get(f"{split.upper()}_CSV", None)
    if override:
        return Path(override)
    return Path(config.DATA.ROOT) / "labels" / config.DATA.FPS / "splits" / f"{split}.csv"


def load_safe_split_df(config, split):
    """
    Read a split csv. If it does not carry `is_ds_keyframe` (or the labels),
    join those columns from labels/<FPS>/metadata.csv on (vid, frame).
    """
    csv_path = get_safe_split_csv(config, split)
    if not csv_path.is_file():
        raise FileNotFoundError(f"SAFE {split} split csv not found: {csv_path}")
    df = pd.read_csv(csv_path, dtype={"frame": str})
    df["frame"] = df["frame"].astype(str).str.zfill(4)

    missing = [c for c in ["is_ds_keyframe"] + SAFE_TARGET_COLUMNS if c not in df.columns]
    if missing:
        meta_path = Path(config.DATA.ROOT) / "labels" / config.DATA.FPS / "metadata.csv"
        print(f"{split}.csv lacks {missing}; joining from {meta_path}")
        meta = pd.read_csv(meta_path, dtype={"frame": str})
        meta["frame"] = meta["frame"].astype(str).str.zfill(4)
        df = df.merge(meta[["vid", "frame"] + missing], on=["vid", "frame"], how="left")
        df["is_ds_keyframe"] = df["is_ds_keyframe"].fillna(False)
    return df


def build_safe_train_augmentation(config):
    """Returns a SequenceAugmentation or None (if DATA.AUGMENT.ENABLE is False)."""
    aug = config.DATA.AUGMENT
    if not aug.ENABLE:
        return None
    return SequenceAugmentation(
        hflip_prob=aug.HFLIP_PROB,
        brightness=aug.BRIGHTNESS,
        contrast=aug.CONTRAST,
        saturation=aug.SATURATION,
        hue=aug.HUE,
    )


class SequenceAugmentation:
    """
    Light augmentation whose random parameters are drawn ONCE per sequence and
    applied identically to all frames (so temporal consistency is preserved).
    Operates on [0, 1] tensors after ToTensor and before Normalize.

    Note: the per-image min-max scaling that follows Normalize largely cancels
    global brightness/contrast changes; saturation/hue and flips survive it.
    """

    def __init__(self, hflip_prob=0.5, brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0):
        self.hflip_prob = hflip_prob
        self.jitter = transforms.ColorJitter(brightness, contrast, saturation, hue)

    def sample_params(self):
        flip = bool(torch.rand(1).item() < self.hflip_prob)
        fn_idx, b, c, s, h = transforms.ColorJitter.get_params(
            self.jitter.brightness, self.jitter.contrast, self.jitter.saturation, self.jitter.hue
        )
        return flip, (fn_idx, b, c, s, h)

    @staticmethod
    def apply(image, params):
        flip, (fn_idx, b, c, s, h) = params
        if flip:
            image = TF.hflip(image)
        for fn_id in fn_idx:
            if fn_id == 0 and b is not None:
                image = TF.adjust_brightness(image, b)
            elif fn_id == 1 and c is not None:
                image = TF.adjust_contrast(image, c)
            elif fn_id == 2 and s is not None:
                image = TF.adjust_saturation(image, s)
            elif fn_id == 3 and h is not None:
                image = TF.adjust_hue(image, h)
        return image


class SafeSwinCVS_Dataset(Dataset):
    """
    Labelled 5-frame SAFE sequences for fine-tuning: returns (images[5,3,H,W], targets[3]).
    Preprocessing matches SAFE inference exactly (Resize -> CenterCrop -> ToTensor ->
    Normalize -> per-image min-max); the optional augmentation is inserted between
    ToTensor and Normalize with parameters shared across the sequence.
    """

    def __init__(self, sequences, targets, config, augmentation=None):
        self.sequences = sequences
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.video_root = Path(config.DATA.ROOT) / "images" / config.DATA.FPS
        self.augmentation = augmentation
        t = config.DATA.TRANSFORMS
        self.pre = transforms.Compose(
            [transforms.Resize(tuple(t.RESIZE)), transforms.CenterCrop(t.CENTER_CROP), transforms.ToTensor()]
        )
        self.normalize = transforms.Normalize(mean=torch.tensor(t.MEAN), std=torch.tensor(t.STD))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        params = self.augmentation.sample_params() if self.augmentation is not None else None

        image_list = []
        for frame in seq["frames"]:
            image = Image.open(build_image_path(self.video_root, seq["vid"], frame)).convert("RGB")
            image = self.pre(image)
            if params is not None:
                image = self.augmentation.apply(image, params)
            image = minmax_scale(self.normalize(image))
            image_list.append(image)

        return torch.stack(image_list), self.targets[idx]


def get_safe_datasets(config, splits=("train", "val", "test")):
    """
    Build labelled SAFE datasets. Only the train split gets augmentation.

    Returns dict split -> (dataset, meta_df).
    """
    out = {}
    for split in splits:
        df = load_safe_split_df(config, split)
        sequences, meta_rows = build_safe_sequences(df)
        meta_df = pd.DataFrame(meta_rows).reset_index(drop=True)
        if meta_df[SAFE_TARGET_COLUMNS].isna().any().any():
            raise ValueError(f"SAFE {split}: some keyframes have missing C1/C2/C3 labels")
        targets = meta_df[SAFE_TARGET_COLUMNS].astype(float).values.tolist()
        augmentation = build_safe_train_augmentation(config) if split == "train" else None
        out[split] = (SafeSwinCVS_Dataset(sequences, targets, config, augmentation), meta_df)
        print(f"SAFE {split}: {len(sequences)} sequences | positives C1/C2/C3 = "
              f"{[int(x) for x in meta_df[SAFE_TARGET_COLUMNS].sum().tolist()]}")
    return out


def compute_pos_weight(dataset):
    """pos_weight = #neg / #pos per class, from the dataset targets (BCEWithLogitsLoss convention)."""
    targets = dataset.targets
    pos = targets.sum(dim=0)
    neg = targets.shape[0] - pos
    if (pos == 0).any():
        raise ValueError(f"Cannot compute pos_weight: class with no positives (pos={pos.tolist()})")
    return neg / pos


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_safe_dataloader(config, dataset, shuffle, batch_size=None):
    generator = torch.Generator()
    generator.manual_seed(config.SEED)
    num_workers = config.TRAIN.NUM_WORKERS
    return DataLoader(
        dataset,
        batch_size=batch_size if batch_size is not None else config.TRAIN.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        worker_init_fn=_seed_worker,
        generator=generator,
    )

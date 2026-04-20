"""
dataset_bc.py
WorkzoneBCDataset – reads work-zone driving annotations and serves
single-frame samples for behavior cloning.

Expected annotation format (JSON list of dicts):
[
    {
        "front_image_path": "D:/data/frame_000.jpg",
        "speed":    12.5,   # km/h (or any consistent unit)
        "steer":    0.05,   # [-1, 1]
        "throttle": 0.6,    # [0, 1]
        "brake":    0.0     # [0, 1]
        // extra fields are ignored
    },
    ...
]
"""

import os
import json
import warnings

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

import config_bc as cfg


# ---------------------------------------------------------------------------
# Validity limits for label sanity-check warnings
# ---------------------------------------------------------------------------
_STEER_RANGE    = (-1.0, 1.0)
_THROTTLE_RANGE = (0.0,  1.0)
_BRAKE_RANGE    = (0.0,  1.0)


def _check_label_range(value: float, lo: float, hi: float, name: str, idx: int):
    if not (lo <= value <= hi):
        warnings.warn(
            f"Sample {idx}: '{name}' value {value:.4f} is outside [{lo}, {hi}]."
        )


# ---------------------------------------------------------------------------
# Speed normalisation helpers
# ---------------------------------------------------------------------------
def normalise_speed(speed: float) -> float:
    """Normalise a raw speed scalar according to config settings."""
    mode = cfg.SPEED_NORM_MODE
    if mode == "max_speed":
        return speed / (cfg.MAX_SPEED + 1e-8)
    elif mode == "zscore":
        return (speed - cfg.SPEED_MEAN) / (cfg.SPEED_STD + 1e-8)
    else:
        raise ValueError(
            f"Unknown SPEED_NORM_MODE '{mode}'. Choose 'max_speed' or 'zscore'."
        )


# ---------------------------------------------------------------------------
# Image transforms
# ---------------------------------------------------------------------------
def build_transform(augment: bool = False) -> T.Compose:
    """
    Build a torchvision transform pipeline.

    Parameters
    ----------
    augment : bool
        If True and config flags are set, add mild data augmentation.
        Horizontal flip is intentionally excluded (steer label not mirrored).
    """
    ops = []

    if augment and cfg.AUG_COLOR_JITTER:
        ops.append(
            T.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.1,
                hue=0.02,
            )
        )

    ops += [
        T.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        T.ToTensor(),
        # ImageNet mean/std normalisation
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ]

    return T.Compose(ops)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class WorkzoneBCDataset(Dataset):
    """
    Single-frame BC dataset for work-zone driving.

    Each __getitem__ returns a dict with:
        image         : FloatTensor [3, H, W]
        speed         : FloatTensor [1]   – normalised current speed
        target_action : FloatTensor [3]   – [steer, throttle, brake]
        target_speed  : FloatTensor [1]   – normalised speed (auxiliary head target)

    Parameters
    ----------
    annotation_path : str
        Path to the JSON annotation file.
    augment : bool
        Whether to apply training-time augmentation.
    frame_stride : int
        Sub-sample the annotation list by this stride (1 = use all frames).
    """

    def __init__(
        self,
        annotation_path: str,
        augment: bool = False,
        frame_stride: int = 1,
    ):
        super().__init__()

        if not os.path.isfile(annotation_path):
            raise FileNotFoundError(
                f"Annotation file not found: {annotation_path}"
            )

        with open(annotation_path, "r", encoding="utf-8") as f:
            raw_samples = json.load(f)

        if not isinstance(raw_samples, list):
            raise ValueError(
                "Annotation JSON must be a list of sample dicts."
            )

        # Apply stride and sort by image path for reproducible ordering
        raw_samples = sorted(raw_samples, key=lambda s: s.get("front_image_path", ""))
        raw_samples = raw_samples[::max(1, int(frame_stride))]

        # Validate and filter samples
        self.samples = []
        for idx, s in enumerate(raw_samples):
            missing = [
                k for k in ("front_image_path", "speed", "steer", "throttle", "brake")
                if k not in s
            ]
            if missing:
                warnings.warn(
                    f"Sample {idx} is missing required keys {missing}. Skipping."
                )
                continue

            img_path = s["front_image_path"]
            if not os.path.isfile(img_path):
                warnings.warn(
                    f"Sample {idx}: image not found at '{img_path}'. Skipping."
                )
                continue

            # Label range checks (warn, do not drop)
            _check_label_range(s["steer"],    *_STEER_RANGE,    "steer",    idx)
            _check_label_range(s["throttle"], *_THROTTLE_RANGE, "throttle", idx)
            _check_label_range(s["brake"],    *_BRAKE_RANGE,    "brake",    idx)

            self.samples.append(s)

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No valid samples found in '{annotation_path}'. "
                "Check image paths and required keys."
            )

        self.transform = build_transform(augment=augment)
        print(
            f"[WorkzoneBCDataset] Loaded {len(self.samples)} samples "
            f"from '{annotation_path}' (stride={frame_stride}, augment={augment})."
        )

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]

        # --- Image ---
        img_path = s["front_image_path"]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load image at '{img_path}': {exc}"
            ) from exc
        image = self.transform(image)  # [3, H, W]

        # --- Speed ---
        raw_speed = float(s["speed"])
        norm_speed = normalise_speed(raw_speed)
        speed_tensor = torch.tensor([norm_speed], dtype=torch.float32)

        # --- Action labels ---
        steer    = float(s["steer"])
        throttle = float(s["throttle"])
        brake    = float(s["brake"])
        target_action = torch.tensor([steer, throttle, brake], dtype=torch.float32)

        # --- Auxiliary speed label (same normalised value) ---
        target_speed = speed_tensor.clone()

        return {
            "image":         image,          # [3, H, W]
            "speed":         speed_tensor,   # [1]
            "target_action": target_action,  # [3]
            "target_speed":  target_speed,   # [1]
            # keep raw path for evaluation logging
            "image_path":    img_path,
        }


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python dataset_bc.py <annotation_json>")
        sys.exit(1)

    ds = WorkzoneBCDataset(sys.argv[1], augment=True, frame_stride=1)
    sample = ds[0]
    print("Keys      :", list(sample.keys()))
    print("image     :", sample["image"].shape, sample["image"].dtype)
    print("speed     :", sample["speed"])
    print("action    :", sample["target_action"])
    print("target_sp :", sample["target_speed"])
    print("img_path  :", sample["image_path"])

"""
eval_bc.py
Open-loop evaluation for CILRSVanillaBC.

Usage
-----
    python eval_bc.py --checkpoint <path/to/best.pth> [--ann <annotation_json>]

If --checkpoint is not given, cfg.EVAL_CHECKPOINT is used.
If --ann      is not given, cfg.VAL_ANNOTATION_PATH is used.

Outputs (written to cfg.RESULT_DIR)
-------
    eval_metrics.json   – scalar metrics summary
    pred_vs_gt.csv      – per-sample predictions vs. ground-truth
"""

import os
import csv
import json
import argparse

import torch
from torch.utils.data import DataLoader
import numpy as np

import config_bc as cfg
from dataset_bc import WorkzoneBCDataset
from model_bc   import CILRSVanillaBC
from utils_bc   import get_device, load_checkpoint


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _mae(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - gt)))


def _rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - gt) ** 2)))


# ---------------------------------------------------------------------------
# Evaluation function
# ---------------------------------------------------------------------------

def evaluate(checkpoint_path: str, annotation_path: str):
    cfg.make_dirs()
    device = get_device(cfg.DEVICE)

    # ---- Dataset -----------------------------------------------------------
    dataset = WorkzoneBCDataset(
        annotation_path=annotation_path,
        augment=False,
        frame_stride=cfg.FRAME_STRIDE,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
    )

    # ---- Model -------------------------------------------------------------
    model = CILRSVanillaBC(
        pretrained=False,          # weights loaded from checkpoint
        speed_emb_dim=cfg.SPEED_EMBEDDING_DIM,
        shared_hidden1=cfg.SHARED_HIDDEN_DIM,
        shared_hidden2=cfg.SHARED_HIDDEN_DIM2,
        dropout=cfg.DROPOUT,
    ).to(device)

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    epoch = load_checkpoint(checkpoint_path, model, device=device)
    print(f"[eval_bc] Loaded checkpoint '{checkpoint_path}' (epoch {epoch}).")

    model.eval()

    # ---- Inference loop ----------------------------------------------------
    all_pred_steer    = []
    all_pred_throttle = []
    all_pred_brake    = []
    all_pred_speed    = []

    all_gt_steer    = []
    all_gt_throttle = []
    all_gt_brake    = []
    all_gt_speed    = []

    all_image_paths = []

    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            speed = batch["speed"].to(device, non_blocking=True)

            out = model(image, speed)
            pred_action = out["action"].cpu().numpy()  # [B, 3]
            pred_speed  = out["speed"].cpu().numpy()   # [B, 1]

            gt_action = batch["target_action"].numpy()  # [B, 3]
            gt_speed  = batch["target_speed"].numpy()   # [B, 1]

            all_pred_steer.extend(pred_action[:, 0].tolist())
            all_pred_throttle.extend(pred_action[:, 1].tolist())
            all_pred_brake.extend(pred_action[:, 2].tolist())
            all_pred_speed.extend(pred_speed[:, 0].tolist())

            all_gt_steer.extend(gt_action[:, 0].tolist())
            all_gt_throttle.extend(gt_action[:, 1].tolist())
            all_gt_brake.extend(gt_action[:, 2].tolist())
            all_gt_speed.extend(gt_speed[:, 0].tolist())

            all_image_paths.extend(batch["image_path"])

    # ---- Compute metrics ---------------------------------------------------
    ps  = np.array(all_pred_steer)
    pt  = np.array(all_pred_throttle)
    pb  = np.array(all_pred_brake)
    psp = np.array(all_pred_speed)

    gs  = np.array(all_gt_steer)
    gt  = np.array(all_gt_throttle)
    gb  = np.array(all_gt_brake)
    gsp = np.array(all_gt_speed)

    metrics = {
        "MAE_steer":          _mae(ps,  gs),
        "MAE_throttle":       _mae(pt,  gt),
        "MAE_brake":          _mae(pb,  gb),
        "MAE_speed":          _mae(psp, gsp),
        "overall_action_MAE": float((_mae(ps, gs) + _mae(pt, gt) + _mae(pb, gb)) / 3),
        "RMSE_steer":         _rmse(ps,  gs),
        "RMSE_throttle":      _rmse(pt,  gt),
        "RMSE_brake":         _rmse(pb,  gb),
        "RMSE_speed":         _rmse(psp, gsp),
        "num_samples":        len(all_image_paths),
        "checkpoint":         checkpoint_path,
        "annotation":         annotation_path,
    }

    # Print summary
    print("\n" + "=" * 50)
    print("  Open-loop Evaluation Results")
    print("=" * 50)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<25s}: {v:.5f}")
        else:
            print(f"  {k:<25s}: {v}")
    print("=" * 50 + "\n")

    # ---- Save eval_metrics.json -------------------------------------------
    metrics_path = os.path.join(cfg.RESULT_DIR, "eval_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to : {metrics_path}")

    # ---- Save pred_vs_gt.csv ----------------------------------------------
    csv_path = os.path.join(cfg.RESULT_DIR, "pred_vs_gt.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_path",
            "gt_steer",    "pred_steer",
            "gt_throttle", "pred_throttle",
            "gt_brake",    "pred_brake",
            "gt_speed",    "pred_speed",
        ])
        for i in range(len(all_image_paths)):
            writer.writerow([
                all_image_paths[i],
                f"{all_gt_steer[i]:.6f}",    f"{all_pred_steer[i]:.6f}",
                f"{all_gt_throttle[i]:.6f}", f"{all_pred_throttle[i]:.6f}",
                f"{all_gt_brake[i]:.6f}",    f"{all_pred_brake[i]:.6f}",
                f"{all_gt_speed[i]:.6f}",    f"{all_pred_speed[i]:.6f}",
            ])
    print(f"Predictions saved to : {csv_path}")

    return metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Open-loop BC evaluation")
    parser.add_argument(
        "--checkpoint",
        default=cfg.EVAL_CHECKPOINT,
        help="Path to model checkpoint (.pth)",
    )
    parser.add_argument(
        "--ann",
        default=cfg.VAL_ANNOTATION_PATH,
        help="Path to annotation JSON (defaults to cfg.VAL_ANNOTATION_PATH)",
    )
    args = parser.parse_args()

    if not args.checkpoint:
        parser.error(
            "No checkpoint specified. "
            "Pass --checkpoint <path> or set cfg.EVAL_CHECKPOINT."
        )

    evaluate(args.checkpoint, args.ann)

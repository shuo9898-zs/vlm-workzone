"""
utils_bc.py
Shared utilities: loss computation, seed fixing, device selection.
"""

import os
import random

import numpy as np
import torch
import torch.nn as nn

import config_bc as cfg


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def compute_bc_loss(
    pred_dict: dict,
    batch: dict,
    alpha_speed: float = None,
    speed_loss_fn: str = None,
) -> dict:
    """
    Compute the total behavior-cloning loss and return a breakdown dict.

    Parameters
    ----------
    pred_dict : dict
        Model output with keys "action" [B, 3] and "speed" [B, 1].
    batch : dict
        DataLoader batch with keys "target_action" [B, 3] and
        "target_speed" [B, 1].  Both must be on the same device as pred_dict.
    alpha_speed : float, optional
        Weight for the auxiliary speed loss.  Defaults to cfg.ALPHA_SPEED.
    speed_loss_fn : str, optional
        "L1" or "MSE".  Defaults to cfg.SPEED_LOSS_FN.

    Returns
    -------
    dict with keys:
        total_loss, action_loss, steer_loss, throttle_loss, brake_loss,
        speed_loss
    """
    alpha_speed   = alpha_speed   if alpha_speed   is not None else cfg.ALPHA_SPEED
    speed_loss_fn = speed_loss_fn if speed_loss_fn is not None else cfg.SPEED_LOSS_FN

    pred_action = pred_dict["action"]    # [B, 3]
    pred_speed  = pred_dict["speed"]     # [B, 1]
    gt_action   = batch["target_action"] # [B, 3]
    gt_speed    = batch["target_speed"]  # [B, 1]

    l1 = nn.functional.l1_loss

    # --- Per-channel control losses (L1) -----------------------------------
    steer_loss    = l1(pred_action[:, 0], gt_action[:, 0])
    throttle_loss = l1(pred_action[:, 1], gt_action[:, 1])
    brake_loss    = l1(pred_action[:, 2], gt_action[:, 2])
    action_loss   = steer_loss + throttle_loss + brake_loss

    # --- Auxiliary speed loss -----------------------------------------------
    if speed_loss_fn.upper() == "MSE":
        speed_loss = nn.functional.mse_loss(pred_speed, gt_speed)
    else:  # default L1
        speed_loss = l1(pred_speed, gt_speed)

    # --- Total loss ---------------------------------------------------------
    total_loss = action_loss + alpha_speed * speed_loss

    return {
        "total_loss":    total_loss,
        "action_loss":   action_loss,
        "steer_loss":    steer_loss,
        "throttle_loss": throttle_loss,
        "brake_loss":    brake_loss,
        "speed_loss":    speed_loss,
    }


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device(requested: str = "cuda") -> torch.device:
    """
    Return the requested device, falling back to CPU if CUDA is unavailable.
    """
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested != "cpu":
        print(
            f"[utils_bc] Requested device '{requested}' not available. "
            "Falling back to CPU."
        )
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# AverageMeter – tracks running mean of a scalar
# ---------------------------------------------------------------------------

class AverageMeter:
    """Computes and stores a running average of a scalar quantity."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum   = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.sum   += val * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count > 0 else 0.0


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(state: dict, path: str):
    """Save a training state dict to *path*."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, model: nn.Module, optimizer=None, device=None):
    """
    Load a checkpoint.

    Returns
    -------
    int  – epoch number stored in the checkpoint (0 if not present)
    """
    if device is None:
        device = torch.device("cpu")
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt.get("epoch", 0)

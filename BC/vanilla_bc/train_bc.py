"""
train_bc.py
Full training pipeline for CILRSVanillaBC.

Usage
-----
    # fresh start
    python train_bc.py

    # resume from a checkpoint (e.g. after adding new data)
    python train_bc.py --resume D:\VLM\BC\vanilla_bc\checkpoints\latest.pth

All hyper-parameters are read from config_bc.py.
Checkpoints, logs, and metrics are written to the directories defined there.
"""

import argparse
import os
import csv
import sys
import time

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

import config_bc as cfg
from dataset_bc import WorkzoneBCDataset
from model_bc   import CILRSVanillaBC
from utils_bc   import (
    compute_bc_loss,
    set_seed,
    get_device,
    AverageMeter,
    save_checkpoint,
    load_checkpoint,
)


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _open_log(log_dir: str, mode: str = "w"):
    """Return (txt_file_handle, csv_writer, csv_file_handle)."""
    os.makedirs(log_dir, exist_ok=True)
    txt_path = os.path.join(log_dir, "train_log.txt")
    csv_path = os.path.join(log_dir, "metrics.csv")

    txt_f = open(txt_path, mode, encoding="utf-8")

    csv_f  = open(csv_path, mode, newline="", encoding="utf-8")
    writer = csv.writer(csv_f)
    # only write header on fresh start
    if mode == "w":
        writer.writerow([
            "epoch",
            "train_total", "train_action", "train_steer",
            "train_throttle", "train_brake", "train_speed",
            "val_total",   "val_action",   "val_steer",
            "val_throttle",   "val_brake",   "val_speed",
            "lr",
        ])

    return txt_f, writer, csv_f


def _log(txt_f, msg: str):
    print(msg)
    txt_f.write(msg + "\n")
    txt_f.flush()


# ---------------------------------------------------------------------------
# One-epoch helpers
# ---------------------------------------------------------------------------

def run_epoch(model, loader, optimizer, device, train: bool):
    """
    Run one full pass over *loader*.

    Returns
    -------
    dict of AverageMeter averages for all tracked losses.
    """
    model.train(train)

    meters = {
        k: AverageMeter()
        for k in ("total_loss", "action_loss", "steer_loss",
                  "throttle_loss", "brake_loss", "speed_loss")
    }

    for batch in loader:
        image   = batch["image"].to(device, non_blocking=True)
        speed   = batch["speed"].to(device, non_blocking=True)
        for k in ("target_action", "target_speed"):
            batch[k] = batch[k].to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            pred_dict = model(image, speed)
            loss_dict = compute_bc_loss(pred_dict, batch)

        if train:
            loss_dict["total_loss"].backward()
            # Gradient clipping for training stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

        n = image.size(0)
        for k, meter in meters.items():
            meter.update(loss_dict[k].item(), n)

    return {k: m.avg for k, m in meters.items()}


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(resume: str = None):
    cfg.make_dirs()
    set_seed(cfg.SEED)
    device = get_device(cfg.DEVICE)

    # ---- Datasets ----------------------------------------------------------
    train_ds = WorkzoneBCDataset(
        annotation_path=cfg.TRAIN_ANNOTATION_PATH,
        augment=True,
        frame_stride=cfg.FRAME_STRIDE,
    )
    val_ds = WorkzoneBCDataset(
        annotation_path=cfg.VAL_ANNOTATION_PATH,
        augment=False,
        frame_stride=cfg.FRAME_STRIDE,
    )

    # Use persistent_workers only when num_workers > 0
    _pw = cfg.NUM_WORKERS > 0
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=_pw,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=_pw,
    )

    # ---- Model -------------------------------------------------------------
    model = CILRSVanillaBC(
        pretrained=cfg.PRETRAINED,
        speed_emb_dim=cfg.SPEED_EMBEDDING_DIM,
        shared_hidden1=cfg.SHARED_HIDDEN_DIM,
        shared_hidden2=cfg.SHARED_HIDDEN_DIM2,
        dropout=cfg.DROPOUT,
    ).to(device)

    # ---- Optimizer ---------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.LR,
        weight_decay=cfg.WEIGHT_DECAY,
    )

    # ---- Scheduler ---------------------------------------------------------
    if cfg.SCHEDULER_TYPE == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS, eta_min=1e-6)
        sched_name = f"CosineAnnealingLR(T_max={cfg.EPOCHS}, eta_min=1e-6)"
    else:
        scheduler = StepLR(
            optimizer,
            step_size=cfg.STEP_LR_STEP_SIZE,
            gamma=cfg.STEP_LR_GAMMA,
        )
        sched_name = (
            f"StepLR(step_size={cfg.STEP_LR_STEP_SIZE}, "
            f"gamma={cfg.STEP_LR_GAMMA})"
        )

    # ---- Resume -----------------------------------------------------------
    start_epoch = 0
    if resume:
        start_epoch = load_checkpoint(resume, model, optimizer, device)
        # fast-forward scheduler to match the resumed epoch
        for _ in range(start_epoch):
            scheduler.step()
        print(f"[resume] Loaded checkpoint '{resume}' (epoch {start_epoch})")

    # ---- Logging -----------------------------------------------------------
    log_mode = "a" if resume else "w"
    txt_f, csv_writer, csv_f = _open_log(cfg.LOG_DIR, mode=log_mode)

    _log(txt_f, "=" * 65)
    _log(txt_f, "  CILRS Vanilla BC – Training" + (f" (resumed from epoch {start_epoch})" if resume else ""))
    _log(txt_f, "=" * 65)
    _log(txt_f, f"  Device       : {device}")
    _log(txt_f, f"  Train samples: {len(train_ds)}")
    _log(txt_f, f"  Val   samples: {len(val_ds)}")
    _log(txt_f, f"  Epochs       : {cfg.EPOCHS}")
    _log(txt_f, f"  Batch size   : {cfg.BATCH_SIZE}")
    _log(txt_f, f"  LR           : {cfg.LR}")
    _log(txt_f, f"  Scheduler    : {sched_name}")
    _log(txt_f, f"  alpha_speed  : {cfg.ALPHA_SPEED}")
    _log(txt_f, f"  Speed loss   : {cfg.SPEED_LOSS_FN}")
    _log(txt_f, "=" * 65)

    best_val_loss = float("inf")
    latest_path   = os.path.join(cfg.CHECKPOINT_DIR, "latest.pth")
    best_path     = os.path.join(cfg.CHECKPOINT_DIR, "best.pth")

    for epoch in range(start_epoch + 1, start_epoch + cfg.EPOCHS + 1):
        t0 = time.time()

        train_losses = run_epoch(model, train_loader, optimizer, device, train=True)
        val_losses   = run_epoch(model, val_loader,   None,      device, train=False)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        elapsed = time.time() - t0

        # --- Console/file log -----------------------------------------------
        msg = (
            f"[{epoch:03d}/{start_epoch + cfg.EPOCHS}] "
            f"train_total={train_losses['total_loss']:.4f}  "
            f"val_total={val_losses['total_loss']:.4f}  "
            f"(steer={val_losses['steer_loss']:.4f}  "
            f"throt={val_losses['throttle_loss']:.4f}  "
            f"brake={val_losses['brake_loss']:.4f}  "
            f"spd={val_losses['speed_loss']:.4f})  "
            f"lr={current_lr:.2e}  "
            f"time={elapsed:.1f}s"
        )
        _log(txt_f, msg)

        # --- CSV log --------------------------------------------------------
        csv_writer.writerow([
            epoch,
            f"{train_losses['total_loss']:.6f}",
            f"{train_losses['action_loss']:.6f}",
            f"{train_losses['steer_loss']:.6f}",
            f"{train_losses['throttle_loss']:.6f}",
            f"{train_losses['brake_loss']:.6f}",
            f"{train_losses['speed_loss']:.6f}",
            f"{val_losses['total_loss']:.6f}",
            f"{val_losses['action_loss']:.6f}",
            f"{val_losses['steer_loss']:.6f}",
            f"{val_losses['throttle_loss']:.6f}",
            f"{val_losses['brake_loss']:.6f}",
            f"{val_losses['speed_loss']:.6f}",
            f"{current_lr:.8f}",
        ])
        csv_f.flush()

        # --- Checkpointing --------------------------------------------------
        state = {
            "epoch":           epoch,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_total_loss":  val_losses["total_loss"],
        }
        save_checkpoint(state, latest_path)

        if val_losses["total_loss"] < best_val_loss:
            best_val_loss = val_losses["total_loss"]
            save_checkpoint(state, best_path)
            _log(txt_f, f"  → New best model saved  (val_loss={best_val_loss:.4f})")

    _log(txt_f, "=" * 65)
    _log(txt_f, f"Training complete. Best val loss: {best_val_loss:.4f}")
    _log(txt_f, f"Checkpoints in : {cfg.CHECKPOINT_DIR}")
    _log(txt_f, f"Logs in        : {cfg.LOG_DIR}")

    txt_f.close()
    csv_f.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint (.pth) to resume training from",
    )
    args = parser.parse_args()
    train(resume=args.resume)

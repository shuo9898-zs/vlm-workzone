"""
model_bc.py
CILRSVanillaBC — single-frame ResNet-34 behavior cloning model.

Architecture
------------
  ImageEncoder   : ResNet-34 (ImageNet pretrained, fc layer removed) → 512-d
  SpeedMLP       : Linear(1→32)→ReLU→Linear(32→64)→ReLU             → 64-d
  SharedTrunk    : Linear(576→512)→ReLU→Dropout→Linear(512→256)→ReLU
  ControlHead    : Linear(256→3) with per-channel activations
                     steer    → tanh
                     throttle → sigmoid
                     brake    → sigmoid
  SpeedHead      : Linear(256→1)  (raw regression, no activation)
"""

import torch
import torch.nn as nn
from torchvision import models

import config_bc as cfg


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class SpeedMLP(nn.Module):
    """Small MLP that embeds a scalar speed into a fixed-dim vector."""

    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, speed: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        speed : Tensor [B, 1]  – normalised speed scalar

        Returns
        -------
        Tensor [B, out_dim]
        """
        return self.net(speed)


class SharedTrunk(nn.Module):
    """Shared FC trunk that processes the fused image+speed feature."""

    def __init__(self, in_dim: int, hidden1: int = 512, hidden2: int = 256,
                 dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class CILRSVanillaBC(nn.Module):
    """
    CILRS-style Vanilla Behavior Cloning model.

    Inputs
    ------
    image : Tensor [B, 3, H, W]
    speed : Tensor [B, 1]   – normalised current speed

    Outputs
    -------
    dict with:
        "action" : Tensor [B, 3]  – [steer, throttle, brake]
        "speed"  : Tensor [B, 1]  – predicted (auxiliary) speed
    """

    def __init__(
        self,
        pretrained: bool       = True,
        speed_emb_dim: int     = 64,
        shared_hidden1: int    = 512,
        shared_hidden2: int    = 256,
        dropout: float         = 0.3,
    ):
        super().__init__()

        # ---- Image encoder (ResNet-34, global average pool output) --------
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet34(weights=weights)
        visual_dim = backbone.fc.in_features          # 512
        backbone.fc = nn.Identity()                   # remove classification head
        self.image_encoder = backbone

        # ---- Speed embedding MLP ------------------------------------------
        self.speed_mlp = SpeedMLP(out_dim=speed_emb_dim)

        # ---- Shared trunk ----------------------------------------------------
        fused_dim = visual_dim + speed_emb_dim       # 512 + 64 = 576
        self.trunk = SharedTrunk(
            in_dim=fused_dim,
            hidden1=shared_hidden1,
            hidden2=shared_hidden2,
            dropout=dropout,
        )

        # ---- Output heads ----------------------------------------------------
        self.control_head = nn.Linear(shared_hidden2, 3)
        self.speed_head   = nn.Linear(shared_hidden2, 1)

    # ------------------------------------------------------------------
    def forward(self, image: torch.Tensor, speed: torch.Tensor) -> dict:
        """
        Parameters
        ----------
        image : Tensor [B, 3, H, W]
        speed : Tensor [B, 1]

        Returns
        -------
        dict:
            "action" : Tensor [B, 3]
            "speed"  : Tensor [B, 1]
        """
        # Visual feature
        visual_feat = self.image_encoder(image)       # [B, 512]

        # Speed embedding
        speed_feat = self.speed_mlp(speed)            # [B, 64]

        # Fuse and pass through trunk
        fused   = torch.cat([visual_feat, speed_feat], dim=1)  # [B, 576]
        trunk   = self.trunk(fused)                   # [B, 256]

        # Control prediction with channel-wise activations
        raw_action = self.control_head(trunk)         # [B, 3]
        steer    = torch.tanh(raw_action[:, 0:1])     # [-1, 1]
        throttle = torch.sigmoid(raw_action[:, 1:2])  # [ 0, 1]
        brake    = torch.sigmoid(raw_action[:, 2:3])  # [ 0, 1]
        action_pred = torch.cat([steer, throttle, brake], dim=1)  # [B, 3]

        # Auxiliary speed prediction (raw regression)
        speed_pred = self.speed_head(trunk)           # [B, 1]

        return {
            "action": action_pred,   # [B, 3]
            "speed":  speed_pred,    # [B, 1]
        }


# ---------------------------------------------------------------------------
# Quick forward test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = CILRSVanillaBC(
        pretrained=cfg.PRETRAINED,
        speed_emb_dim=cfg.SPEED_EMBEDDING_DIM,
        shared_hidden1=cfg.SHARED_HIDDEN_DIM,
        shared_hidden2=cfg.SHARED_HIDDEN_DIM2,
        dropout=cfg.DROPOUT,
    )
    model.eval()

    B = 4
    dummy_image = torch.randn(B, 3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)
    dummy_speed = torch.randn(B, 1)

    with torch.no_grad():
        out = model(dummy_image, dummy_speed)

    print("action shape :", out["action"].shape)   # [4, 3]
    print("speed  shape :", out["speed"].shape)    # [4, 1]
    print("steer  range : [{:.3f}, {:.3f}]".format(
        out["action"][:, 0].min().item(), out["action"][:, 0].max().item()))
    print("throttle range : [{:.3f}, {:.3f}]".format(
        out["action"][:, 1].min().item(), out["action"][:, 1].max().item()))
    print("brake  range : [{:.3f}, {:.3f}]".format(
        out["action"][:, 2].min().item(), out["action"][:, 2].max().item()))
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

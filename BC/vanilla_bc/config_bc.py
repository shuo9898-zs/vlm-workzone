"""
config_bc.py
All hyperparameters and paths for the CILRS-style Vanilla BC baseline.
Edit this file to change training behavior; do not scatter magic numbers in other files.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Annotation JSONs must each contain a list of sample dicts.
# Each dict must have at minimum:
#   front_image_path, speed, steer, throttle, brake
TRAIN_ANNOTATION_PATH = r"D:\VLM\BC\vanilla_bc\annotations\train.json"
VAL_ANNOTATION_PATH   = r"D:\VLM\BC\vanilla_bc\annotations\val.json"

CHECKPOINT_DIR = r"D:\VLM\BC\vanilla_bc\checkpoints"
LOG_DIR        = r"D:\VLM\BC\vanilla_bc\logs"
RESULT_DIR     = r"D:\VLM\BC\vanilla_bc\results"

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
IMAGE_SIZE  = 224          # resize to (IMAGE_SIZE x IMAGE_SIZE)
BATCH_SIZE  = 64
NUM_WORKERS = 8            # set to 0 on Windows if multiprocessing issues arise

# Speed normalisation
# Options: "max_speed" (x / max_speed)  |  "zscore" (standard score)
SPEED_NORM_MODE = "max_speed"
MAX_SPEED       = 50.0     # km/h  – used when SPEED_NORM_MODE == "max_speed"
SPEED_MEAN      = 0.0      # used when SPEED_NORM_MODE == "zscore"
SPEED_STD       = 1.0      # used when SPEED_NORM_MODE == "zscore"

# Frame stride for 10 Hz data.
# 1 = use every frame (full 10 Hz)
# 2 = use every other frame (5 Hz)
# 5 = use every 5th frame (2 Hz)
FRAME_STRIDE = 1

# Data augmentation flags
AUG_COLOR_JITTER    = True   # mild brightness / contrast / saturation jitter
AUG_HORIZONTAL_FLIP = False  # disabled – safe only when steer label is mirrored

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
PRETRAINED          = True   # use ImageNet pretrained ResNet-34 weights
SPEED_EMBEDDING_DIM = 64     # output dim of the speed MLP branch
SHARED_HIDDEN_DIM   = 512    # first FC after concat
SHARED_HIDDEN_DIM2  = 256    # second FC
DROPOUT             = 0.3

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
EPOCHS         = 50
LR             = 1e-4
WEIGHT_DECAY   = 1e-4
ALPHA_SPEED    = 0.1         # weight for the auxiliary speed prediction loss
SPEED_LOSS_FN  = "L1"        # "L1" or "MSE"

# Scheduler: "cosine" or "step"
SCHEDULER_TYPE     = "cosine"
STEP_LR_STEP_SIZE  = 10      # only used when SCHEDULER_TYPE == "step"
STEP_LR_GAMMA      = 0.5     # only used when SCHEDULER_TYPE == "step"

SEED   = 42
DEVICE = "cuda"              # "cuda" or "cpu"; auto-falls-back to cpu if no GPU

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
EVAL_CHECKPOINT = ""         # set to path of a .pth file to evaluate

# ---------------------------------------------------------------------------
# Derived paths (created automatically at runtime)
# ---------------------------------------------------------------------------
def make_dirs():
    """Create output directories if they do not exist."""
    for d in (CHECKPOINT_DIR, LOG_DIR, RESULT_DIR):
        os.makedirs(d, exist_ok=True)

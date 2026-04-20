"""
prepare_annotations.py
======================
Build train/val annotation JSONs for the WorkZone BC baseline.

Data structure expected
-----------------------
DATA_ROOT/
  P{i}/
    S{j}/
      carla_data_<session>/
        camera_front/
          camera_1765314677155.jpg   ← filename encodes ms-timestamp
          camera_1765314677270.jpg
          ...
        vehicle_data.csv             ← 10 Hz, columns include:
                                        timestamp_  (MM:SS.d relative)
                                        Speed, Steering, Throttle, Brake

WZ_ANNOTATION_CSV: one row per (Participant × Scenario) with columns:
  [row label = "P2_S1" etc.]
  wz1_start, wz1_end
  wz2_start, wz2_end
  wz3_start_warning, wz3_end        ← note: warning column, not plain wz3_start

All wz timestamps are strings like "merged_1765314710.576000"
(strip "merged_" → float seconds, same epoch as image filenames ÷ 1000).

Output
------
annotations/train.json
annotations/val.json

Each JSON is a list of sample dicts:
{
    "front_image_path": "D:/...",
    "speed":    12.5,
    "steer":    0.05,
    "throttle": 0.6,
    "brake":    0.0,
    "participant": "P5",
    "scenario":    "S1",
    "workzone":    "wz2",
    "session":     "carla_data_2025-12-09_16-11-16"
}

Usage
-----
    python prepare_annotations.py

Edit the CONFIG section below before running.
"""

import csv
import glob
import json
import math
import os
import re
import warnings
from typing import List, Optional, Tuple


# =============================================================================
# CONFIG — edit these before running
# =============================================================================

DATA_ROOT = r"D:\Shuo_WorkZone_Data"

# Path to the WorkZone timing file — supports both .xlsx and .csv directly.
# Columns needed (exact names, case-sensitive):
#   wz1_start, wz1_end,
#   wz2_start, wz2_end,
#   wz3_start_warning, wz3_end
# Row labels (first column) must be like "P2_S1", "P5_S1", etc.
WZ_ANNOTATION_PATH = r"D:\Shuo_WorkZone_Data\workzone driving data.xlsx"

# Seconds padding added before wz_start and after wz_end
WZ_PAD_SEC = 2.0

# Participants and scenarios to include (None = auto-discover all).
# ── SMOKE TEST: set PARTICIPANTS = ["P5"] to run on P5 only ──
PARTICIPANTS = ["P5"]   # change to None to use all participants
SCENARIOS    = None     # e.g. ["S1", "S2"] or None for all

# WorkZone segments to include
WZ_SEGMENTS = ["wz1", "wz2", "wz3"]   # subset of ["wz1", "wz2", "wz3"]

# Train / Val split — by segment (task)
# All P×S×wz clips are pooled, shuffled, then split at the SEGMENT level.
# VAL_RATIO = 0.2  →  ~20% of segments go to val (no frame leaks across splits).
VAL_RATIO = 0.2
SPLIT_SEED = 42   # for reproducible shuffle

# Matching tolerance: max seconds between CSV row and nearest image
# If a CSV row has no image within this window it is skipped
MATCH_TOLERANCE_SEC = 0.15   # 1.5× a 10 Hz tick

# Speed unit in vehicle_data.csv
# If CARLA reports in m/s, set to "ms" → converted to km/h internally
# If already in km/h, set to "kmh"
SPEED_UNIT = "ms"   # "ms" or "kmh"

# Output directory for annotation JSONs
OUTPUT_DIR = r"D:\VLM\BC\vanilla_bc\annotations"

# =============================================================================
# END CONFIG
# =============================================================================


# ---------------------------------------------------------------------------
# Timestamp parsing helpers
# ---------------------------------------------------------------------------

def parse_image_ts(filename: str) -> Optional[float]:
    """
    Extract absolute timestamp (seconds) from image filename.

    camera_1765314677155.jpg  →  1765314677.155
    """
    m = re.search(r"camera_(\d+)\.jpg$", os.path.basename(filename),
                  re.IGNORECASE)
    if not m:
        return None
    return int(m.group(1)) / 1000.0


def parse_csv_ts(ts_str: str) -> Optional[float]:
    """
    Parse a CSV timestamp string to a Python datetime.

    Supports:
      "2025-12-09 16:11:20.101"  → datetime object
      "MM:SS.d" legacy format    → still handled

    Returns the datetime as a float (unix seconds) or None on failure.
    We only use the relative difference between rows, so timezone does not matter.
    """
    from datetime import datetime
    ts_str = ts_str.strip()
    # Try full datetime format first
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(ts_str, fmt).timestamp()
        except ValueError:
            pass
    # Legacy MM:SS.d fallback
    m = re.match(r"^(\d+):(\d+(?:\.\d+)?)$", ts_str)
    if m:
        return int(m.group(1)) * 60.0 + float(m.group(2))
    return None


def parse_wz_ts(wz_str: str) -> Optional[float]:
    """
    Parse a workzone timestamp string → absolute seconds.

    "merged_1765314710.576000"  →  1765314710.576
    Also handles plain floats (already converted).
    """
    if not wz_str or str(wz_str).strip() in ("", "nan", "None"):
        return None
    s = str(wz_str).strip()
    s = re.sub(r"^merged_", "", s)
    try:
        return float(s)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Per-session data loading
# ---------------------------------------------------------------------------

def load_session(session_dir: str) -> Tuple[List[dict], Optional[float]]:
    """
    Load the vehicle_data.csv and image list from one session directory.

    Returns
    -------
    (rows, session_base_abs)
        rows             : list of dicts (one per CSV row, with abs_ts added)
        session_base_abs : absolute Unix timestamp of the session start (seconds)
                           derived from the earliest image filename
    """
    csv_path  = os.path.join(session_dir, "vehicle_data.csv")
    cam_dir   = os.path.join(session_dir, "camera_front")

    if not os.path.isfile(csv_path):
        warnings.warn(f"No vehicle_data.csv in '{session_dir}'. Skipping.")
        return [], None
    if not os.path.isdir(cam_dir):
        warnings.warn(f"No camera_front/ in '{session_dir}'. Skipping.")
        return [], None

    # ---- Collect and sort images by absolute timestamp --------------------
    img_files = sorted(
        glob.glob(os.path.join(cam_dir, "camera_*.jpg")),
        key=lambda p: parse_image_ts(p) or 0.0,
    )
    if not img_files:
        warnings.warn(f"No camera images found in '{cam_dir}'. Skipping.")
        return [], None

    img_ts = [(parse_image_ts(f), f) for f in img_files]
    img_ts = [(ts, f) for ts, f in img_ts if ts is not None]
    if not img_ts:
        return [], None

    # ---- Read CSV ---------------------------------------------------------
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))

    if not rows:
        warnings.warn(f"Empty vehicle_data.csv in '{session_dir}'.")
        return [], None

    # ---- Parse CSV timestamps -------------------------------------------
    # Use timestamp_str (full datetime) or timestamp_ (legacy MM:SS.d)
    ts_col = "timestamp_str" if "timestamp_str" in rows[0] else "timestamp_"
    csv_abs = []
    for row in rows:
        ts = parse_csv_ts(row.get(ts_col, ""))
        csv_abs.append(ts)

    valid_csv = [(i, t) for i, t in enumerate(csv_abs) if t is not None]
    if not valid_csv:
        warnings.warn(f"Could not parse any CSV timestamps in '{session_dir}'.")
        return [], None

    # ---- Align via relative offset (timezone-agnostic) -------------------
    # Both CSV and images are 10 Hz from the same session start.
    # Anchor: first image abs_ts = first CSV abs_ts (same moment, different format).
    # For each CSV row: abs_ts = first_image_abs + (csv_ts_i - csv_ts_0)
    first_img_abs = img_ts[0][0]
    first_csv_abs = valid_csv[0][1]

    # ---- Nearest-neighbour match CSV row → image -------------------------
    img_sorted_ts  = [t for t, _ in img_ts]
    img_sorted_fps = [f for _, f in img_ts]

    matched_rows = []
    for i, row in enumerate(rows):
        csv_ts_i = csv_abs[i]
        if csv_ts_i is None:
            continue

        # Timezone-agnostic absolute timestamp:
        # shift CSV datetime to the image time axis via the first-frame offset
        abs_ts = first_img_abs + (csv_ts_i - first_csv_abs)

        # Binary search for nearest image timestamp
        nearest_img, dist = _find_nearest(img_sorted_ts, img_sorted_fps, abs_ts)
        if dist > MATCH_TOLERANCE_SEC:
            continue   # no close-enough image

        speed_raw = _safe_float(row.get("Speed", ""))
        steer     = _safe_float(row.get("Steering", ""))
        throttle  = _safe_float(row.get("Throttle", ""))
        brake     = _safe_float(row.get("Brake", ""))

        if any(v is None for v in (speed_raw, steer, throttle, brake)):
            continue

        # Convert speed to km/h if needed
        speed_kmh = speed_raw * 3.6 if SPEED_UNIT == "ms" else speed_raw

        matched_rows.append({
            "front_image_path": nearest_img,
            "speed":            round(speed_kmh,  4),
            "steer":            round(steer,       6),
            "throttle":         round(throttle,    6),
            "brake":            round(brake,       6),
            "abs_ts":           abs_ts,    # keep for WZ filtering, removed later
        })

    return matched_rows, first_img_abs


def _find_nearest(sorted_ts: list, sorted_fps: list, target: float):
    """Binary search: return (filepath, abs_distance) of nearest timestamp."""
    lo, hi = 0, len(sorted_ts) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_ts[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    # Check lo and lo-1
    best_idx  = lo
    best_dist = abs(sorted_ts[lo] - target)
    if lo > 0:
        d = abs(sorted_ts[lo - 1] - target)
        if d < best_dist:
            best_idx, best_dist = lo - 1, d
    return sorted_fps[best_idx], best_dist


def _safe_float(s) -> Optional[float]:
    try:
        return float(str(s).strip())
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# WorkZone annotation loading
# ---------------------------------------------------------------------------

def _load_annotation_rows(path: str) -> list:
    """
    Load the workzone timing file as a list of row-dicts.
    Supports both .xlsx (via openpyxl) and .csv.
    Returns list of dicts with string keys and string values.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in (".xlsx", ".xls"):
        try:
            import openpyxl
        except ImportError:
            raise ImportError(
                "openpyxl is required to read .xlsx files.\n"
                "Install it with:  pip install openpyxl"
            )
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        ws = wb.active
        rows_iter = ws.iter_rows(values_only=True)
        headers = [str(h).strip() if h is not None else "" for h in next(rows_iter)]
        result = []
        for row in rows_iter:
            # zip stops at the shorter of headers/row, handles ragged Excel sheets
            d = {headers[i]: (str(v).strip() if v is not None else "")
                 for i, v in zip(range(len(headers)), row)}
            result.append(d)
        wb.close()
        return result
    else:
        # CSV fallback
        with open(path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            return [{k: (str(v).strip() if v is not None else "")
                     for k, v in row.items()}
                    for row in reader]


def load_wz_annotations(wz_path: str) -> dict:
    """
    Load workzone timing file (.xlsx or .csv).

    Returns
    -------
    dict mapping "P5_S1" → {
        "wz1": (start_abs, end_abs) or None,
        "wz2": (start_abs, end_abs) or None,
        "wz3": (start_abs, end_abs) or None,
    }
    """
    if not os.path.isfile(wz_path):
        raise FileNotFoundError(
            f"WorkZone annotation file not found: '{wz_path}'\n"
            "Check WZ_ANNOTATION_PATH in the CONFIG section."
        )

    rows = _load_annotation_rows(wz_path)

    result = {}
    for row in rows:
        # First non-empty column is the row label (e.g. "P5_S1")
        label = ""
        for v in row.values():
            v = str(v).strip()
            if v:
                label = v
                break

        if not re.match(r"^P\d+_S\d+$", label):
            continue

        def _window(start_key, end_key, _row=row):
            s = parse_wz_ts(_row.get(start_key, ""))
            e = parse_wz_ts(_row.get(end_key,   ""))
            if s is None or e is None:
                return None
            return (s - WZ_PAD_SEC, e + WZ_PAD_SEC)

        result[label] = {
            "wz1": _window("wz1_start",         "wz1_end"),
            "wz2": _window("wz2_start",         "wz2_end"),
            "wz3": _window("wz3_start_warning", "wz3_end"),
        }

    print(f"[prepare] Loaded WZ annotations for {len(result)} P×S combinations.")
    return result


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_annotations():
    import random
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- Load WZ timing ---------------------------------------------------
    wz_annotations = load_wz_annotations(WZ_ANNOTATION_PATH)

    # ---- Discover participants and scenarios ------------------------------
    all_participants = PARTICIPANTS or sorted(
        d for d in os.listdir(DATA_ROOT)
        if re.match(r"^P\d+$", d) and os.path.isdir(os.path.join(DATA_ROOT, d))
    )

    print(f"[prepare] Participants: {all_participants}")

    # segments: list of (segment_id_str, [sample_dicts])
    # Collect ALL segments first, split afterwards.
    all_segments = []

    for ppt in all_participants:
        ppt_dir = os.path.join(DATA_ROOT, ppt)

        # Discover scenario folders: match S1_normal, S2_noWarning, S3_rainy etc.
        # Extract the short key (e.g. "S1") from the folder name for annotation lookup.
        if SCENARIOS is None:
            raw_scn_dirs = sorted(
                d for d in os.listdir(ppt_dir)
                if re.match(r"^S\d+", d) and os.path.isdir(os.path.join(ppt_dir, d))
            )
            # Build list of (folder_name, short_key) e.g. ("S1_normal", "S1")
            scn_pairs = []
            for d in raw_scn_dirs:
                m = re.match(r"^(S\d+)", d)
                if m:
                    scn_pairs.append((d, m.group(1)))
        else:
            # SCENARIOS given as short keys like ["S1","S2"]; find matching folders
            scn_pairs = []
            for short_key in SCENARIOS:
                for d in os.listdir(ppt_dir):
                    if d.startswith(short_key) and os.path.isdir(os.path.join(ppt_dir, d)):
                        scn_pairs.append((d, short_key))
                        break

        for scn_folder, scn_key in scn_pairs:
            scn_dir = os.path.join(ppt_dir, scn_folder)

            key = f"{ppt}_{scn_key}"
            wz_info = wz_annotations.get(key)
            if wz_info is None:
                warnings.warn(
                    f"No WZ annotation found for '{key}'. Skipping."
                )
                continue

            sessions = sorted(
                os.path.join(scn_dir, d) for d in os.listdir(scn_dir)
                if d.startswith("carla_data_") and
                os.path.isdir(os.path.join(scn_dir, d))
            )
            if not sessions:
                warnings.warn(f"No carla_data_* session found in '{scn_dir}'.")
                continue

            for session_dir in sessions:
                session_name = os.path.basename(session_dir)
                matched_rows, _ = load_session(session_dir)
                if not matched_rows:
                    continue

                for wz_label in WZ_SEGMENTS:
                    window = wz_info.get(wz_label)
                    if window is None:
                        continue
                    wz_start, wz_end = window

                    segment_samples = []
                    for row in matched_rows:
                        if wz_start <= row["abs_ts"] <= wz_end:
                            sample = {k: v for k, v in row.items() if k != "abs_ts"}
                            sample["participant"] = ppt
                            sample["scenario"]    = scn_key
                            sample["workzone"]    = wz_label
                            sample["session"]     = session_name
                            segment_samples.append(sample)

                    if segment_samples:
                        seg_id = f"{key}_{wz_label}_{session_name}"
                        all_segments.append((seg_id, segment_samples))
                        print(
                            f"  [collected] {seg_id}: "
                            f"{len(segment_samples)} samples"
                        )

    # ---- Segment-level shuffle then split ---------------------------------
    # Split at segment level so NO continuous clip is split across train/val.
    random.seed(SPLIT_SEED)
    random.shuffle(all_segments)

    n_val   = max(1, round(len(all_segments) * VAL_RATIO))
    n_train = len(all_segments) - n_val

    train_segs = all_segments[:n_train]
    val_segs   = all_segments[n_train:]

    train_samples = [s for _, segs in train_segs for s in segs]
    val_samples   = [s for _, segs in val_segs   for s in segs]

    # ---- Save JSONs -------------------------------------------------------
    train_path = os.path.join(OUTPUT_DIR, "train.json")
    val_path   = os.path.join(OUTPUT_DIR, "val.json")

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_samples, f, indent=2)
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_samples, f, indent=2)

    print("\n" + "=" * 60)
    print(f"  Total segments: {len(all_segments)}  "
          f"(train {n_train} / val {n_val})")
    print(f"  Train samples : {len(train_samples):>6}")
    print(f"  Val   samples : {len(val_samples):>6}")
    print(f"  Train JSON    : {train_path}")
    print(f"  Val   JSON    : {val_path}")
    print("=" * 60)

    print("\nVal segments:")
    for seg_id, segs in val_segs:
        print(f"  {seg_id}  ({len(segs)} frames)")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    build_annotations()

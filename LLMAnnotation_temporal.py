# 时序驾驶场景 LLM 标注脚本（本地模型版，支持 Qwen2.5-VL 等）
# 数据结构: D:\User_data\M_Gaze_screen\Journal_2_data\{Participant}\{Scenario}\merged_ui\*.jpg

import argparse
import json
import os
import csv as csvmod
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from Prompt import Prompt as LLMPrompt

THIS = os.path.dirname(__file__)
DEFAULT_DATA_ROOT = r"D:\User_data\M_Gaze_screen\Journal_2_data"
DEFAULT_OUTPUT = os.path.join(THIS, "annotation_outputs", "temporal_annotation_results.json")
DEFAULT_MODEL_DIR = r"D:\models\Qwen2.5-VL-3B-Instruct"

CONTEXT_FRAMES = 10
STRIDE = 10
IMAGE_SIZE = 336
NUM_WORKERS = 8


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--participants", type=str, nargs="+", default=None)
    parser.add_argument("--scenarios", type=str, nargs="+", default=None)
    parser.add_argument("--context-frames", type=int, default=CONTEXT_FRAMES)
    parser.add_argument("--stride", type=int, default=STRIDE)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-dir", type=str, default=DEFAULT_MODEL_DIR)
    return parser.parse_args()


def _extract_timestamp(filename):
    m = re.search(r"merged_(\d+\.\d+)\.jpg$", filename)
    return float(m.group(1)) if m else 0.0


def load_vehicle_log(csv_path):
    """Load vehicle_data.csv. Returns list of dicts with 'rel_t' (seconds from first row),
    or None if the file is missing / unreadable."""
    rows = []
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            for r in csvmod.DictReader(f):
                ts = datetime.strptime(r["timestamp_str"], "%Y-%m-%d %H:%M:%S.%f")
                rows.append({
                    "ts": ts,
                    "speed": float(r["Speed"]),
                    "acceleration": float(r["Acceleration"]),
                    "steering": float(r["Steering"]),
                    "throttle": float(r["Throttle"]),
                    "brake": float(r["Brake"]),
                    "yaw": float(r["Yaw"]),
                    "dist_left": float(r["dist_left"]),
                    "dist_right": float(r["dist_right"]),
                })
    except Exception as e:
        print(f"  [vehicle log] Warning: could not load {csv_path}: {e}")
        return None
    if not rows:
        return None
    t0 = rows[0]["ts"]
    for row in rows:
        row["rel_t"] = (row["ts"] - t0).total_seconds()
    # Signed acceleration: Δspeed / Δt  (km/h → m/s: ÷3.6)
    for i, row in enumerate(rows):
        if i == 0:
            j = 1 if len(rows) > 1 else 0
            dt = rows[j]["rel_t"] - rows[0]["rel_t"]
            ds = rows[j]["speed"] - rows[0]["speed"]
        else:
            dt = row["rel_t"] - rows[i - 1]["rel_t"]
            ds = row["speed"] - rows[i - 1]["speed"]
        row["accel_signed"] = (ds / 3.6 / dt) if dt > 1e-6 else 0.0
    return rows


def lookup_vehicle(vehicle_rows, target_rel, max_dt=0.5):
    """Binary-search for the vehicle-log row nearest to target_rel (seconds from session start).
    Returns None if the nearest row is more than max_dt seconds away (sensor drop protection).
    """
    if not vehicle_rows:
        return None
    lo, hi = 0, len(vehicle_rows) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if vehicle_rows[mid]["rel_t"] < target_rel:
            lo = mid + 1
        else:
            hi = mid
    if lo > 0 and (
        lo == len(vehicle_rows)
        or abs(vehicle_rows[lo - 1]["rel_t"] - target_rel) <= abs(vehicle_rows[lo]["rel_t"] - target_rel)
    ):
        lo -= 1
    if abs(vehicle_rows[lo]["rel_t"] - target_rel) > max_dt:
        return None  # sensor drop — no valid data within threshold
    return vehicle_rows[lo]


def scan_sequences(data_root, participants=None, scenarios=None):
    sequences = []
    root = Path(data_root)
    for p_dir in sorted(root.iterdir()):
        if not p_dir.is_dir():
            continue
        if participants and p_dir.name not in participants:
            continue
        for s_dir in sorted(p_dir.iterdir()):
            if not s_dir.is_dir():
                continue
            if scenarios and s_dir.name not in scenarios:
                continue
            merged_ui = s_dir / "merged_ui"
            if not merged_ui.exists():
                continue
            jpg_files = sorted(merged_ui.glob("merged_*.jpg"), key=lambda f: _extract_timestamp(f.name))
            if not jpg_files:
                continue
            vehicle_csv = s_dir / "vehicle_data.csv"
            vehicle_rows = load_vehicle_log(str(vehicle_csv)) if vehicle_csv.exists() else None
            # Pre-match every frame to its nearest CSV row (threshold 0.5s)
            # img_t0 anchors both timelines: rel_t = img_ts - img_t0
            img_t0 = _extract_timestamp(jpg_files[0].name)
            frames = []
            for jf in jpg_files:
                ts = _extract_timestamp(jf.name)
                veh = lookup_vehicle(vehicle_rows, ts - img_t0) if vehicle_rows else None
                frames.append({"timestamp": ts, "image_path": str(jf), "vehicle": veh})
            matched = sum(1 for fr in frames if fr["vehicle"] is not None)
            if vehicle_rows:
                print(f"  [vehicle log] {p_dir.name}/{s_dir.name}: {matched}/{len(frames)} frames matched (threshold=0.5s)")
            sequences.append({
                "participant": p_dir.name,
                "scenario": s_dir.name,
                "frames": frames,
            })
    return sequences


def build_annotation_targets(sequences, context_frames, stride, sparse_seconds=None, future_seconds=None):
    """Build annotation targets with non-uniform temporal sampling.

    Past sparse  : one frame per second in `sparse_seconds` (e.g. 5s,4s,3s,2s ago).
    Past dense   : last `context_frames` frames (~1s at 10Hz), ending at target.
    Future sparse: one frame per second in `future_seconds` (e.g. 1s,2s ahead).
    Frames are ordered oldest → newest so VLM sees time flowing forward.
    """
    if sparse_seconds is None:
        sparse_seconds = []
    if future_seconds is None:
        future_seconds = []
    fps = 10
    targets = []
    max_lookback = context_frames + int(max(sparse_seconds, default=0) * fps)
    max_lookahead = int(max(future_seconds, default=0) * fps)
    for seq in sequences:
        frames = seq["frames"]
        for idx in range(max_lookback - 1, len(frames) - max_lookahead, stride):
            # Past sparse frames (oldest first)
            sparse = []
            for s in sorted(sparse_seconds, reverse=True):
                sparse_idx = idx - int(s * fps)
                if sparse_idx >= 0:
                    f = dict(frames[sparse_idx])
                    f["sample_type"] = f"sparse_{s}s"
                    sparse.append(f)
            # Past dense frames (current moment is last)
            dense = []
            for f in frames[idx - context_frames + 1: idx + 1]:
                fd = dict(f)
                fd["sample_type"] = "dense_1s"
                dense.append(fd)
            # Future sparse frames (1s, 2s ahead)
            future = []
            for s in sorted(future_seconds):
                future_idx = idx + int(s * fps)
                if future_idx < len(frames):
                    f = dict(frames[future_idx])
                    f["sample_type"] = f"future_{s}s"
                    future.append(f)
            context = sparse + dense + future  # oldest → newest
            targets.append({
                "participant": seq["participant"],
                "scenario": seq["scenario"],
                "target_frame_idx": idx,
                "target_timestamp": frames[idx]["timestamp"],
                "context_frames": context,
            })
    return targets


def _load_one_frame(frame_info, image_size):
    img = Image.open(frame_info["image_path"]).convert("RGB")
    if image_size:
        img = img.resize((image_size, image_size), Image.Resampling.BILINEAR)
    return img


def load_context_images(target, image_size, num_workers):
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        return list(executor.map(lambda f: _load_one_frame(f, image_size), target["context_frames"]))


def format_messages(prompt, images, timestamps, sample_types=None, vehicle_telemetry=None):
    # Find current frame: last dense_1s frame (before any future frames)
    current_idx = len(timestamps) - 1
    if sample_types:
        for j in range(len(sample_types) - 1, -1, -1):
            if sample_types[j] == "dense_1s":
                current_idx = j
                break
    t_current = timestamps[current_idx]
    user_content = []
    for i, (img, ts) in enumerate(zip(images, timestamps)):
        delta = ts - t_current
        stype = sample_types[i] if sample_types else ("dense_1s" if i >= len(timestamps) - 10 else "sparse")
        if i == current_idx:
            label = "[CURRENT FRAME | dt=0.00s]"
        elif stype.startswith("future"):
            label = f"[Future context | dt=+{delta:.1f}s — additional info only]"
        elif stype.startswith("sparse"):
            label = f"[Past context | dt={delta:.1f}s]"
        else:
            label = f"[Recent frame | dt={delta:.2f}s]"
        user_content.append({"type": "text", "text": label})
        user_content.append({"type": "image", "image": img})
    # Vehicle telemetry block — driver actions (throttle/brake/steering) are ground-truth labels
    if vehicle_telemetry and any(v is not None for v in vehicle_telemetry):
        tlines = ["[Vehicle telemetry — driver actions are ground truth]",
                  "  format: dt | speed accel | driver: throttle brake steering | (current only) lane dist"]
        for i, (ts, veh) in enumerate(zip(timestamps, vehicle_telemetry)):
            if veh is None:
                continue
            dt = ts - t_current
            is_current = (i == len(timestamps) - 1)
            stype = (sample_types[i] if sample_types else "dense") if not is_current else "current"
            prefix = "[CURRENT ] " if is_current else f"[t={dt:+.2f}s] "
            tag = " [sparse]" if (stype or "").startswith("sparse") else ""
            line = (
                f"{prefix}{tag} "
                f"speed={veh['speed']:5.1f} km/h  accel={veh['accel_signed']:+.2f} m/s²  |  "
                f"driver: throttle={veh['throttle']:.3f}  brake={veh['brake']:.3f}  steer={veh['steering']:+.4f}"
            )
            if is_current:
                line += f"  |  dist_left={veh['dist_left']:.1f}m  dist_right={veh['dist_right']:.1f}m"
            tlines.append(line)
        # Speed-delta summary over the full window (ground-truth motion trend)
        valid = [(ts, v) for ts, v in zip(timestamps, vehicle_telemetry) if v is not None]
        if len(valid) >= 2:
            spd_first = valid[0][1]["speed"]
            spd_last  = valid[-1][1]["speed"]
            dt_window = valid[-1][0] - valid[0][0]
            d_spd = spd_last - spd_first
            if d_spd > 0.5:
                trend_str = "ACCELERATING"
            elif d_spd < -0.5:
                trend_str = "DECELERATING"
            else:
                trend_str = "CONSTANT SPEED"
            tlines.append(
                f"[Speed trend over {dt_window:.1f}s] "
                f"{spd_first:.1f} → {spd_last:.1f} km/h  "
                f"(Δ{d_spd:+.1f} km/h, net: {trend_str})"
            )
        user_content.append({"type": "text", "text": "\n".join(tlines)})
    n_frames = len(images)
    duration = timestamps[-1] - timestamps[0]
    msg = prompt.user_message
    msg = msg.replace("{n_frames}", str(n_frames))
    msg = re.sub(r"\{duration(?::[^}]*)?\}", f"{duration:.1f}", msg)
    user_content.append({"type": "text", "text": msg})
    return [{"role": "system", "content": prompt.system_message}, {"role": "user", "content": user_content}]


def run_temporal_mode(model, processor, mode_name, prompt, targets, image_size, num_workers, max_new_tokens, checkpoint_path=None, done_keys=None):
    outputs = []
    start = time.time()
    if checkpoint_path:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    for i, target in enumerate(targets):
        key = (target["participant"], target["scenario"], target["target_frame_idx"])
        if done_keys and key in done_keys:
            print(f"  [{mode_name}] {i+1}/{len(targets)} SKIP (already done)")
            continue
        print(f"  [{mode_name}] {i+1}/{len(targets)} {target['participant']}/{target['scenario']} frame={target['target_frame_idx']}")
        images = load_context_images(target, image_size=image_size, num_workers=num_workers)
        timestamps = [f["timestamp"] for f in target["context_frames"]]
        sample_types = [f.get("sample_type", "dense_1s") for f in target["context_frames"]]
        # vehicle data pre-matched at scan time — just read from frame dicts
        vehicle_telemetry = [f.get("vehicle") for f in target["context_frames"]]
        if not any(v is not None for v in vehicle_telemetry):
            vehicle_telemetry = None
        messages = format_messages(prompt, images, timestamps, sample_types=sample_types, vehicle_telemetry=vehicle_telemetry)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        all_images = [item["image"] for msg in messages if isinstance(msg["content"], list) for item in msg["content"] if item.get("type") == "image"]
        inputs = processor(text=[text], images=all_images if all_images else None, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        new_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
        response_text = processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()
        record = {"participant": target["participant"], "scenario": target["scenario"], "target_frame_idx": target["target_frame_idx"], "target_timestamp": target["target_timestamp"], "mode": mode_name, "response": response_text}
        outputs.append(record)
        if checkpoint_path:
            with open(checkpoint_path, "a", encoding="utf-8") as ckpt:
                ckpt.write(json.dumps(record, ensure_ascii=False) + "\n")
    return outputs, time.time() - start


def main():
    args = parse_args()
    print(f"Scanning data from: {args.data_root}")
    sequences = scan_sequences(args.data_root, args.participants, args.scenarios)
    print(f"Found {len(sequences)} sequence(s), {sum(len(s['frames']) for s in sequences)} total frames.")
    targets = build_annotation_targets(sequences, args.context_frames, args.stride, sparse_seconds=[2, 3, 4, 5], future_seconds=[1, 2])
    print(f"Built {len(targets)} annotation target(s) (context={args.context_frames} frames = {args.context_frames/10:.1f}s, stride={args.stride}).")

    print(f"Loading model from {args.model_dir} ...")
    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True).eval()
    print("Model loaded.")

    freedom_prompt = LLMPrompt(seed="FREEDOM")
    structured_prompt = LLMPrompt(seed="STRUCTURED")

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base, ext = os.path.splitext(args.output)
    stamped_output = f"{base}_{run_ts}{ext}"

    out_dir = os.path.dirname(args.output)
    os.makedirs(out_dir, exist_ok=True)
    freedom_ckpt = os.path.join(out_dir, f"ckpt_freedom_{run_ts}.jsonl")
    structured_ckpt = os.path.join(out_dir, f"ckpt_structured_{run_ts}.jsonl")

    # Load already-completed items to allow resume
    def load_ckpt(path):
        done, records = set(), []
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    done.add((r["participant"], r["scenario"], r["target_frame_idx"]))
                    records.append(r)
        return done, records

    freedom_done, freedom_prev = load_ckpt(freedom_ckpt)
    structured_done, structured_prev = load_ckpt(structured_ckpt)

    print(f"Running FREEDOM mode... ({len(freedom_done)} already done, {len(targets)-len(freedom_done)} remaining)")
    freedom_new, freedom_elapsed = run_temporal_mode(model, processor, "freedom", freedom_prompt, targets, args.image_size, args.num_workers, args.max_new_tokens, checkpoint_path=freedom_ckpt, done_keys=freedom_done)
    freedom_outputs = freedom_prev + freedom_new

    print(f"Running STRUCTURED mode... ({len(structured_done)} already done, {len(targets)-len(structured_done)} remaining)")
    structured_new, structured_elapsed = run_temporal_mode(model, processor, "structured", structured_prompt, targets, args.image_size, args.num_workers, args.max_new_tokens, checkpoint_path=structured_ckpt, done_keys=structured_done)
    structured_outputs = structured_prev + structured_new

    Key = lambda o: (o["participant"], o["scenario"], o["target_frame_idx"])
    freedom_map = {Key(o): o["response"] for o in freedom_outputs}
    structured_map = {Key(o): o["response"] for o in structured_outputs}
    merged = [{"participant": t["participant"], "scenario": t["scenario"], "target_frame_idx": t["target_frame_idx"], "target_timestamp": t["target_timestamp"], "freedom_response": freedom_map.get((t["participant"], t["scenario"], t["target_frame_idx"]), ""), "structured_response": structured_map.get((t["participant"], t["scenario"], t["target_frame_idx"]), "")} for t in targets]

    # Group by (scenario, participant) and write one file each
    # Output structure: annotation_outputs/{scenario}/{participant}_{run_ts}.json
    from collections import defaultdict
    groups = defaultdict(list)
    for record in merged:
        groups[(record["scenario"], record["participant"])].append(record)

    base_out = os.path.dirname(args.output)
    saved_files = []
    for (scenario, participant), records in groups.items():
        scenario_dir = os.path.join(base_out, scenario)
        os.makedirs(scenario_dir, exist_ok=True)
        out_path = os.path.join(scenario_dir, f"{participant}_{run_ts}.json")
        payload = {
            "config": {"run_timestamp": run_ts, "participant": participant, "scenario": scenario,
                       "data_root": args.data_root, "model_dir": args.model_dir,
                       "context_frames": args.context_frames, "fps": 10,
                       "stride": args.stride, "max_new_tokens": args.max_new_tokens,
                       "image_size": args.image_size},
            "timing": {"freedom_seconds": freedom_elapsed, "structured_seconds": structured_elapsed},
            "results": records,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        saved_files.append(out_path)
        print(f"  Saved {len(records)} record(s) → {out_path}")
    print(f"Done. {len(saved_files)} file(s) written.")


if __name__ == "__main__":
    main()
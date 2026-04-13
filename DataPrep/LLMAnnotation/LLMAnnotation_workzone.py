# Workzone-aware single-frame LLM annotation pipeline
# Input : D:\User_data\M_Gaze_screen\Journal_2_data\{P}\{S*}\merged_ui\merged_*.jpg
# Events: workzone driving data.xlsx  (wz1/wz2/wz3 per participant+scenario)
# Output: annotation_outputs/workzone/{participant}_{run_ts}.json

import argparse
import json
import os
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

DEFAULT_DATA_ROOT  = r"D:\User_data\M_Gaze_screen\Journal_2_data"
DEFAULT_XLSX       = r"D:\User_data\M_Gaze_screen\Journal_2_data\workzone driving data.xlsx"
DEFAULT_OUTPUT_DIR = os.path.join(THIS_DIR, "annotation_outputs", "workzone")
DEFAULT_MODEL_DIR  = r"D:\models\Qwen2.5-VL-3B-Instruct"
IMAGE_SIZE         = 336
NUM_WORKERS        = 4
EVENT_PAD_S        = 2.0   # seconds before/after workzone window
SAMPLE_HZ          = 2     # 2 fps → every 5th frame at 10 Hz


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root",    default=DEFAULT_DATA_ROOT)
    p.add_argument("--xlsx",         default=DEFAULT_XLSX)
    p.add_argument("--output-dir",   default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--model-dir",    default=DEFAULT_MODEL_DIR)
    p.add_argument("--participants",  nargs="+", default=None)
    p.add_argument("--scenarios",     nargs="+", default=None)
    p.add_argument("--image-size",   type=int, default=IMAGE_SIZE)
    p.add_argument("--num-workers",  type=int, default=NUM_WORKERS)
    p.add_argument("--max-new-tokens", type=int, default=256)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Xlsx parsing
# ---------------------------------------------------------------------------

def _extract_ts(cell_value):
    """Extract float timestamp from 'merged_1234567890.123000' or return None."""
    if cell_value is None:
        return None
    s = str(cell_value).strip()
    m = re.search(r"merged_(\d+(?:\.\d+)?)", s)
    return float(m.group(1)) if m else None


def parse_workzone_xlsx(xlsx_path):
    """Return list of dicts:
       {participant, scenario_prefix, wz_id, t_start, t_end}
       scenario_prefix is e.g. 'S1' — matched against folder names.
    """
    import openpyxl
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    events = []
    for ws in wb.worksheets:
        for row in ws.iter_rows(values_only=True):
            if row[0] is None:
                continue
            label = str(row[0]).strip()
            # Expect format P2_S1, P3_S2, etc.
            m = re.match(r"(P\d+)_(S\d+)", label)
            if not m:
                continue
            participant = m.group(1)
            scenario_prefix = m.group(2)
            # wz1: cols 1,2
            for wz_id, (si, ei) in enumerate([
                (1, 2),    # wz1: wz1_start → wz1_end
                (5, 6),    # wz2: wz2_start → wz2_end
                (10, 11),  # wz3: wz3_start_warning → wz3_end
            ], start=1):
                t_start = _extract_ts(row[si] if len(row) > si else None)
                t_end   = _extract_ts(row[ei] if len(row) > ei else None)
                if t_start is not None and t_end is not None:
                    events.append({
                        "participant":    participant,
                        "scenario_prefix": scenario_prefix,
                        "wz_id":          wz_id,
                        "t_start":        t_start,
                        "t_end":          t_end,
                    })
    return events


# ---------------------------------------------------------------------------
# Frame discovery & sampling
# ---------------------------------------------------------------------------

def _ts_from_filename(name):
    m = re.search(r"merged_(\d+(?:\.\d+)?)", name)
    return float(m.group(1)) if m else None


def find_scenario_dir(data_root, participant, scenario_prefix):
    """Find folder matching {data_root}/{participant}/{scenario_prefix*}."""
    p_dir = Path(data_root) / participant
    if not p_dir.exists():
        return None
    for s_dir in sorted(p_dir.iterdir()):
        if s_dir.is_dir() and s_dir.name.startswith(scenario_prefix):
            return s_dir
    return None


def sample_event_frames(data_root, event, pad_s=EVENT_PAD_S, sample_hz=SAMPLE_HZ):
    """Return list of {timestamp, image_path} for one workzone event at sample_hz fps."""
    s_dir = find_scenario_dir(data_root, event["participant"], event["scenario_prefix"])
    if s_dir is None:
        return []
    merged_ui = s_dir / "merged_ui"
    if not merged_ui.exists():
        return []
    all_jpgs = sorted(merged_ui.glob("merged_*.jpg"),
                      key=lambda f: _ts_from_filename(f.name) or 0)
    if not all_jpgs:
        return []

    t_lo = event["t_start"] - pad_s
    t_hi = event["t_end"]   + pad_s
    # native fps ≈ 10 Hz → step = 10 / sample_hz
    native_fps = 10
    step = max(1, native_fps // sample_hz)

    window = [f for f in all_jpgs
              if t_lo <= (_ts_from_filename(f.name) or 0) <= t_hi]

    # subsample at step
    sampled = window[::step]
    return [{"timestamp": _ts_from_filename(f.name), "image_path": str(f)}
            for f in sampled]


def build_targets(data_root, events, participants=None, scenarios=None):
    targets = []
    for ev in events:
        if participants and ev["participant"] not in participants:
            continue
        if scenarios and ev["scenario_prefix"] not in scenarios:
            continue
        frames = sample_event_frames(data_root, ev)
        if not frames:
            print(f"  [WARN] no frames for {ev['participant']}/{ev['scenario_prefix']} wz{ev['wz_id']}")
            continue
        for fr in frames:
            targets.append({
                "participant":    ev["participant"],
                "scenario_prefix": ev["scenario_prefix"],
                "wz_id":          ev["wz_id"],
                "t_start":        ev["t_start"],
                "t_end":          ev["t_end"],
                "timestamp":      fr["timestamp"],
                "image_path":     fr["image_path"],
            })
    return targets


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def load_image(path, image_size):
    img = Image.open(path).convert("RGB")
    if image_size:
        img = img.resize((image_size, image_size), Image.Resampling.BILINEAR)
    return img


def build_messages(prompt, image):
    user_content = [
        {"type": "image", "image": image},
        {"type": "text",  "text": prompt.user_message},
    ]
    return [
        {"role": "system", "content": prompt.system_message},
        {"role": "user",   "content": user_content},
    ]


def run_mode(model, processor, mode_name, prompt, targets,
             image_size, num_workers, max_new_tokens,
             checkpoint_path=None, done_keys=None):
    outputs = []
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True) if checkpoint_path else None

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        for i, t in enumerate(targets):
            key = (t["participant"], t["scenario_prefix"], t["wz_id"], t["timestamp"])
            if done_keys and key in done_keys:
                print(f"  [{mode_name}] {i+1}/{len(targets)} SKIP")
                continue
            print(f"  [{mode_name}] {i+1}/{len(targets)} "
                  f"{t['participant']}/{t['scenario_prefix']} wz{t['wz_id']} ts={t['timestamp']:.1f}")

            image = load_image(t["image_path"], image_size)
            messages = build_messages(prompt, image)
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(
                text=[text], images=[image],
                return_tensors="pt", padding=True
            ).to(model.device)

            with torch.no_grad():
                gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
            new_ids = gen_ids[:, inputs["input_ids"].shape[1]:]
            response = processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()

            record = {
                "participant":    t["participant"],
                "scenario_prefix": t["scenario_prefix"],
                "wz_id":          t["wz_id"],
                "t_start":        t["t_start"],
                "t_end":          t["t_end"],
                "timestamp":      t["timestamp"],
                "mode":           mode_name,
                "response":       response,
            }
            outputs.append(record)
            if checkpoint_path:
                with open(checkpoint_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return outputs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_ckpt(path):
    done, records = set(), []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                done.add((r["participant"], r["scenario_prefix"], r["wz_id"], r["timestamp"]))
                records.append(r)
    return done, records


def main():
    args = parse_args()

    print(f"Parsing workzone events from: {args.xlsx}")
    events = parse_workzone_xlsx(args.xlsx)
    print(f"  Found {len(events)} workzone events.")

    targets = build_targets(args.data_root, events, args.participants, args.scenarios)
    print(f"  Built {len(targets)} frame targets.")

    print(f"Loading model from {args.model_dir} ...")
    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16,
        device_map="cuda", trust_remote_code=True
    ).eval()
    print("Model loaded.")

    freedom_prompt    = LLMPrompt(seed="FREEDOM")
    structured_prompt = LLMPrompt(seed="STRUCTURED")

    run_ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.output_dir, exist_ok=True)
    freedom_ckpt    = os.path.join(args.output_dir, f"ckpt_freedom_{run_ts}.jsonl")
    structured_ckpt = os.path.join(args.output_dir, f"ckpt_structured_{run_ts}.jsonl")

    freedom_done,    freedom_prev    = load_ckpt(freedom_ckpt)
    structured_done, structured_prev = load_ckpt(structured_ckpt)

    print(f"Running FREEDOM ({len(freedom_done)} done, {len(targets)-len(freedom_done)} remaining)...")
    freedom_new = run_mode(model, processor, "freedom", freedom_prompt, targets,
                           args.image_size, args.num_workers, args.max_new_tokens,
                           checkpoint_path=freedom_ckpt, done_keys=freedom_done)
    freedom_outputs = freedom_prev + freedom_new

    print(f"Running STRUCTURED ({len(structured_done)} done, {len(targets)-len(structured_done)} remaining)...")
    structured_new = run_mode(model, processor, "structured", structured_prompt, targets,
                              args.image_size, args.num_workers, args.max_new_tokens,
                              checkpoint_path=structured_ckpt, done_keys=structured_done)
    structured_outputs = structured_prev + structured_new

    # Merge by (participant, scenario_prefix, wz_id, timestamp)
    Key = lambda o: (o["participant"], o["scenario_prefix"], o["wz_id"], o["timestamp"])
    freedom_map    = {Key(o): o["response"] for o in freedom_outputs}
    structured_map = {Key(o): o["response"] for o in structured_outputs}

    merged = [
        {
            "participant":      t["participant"],
            "scenario_prefix":  t["scenario_prefix"],
            "wz_id":            t["wz_id"],
            "t_start":          t["t_start"],
            "t_end":            t["t_end"],
            "timestamp":        t["timestamp"],
            "freeform_response":   freedom_map.get(Key(t), ""),
            "structured_response": structured_map.get(Key(t), ""),
        }
        for t in targets
    ]

    out_path = os.path.join(args.output_dir, f"workzone_annotations_{run_ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"run_ts": run_ts, "total": len(merged), "results": merged},
                  f, ensure_ascii=False, indent=2)
    print(f"Saved {len(merged)} annotations → {out_path}")


if __name__ == "__main__":
    main()

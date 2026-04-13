# Workzone-aware single-frame annotation pipeline — OpenAI GPT (Batch API)
#
# Workflow:
#   STEP 1  build   → 生成 batch request .jsonl 文件（本地，不调用 API）
#   STEP 2  submit  → 上传 .jsonl，提交 Batch Job（~5 min ~ 24h，通常 <1h）
#   STEP 3  collect → 轮询 Job 状态，完成后下载结果，保存最终 JSON
#
# Usage examples:
#   python LLMAnnotation_workzone_GPT.py build
#   python LLMAnnotation_workzone_GPT.py submit --batch-input batch_requests_20260410_120000.jsonl
#   python LLMAnnotation_workzone_GPT.py collect --batch-id batch_abc123
#
# Cost tip:
#   Batch API = 50% off vs real-time.  ~$20 for 36K frames with gpt-4o-mini.

import argparse
import base64
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from openai import OpenAI

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from Prompt import Prompt as LLMPrompt

# ---------------------------------------------------------------------------
# Defaults (override via CLI args)
# ---------------------------------------------------------------------------
DEFAULT_DATA_ROOT  = r"D:\User_data\M_Gaze_screen\Journal_2_data"
DEFAULT_XLSX       = r"D:\User_data\M_Gaze_screen\Journal_2_data\workzone driving data.xlsx"
DEFAULT_OUTPUT_DIR = os.path.join(THIS_DIR, "annotation_outputs", "workzone_gpt")
DEFAULT_MODEL      = "gpt-4o-mini"   # swap to "gpt-4o" for higher quality
IMAGE_SIZE         = 1024            # px — resize before encoding (saves tokens)
EVENT_PAD_S        = 2.0
SAMPLE_HZ          = 2               # 2 fps


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    sub = p.add_subparsers(dest="command", required=True)

    # ---- build ----
    b = sub.add_parser("build", help="Build batch request .jsonl (no API calls)")
    b.add_argument("--data-root",    default=DEFAULT_DATA_ROOT)
    b.add_argument("--xlsx",         default=DEFAULT_XLSX)
    b.add_argument("--output-dir",   default=DEFAULT_OUTPUT_DIR)
    b.add_argument("--model",        default=DEFAULT_MODEL)
    b.add_argument("--participants",  nargs="+", default=None)
    b.add_argument("--scenarios",     nargs="+", default=None)
    b.add_argument("--image-size",   type=int, default=IMAGE_SIZE)
    b.add_argument("--max-tokens",   type=int, default=512)
    b.add_argument("--detail",       default="high", choices=["low", "high"])

    # ---- submit ----
    s = sub.add_parser("submit", help="Upload .jsonl and create Batch Job")
    s.add_argument("--batch-input",  required=True, help="Path to .jsonl from 'build'")
    s.add_argument("--output-dir",   default=DEFAULT_OUTPUT_DIR)

    # ---- run (real-time) ----
    r = sub.add_parser("run", help="Real-time API calls — fast but 2x cost vs batch")
    r.add_argument("--data-root",    default=DEFAULT_DATA_ROOT)
    r.add_argument("--xlsx",         default=DEFAULT_XLSX)
    r.add_argument("--output-dir",   default=DEFAULT_OUTPUT_DIR)
    r.add_argument("--model",        default=DEFAULT_MODEL)
    r.add_argument("--participants",  nargs="+", default=None)
    r.add_argument("--scenarios",     nargs="+", default=None)
    r.add_argument("--image-size",   type=int, default=IMAGE_SIZE)
    r.add_argument("--max-tokens",   type=int, default=512)
    r.add_argument("--detail",       default="high", choices=["low", "high"])
    r.add_argument("--concurrency",  type=int, default=2, help="Parallel API calls")

    # ---- collect ----
    c = sub.add_parser("collect", help="Poll batch status, download & merge results")
    c.add_argument("--batch-id",     required=True, help="Batch Job ID from 'submit'")
    c.add_argument("--output-dir",   default=DEFAULT_OUTPUT_DIR)
    c.add_argument("--poll-interval", type=int, default=60, help="Seconds between status checks")
    # Need the original targets to reconstruct metadata
    c.add_argument("--data-root",    default=DEFAULT_DATA_ROOT)
    c.add_argument("--xlsx",         default=DEFAULT_XLSX)
    c.add_argument("--participants",  nargs="+", default=None)
    c.add_argument("--scenarios",     nargs="+", default=None)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Xlsx parsing (identical logic to LLMAnnotation_workzone.py)
# ---------------------------------------------------------------------------
def _extract_ts(cell_value):
    if cell_value is None:
        return None
    s = str(cell_value).strip()
    m = re.search(r"merged_(\d+(?:\.\d+)?)", s)
    return float(m.group(1)) if m else None


def parse_workzone_xlsx(xlsx_path):
    import openpyxl
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    events = []
    for ws in wb.worksheets:
        for row in ws.iter_rows(values_only=True):
            if row[0] is None:
                continue
            label = str(row[0]).strip()
            m = re.match(r"(P\d+)_(S\d+)", label)
            if not m:
                continue
            participant    = m.group(1)
            scenario_prefix = m.group(2)
            for wz_id, (si, ei) in enumerate([(1, 2), (5, 6), (10, 11)], start=1):
                t_start = _extract_ts(row[si] if len(row) > si else None)
                t_end   = _extract_ts(row[ei] if len(row) > ei else None)
                if t_start is not None and t_end is not None:
                    events.append({
                        "participant":     participant,
                        "scenario_prefix": scenario_prefix,
                        "wz_id":           wz_id,
                        "t_start":         t_start,
                        "t_end":           t_end,
                    })
    return events


# ---------------------------------------------------------------------------
# Frame discovery & sampling
# ---------------------------------------------------------------------------
def _ts_from_filename(name):
    m = re.search(r"merged_(\d+(?:\.\d+)?)", name)
    return float(m.group(1)) if m else None


def find_scenario_dir(data_root, participant, scenario_prefix):
    p_dir = Path(data_root) / participant
    if not p_dir.exists():
        return None
    for s_dir in sorted(p_dir.iterdir()):
        if s_dir.is_dir() and s_dir.name.startswith(scenario_prefix):
            return s_dir
    return None


def sample_event_frames(data_root, event, pad_s=EVENT_PAD_S, sample_hz=SAMPLE_HZ):
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
    step = max(1, 10 // sample_hz)  # native 10 Hz
    window  = [f for f in all_jpgs if t_lo <= (_ts_from_filename(f.name) or 0) <= t_hi]
    sampled = window[::step]
    return [{"timestamp": _ts_from_filename(f.name), "image_path": str(f)} for f in sampled]


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
            targets.append({**ev, **fr})
    return targets


# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------
def encode_image(path: str, image_size: int) -> str:
    """Return base64-encoded JPEG string, optionally resized."""
    from PIL import Image
    img = Image.open(path).convert("RGB")
    if image_size:
        img.thumbnail((image_size, image_size), Image.Resampling.LANCZOS)
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# Batch request building
# ---------------------------------------------------------------------------
def _make_request(custom_id: str, model: str, system_msg: str, user_msg: str,
                  b64_image: str, detail: str, max_tokens: int) -> dict:
    """Build one line for the batch .jsonl file."""
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "response_format": {"type": "json_object"},
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system_msg},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url":    f"data:image/jpeg;base64,{b64_image}",
                                "detail": detail,
                            },
                        },
                        {"type": "text", "text": user_msg},
                    ],
                },
            ],
        },
    }


def _make_custom_id(mode: str, t: dict) -> str:
    """Unique, reversible ID encoding all metadata."""
    return (
        f"{mode}__{t['participant']}__{t['scenario_prefix']}"
        f"__wz{t['wz_id']}__{t['timestamp']:.3f}"
    )


def parse_custom_id(custom_id: str) -> dict:
    parts = custom_id.split("__")
    return {
        "mode":            parts[0],
        "participant":     parts[1],
        "scenario_prefix": parts[2],
        "wz_id":           int(parts[3].replace("wz", "")),
        "timestamp":       float(parts[4]),
    }


# ---------------------------------------------------------------------------
# STEP 1: build
# ---------------------------------------------------------------------------
def cmd_build(args):
    print(f"Parsing workzone events from: {args.xlsx}")
    events  = parse_workzone_xlsx(args.xlsx)
    targets = build_targets(args.data_root, events, args.participants, args.scenarios)
    print(f"  {len(events)} events → {len(targets)} frame targets.")

    freedom_prompt    = LLMPrompt(seed="FREEDOM")
    structured_prompt = LLMPrompt(seed="STRUCTURED")

    os.makedirs(args.output_dir, exist_ok=True)
    run_ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_jsonl  = os.path.join(args.output_dir, f"batch_requests_{run_ts}.jsonl")

    total = 0
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for i, t in enumerate(targets):
            print(f"  Encoding frame {i+1}/{len(targets)}: {t['image_path']}", end="\r")
            b64 = encode_image(t["image_path"], args.image_size)

            for mode, prompt in [("freedom", freedom_prompt), ("structured", structured_prompt)]:
                cid     = _make_custom_id(mode, t)
                request = _make_request(
                    custom_id  = cid,
                    model      = args.model,
                    system_msg = prompt.system_message,
                    user_msg   = prompt.user_message,
                    b64_image  = b64,
                    detail     = args.detail,
                    max_tokens = args.max_tokens,
                )
                f.write(json.dumps(request, ensure_ascii=False) + "\n")
                total += 1

    print(f"\nWrote {total} requests → {out_jsonl}")
    print(f"\nNext step:\n  python LLMAnnotation_workzone_GPT.py submit --batch-input {out_jsonl}")


# ---------------------------------------------------------------------------
# STEP 2: submit
# ---------------------------------------------------------------------------
def cmd_submit(args):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    print(f"Uploading {args.batch_input} ...")
    with open(args.batch_input, "rb") as f:
        upload = client.files.create(file=f, purpose="batch")
    print(f"  File uploaded: {upload.id}")

    batch = client.batches.create(
        input_file_id    = upload.id,
        endpoint         = "/v1/chat/completions",
        completion_window = "24h",
    )
    print(f"  Batch job created: {batch.id}  status={batch.status}")

    # Save job info for later collect step
    os.makedirs(args.output_dir, exist_ok=True)
    info_path = os.path.join(args.output_dir, f"batch_job_{batch.id}.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump({"batch_id": batch.id, "file_id": upload.id,
                   "input": args.batch_input, "submitted_at": datetime.now().isoformat()},
                  f, indent=2)
    print(f"  Job info saved → {info_path}")
    print(f"\nNext step:\n  python LLMAnnotation_workzone_GPT.py collect --batch-id {batch.id}")


# ---------------------------------------------------------------------------
# STEP 3: collect
# ---------------------------------------------------------------------------
def cmd_collect(args):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # ---- Poll until finished ----
    print(f"Polling batch job {args.batch_id} ...")
    while True:
        batch = client.batches.retrieve(args.batch_id)
        status = batch.status
        completed = batch.request_counts.completed if batch.request_counts else "?"
        total     = batch.request_counts.total     if batch.request_counts else "?"
        print(f"  status={status}  completed={completed}/{total}")

        if status in ("completed", "failed", "expired", "cancelled"):
            break
        time.sleep(args.poll_interval)

    if status != "completed":
        print(f"[ERROR] Batch ended with status: {status}")
        sys.exit(1)

    # ---- Download output file ----
    print(f"Downloading results (file_id={batch.output_file_id}) ...")
    content = client.files.content(batch.output_file_id)
    raw_lines = content.text.strip().splitlines()
    print(f"  Downloaded {len(raw_lines)} result lines.")

    # ---- Parse results ----
    responses = {}   # custom_id → text
    errors    = []
    for line in raw_lines:
        obj = json.loads(line)
        cid = obj["custom_id"]
        if obj.get("error"):
            errors.append({"custom_id": cid, "error": obj["error"]})
            continue
        text = obj["response"]["body"]["choices"][0]["message"]["content"]
        responses[cid] = text

    if errors:
        print(f"  [WARN] {len(errors)} failed requests:")
        for e in errors[:5]:
            print(f"    {e}")

    # ---- Rebuild targets for metadata ----
    print("Rebuilding targets for metadata merge ...")
    events  = parse_workzone_xlsx(args.xlsx)
    targets = build_targets(args.data_root, events, args.participants, args.scenarios)

    def _key(t):
        return (t["participant"], t["scenario_prefix"], t["wz_id"], t["timestamp"])

    target_map = {_key(t): t for t in targets}

    # ---- Merge freedom + structured per frame ----
    merged = {}
    for cid, text in responses.items():
        meta = parse_custom_id(cid)
        k    = (meta["participant"], meta["scenario_prefix"], meta["wz_id"], meta["timestamp"])
        if k not in merged:
            t = target_map.get(k, {})
            merged[k] = {
                "participant":     meta["participant"],
                "scenario_prefix": meta["scenario_prefix"],
                "wz_id":           meta["wz_id"],
                "t_start":         t.get("t_start"),
                "t_end":           t.get("t_end"),
                "timestamp":       meta["timestamp"],
                "image_path":      t.get("image_path", ""),
                "freedom_response":    None,
                "structured_response": None,
            }
        # Try to parse as JSON, fall back to raw string
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = text

        if meta["mode"] == "freedom":
            merged[k]["freedom_response"] = parsed
        else:
            merged[k]["structured_response"] = parsed

    results = sorted(merged.values(),
                     key=lambda r: (r["participant"], r["scenario_prefix"], r["wz_id"], r["timestamp"]))

    # ---- Save ----
    os.makedirs(args.output_dir, exist_ok=True)
    run_ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.output_dir, f"workzone_gpt_annotations_{run_ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "batch_id":  args.batch_id,
            "run_ts":    run_ts,
            "total":     len(results),
            "errors":    len(errors),
            "results":   results,
        }, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(results)} annotated frames → {out_path}")
    if errors:
        err_path = out_path.replace(".json", "_errors.json")
        with open(err_path, "w", encoding="utf-8") as f:
            json.dump(errors, f, indent=2)
        print(f"Saved {len(errors)} errors → {err_path}")


# ---------------------------------------------------------------------------
# STEP: run (real-time API, concurrent)
# ---------------------------------------------------------------------------
def _call_api(client, model, system_msg, user_msg, b64_image, detail, max_tokens, json_mode=True):
    """Call OpenAI chat completions with exponential backoff on rate limit (429)."""
    # max_completion_tokens supported in openai>=1.30; fall back to max_tokens for older versions
    import openai as _oai
    _ver = tuple(int(x) for x in _oai.__version__.split(".")[:2])
    _tokens_key = "max_completion_tokens" if _ver >= (1, 30) else "max_tokens"

    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url":    f"data:image/jpeg;base64,{b64_image}",
                            "detail": detail,
                        },
                    },
                    {"type": "text", "text": user_msg},
                ],
            },
        ],
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    wait = 2
    for attempt in range(6):
        try:
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            msg = str(e)
            if "429" in msg or "rate_limit" in msg.lower():
                time.sleep(wait)
                wait = min(wait * 2, 60)
            else:
                raise
    raise RuntimeError("Max retries exceeded due to rate limit")


def cmd_run(args):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    print(f"Parsing workzone events from: {args.xlsx}")
    events  = parse_workzone_xlsx(args.xlsx)
    targets = build_targets(args.data_root, events, args.participants, args.scenarios)
    print(f"  {len(events)} events → {len(targets)} frame targets.")
    print(f"  Total API calls: {len(targets) * 2} (freedom + structured)")

    freedom_prompt    = LLMPrompt(seed="FREEDOM")
    structured_prompt = LLMPrompt(seed="STRUCTURED")

    # Build all tasks: (target, mode, prompt)
    tasks = []
    for t in targets:
        for mode, prompt in [("freedom", freedom_prompt), ("structured", structured_prompt)]:
            tasks.append((t, mode, prompt))

    results_map = {}  # (participant, scenario_prefix, wz_id, timestamp) → record
    done = 0
    total = len(tasks)

    def _process(task):
        t, mode, prompt = task
        b64 = encode_image(t["image_path"], args.image_size)
        text = _call_api(
            client, args.model,
            prompt.system_message, prompt.user_message,
            b64, args.detail, args.max_tokens,
            json_mode=(mode == "structured"),
        )
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = text
        return t, mode, parsed

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(_process, task): task for task in tasks}
        for future in as_completed(futures):
            try:
                t, mode, parsed = future.result()
            except Exception as e:
                task = futures[future]
                print(f"  [ERROR] {task[0]['participant']} {task[1]}: {e}")
                continue

            done += 1
            k = (t["participant"], t["scenario_prefix"], t["wz_id"], t["timestamp"])
            if k not in results_map:
                results_map[k] = {
                    "participant":     t["participant"],
                    "scenario_prefix": t["scenario_prefix"],
                    "wz_id":           t["wz_id"],
                    "t_start":         t["t_start"],
                    "t_end":           t["t_end"],
                    "timestamp":       t["timestamp"],
                    "image_path":      t["image_path"],
                    "freedom_response":    None,
                    "structured_response": None,
                }
            if mode == "freedom":
                results_map[k]["freedom_response"] = parsed
            else:
                results_map[k]["structured_response"] = parsed

            print(f"  [{done}/{total}] {t['participant']}/{t['scenario_prefix']} wz{t['wz_id']} ts={t['timestamp']:.1f} [{mode}]")

    results = sorted(results_map.values(),
                     key=lambda r: (r["participant"], r["scenario_prefix"], r["wz_id"], r["timestamp"]))

    os.makedirs(args.output_dir, exist_ok=True)
    run_ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(args.output_dir, f"workzone_gpt_rt_{run_ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"run_ts": run_ts, "model": args.model,
                   "total": len(results), "results": results},
                  f, ensure_ascii=False, indent=2)
    print(f"Saved {len(results)} annotated frames → {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    if args.command == "build":
        cmd_build(args)
    elif args.command == "submit":
        cmd_submit(args)
    elif args.command == "collect":
        cmd_collect(args)
    elif args.command == "run":
        cmd_run(args)

# Visualize GPT annotation results: overlay structured + freedom output onto the original image.
# Output: annotation_outputs/human_validation/{participant}_{scenario}_{wz_id}_{timestamp}.jpg
#
# Usage:
#   python visualize_annotations.py --input annotation_outputs/workzone_gpt/workzone_gpt_rt_XXXX.json

import argparse
import json
import os
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(THIS_DIR, "annotation_outputs", "human_validation")

# Layout constants
IMG_W        = 960    # final output width
PANEL_H      = 340    # height of the text panel below the image
FONT_SIZE    = 15
SMALL_FONT   = 13
PADDING      = 14
LINE_H       = 19
BG_COLOR     = (20, 20, 20)
HEADER_COLOR = (255, 200, 50)
KEY_COLOR    = (130, 200, 255)
VAL_COLOR    = (220, 220, 220)
FREE_COLOR   = (190, 240, 190)
NULL_COLOR   = (150, 150, 150)

RISK_COLORS = {
    "low":    (60, 180, 60),
    "medium": (220, 160, 0),
    "high":   (220, 50, 50),
}


def load_font(size):
    # Try common monospace fonts available on Windows
    candidates = [
        "consola.ttf", "Consolas.ttf",
        "cour.ttf",    # Courier New
        "lucon.ttf",   # Lucida Console
    ]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def draw_structured(draw, s, x, y, font, font_small):
    """Draw structured response fields as a two-column table."""
    if s is None:
        draw.text((x, y), "structured_response: null", font=font, fill=NULL_COLOR)
        return y + LINE_H

    fields = [
        ("workzone_present",  s.get("workzone_present", "—")),
        ("workzone_type",     s.get("workzone_type", "—")),
        ("traffic_condition", s.get("traffic_condition", "—")),
        ("primary_hazard",    s.get("primary_hazard", "—")),
        ("gaze_target",       s.get("gaze_target", "—")),
        ("attention",         s.get("attention_alignment", "—")),
        ("risk_level",        s.get("risk_level", "—")),
        ("action",            s.get("recommended_action", "—")),
    ]

    # Two columns
    col2_x = x + 300
    for i, (k, v) in enumerate(fields):
        cx = x if i % 2 == 0 else col2_x
        cy = y + (i // 2) * LINE_H
        draw.text((cx, cy), f"{k}: ", font=font_small, fill=KEY_COLOR)
        kw = draw.textlength(f"{k}: ", font=font_small)
        # Color risk_level specially
        val_col = RISK_COLORS.get(v, VAL_COLOR) if k == "risk_level" else VAL_COLOR
        draw.text((cx + kw, cy), str(v), font=font_small, fill=val_col)

    # Reasoning on its own line
    reasoning_y = y + (len(fields) // 2) * LINE_H + 4
    reasoning = s.get("reasoning", "")
    draw.text((x, reasoning_y), "reasoning: ", font=font_small, fill=KEY_COLOR)
    kw = draw.textlength("reasoning: ", font=font_small)
    # Wrap reasoning text
    wrapped = textwrap.wrap(reasoning, width=85)
    for li, line in enumerate(wrapped[:2]):  # max 2 lines
        draw.text((x + kw if li == 0 else x + kw, reasoning_y + li * LINE_H),
                  line, font=font_small, fill=VAL_COLOR)

    return reasoning_y + max(len(wrapped[:2]), 1) * LINE_H + 6


def draw_freedom(draw, text, x, y, w, font_small):
    """Draw freedom response as wrapped paragraph."""
    if text is None:
        draw.text((x, y), "freedom_response: null", font=font_small, fill=NULL_COLOR)
        return
    # First 3 lines only (keep it compact)
    wrapped = textwrap.wrap(text.replace("\n", " "), width=110)
    draw.text((x, y), "freedom: ", font=font_small, fill=KEY_COLOR)
    kw = draw.textlength("freedom: ", font=font_small)
    for li, line in enumerate(wrapped[:3]):
        draw.text((x + (kw if li == 0 else 0), y + li * LINE_H),
                  line, font=font_small, fill=FREE_COLOR)


def render_frame(record, font, font_small):
    img_path = record.get("image_path", "")
    structured = record.get("structured_response")
    freedom    = record.get("freedom_response")

    # Load image
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception:
        img = Image.new("RGB", (IMG_W, 400), (60, 60, 60))

    # Resize image to target width, preserve aspect ratio
    iw, ih = img.size
    new_h  = int(ih * IMG_W / iw)
    img    = img.resize((IMG_W, new_h), Image.Resampling.LANCZOS)

    # Create canvas: image + panel
    total_h = new_h + PANEL_H
    canvas  = Image.new("RGB", (IMG_W, total_h), BG_COLOR)
    canvas.paste(img, (0, 0))

    draw = ImageDraw.Draw(canvas)

    # Header bar
    header_y = new_h + 4
    label = (f"{record['participant']}  {record['scenario_prefix']}  "
             f"wz{record['wz_id']}  ts={record['timestamp']:.1f}")
    draw.text((PADDING, header_y), label, font=font, fill=HEADER_COLOR)

    # Risk badge (top-right of header)
    risk = structured.get("risk_level", "") if structured else ""
    badge_col = RISK_COLORS.get(risk, (100, 100, 100))
    badge_text = f" {risk.upper()} "
    bw = draw.textlength(badge_text, font=font)
    draw.rectangle([IMG_W - bw - PADDING - 4, header_y - 2,
                    IMG_W - PADDING,          header_y + FONT_SIZE + 2], fill=badge_col)
    draw.text((IMG_W - bw - PADDING, header_y), badge_text, font=font, fill=(255, 255, 255))

    # Divider
    div_y = header_y + FONT_SIZE + 6
    draw.line([(PADDING, div_y), (IMG_W - PADDING, div_y)], fill=(80, 80, 80), width=1)

    # Structured block
    struct_y = div_y + 6
    after_struct = draw_structured(draw, structured, PADDING, struct_y, font, font_small)

    # Divider
    draw.line([(PADDING, after_struct + 2), (IMG_W - PADDING, after_struct + 2)],
              fill=(60, 60, 60), width=1)

    # Freedom block
    draw_freedom(draw, freedom, PADDING, after_struct + 6, IMG_W - 2 * PADDING, font_small)

    return canvas


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",      required=True, help="Path to workzone_gpt_rt_XXXX.json")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--quality",    type=int, default=88)
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])
    print(f"Loaded {len(results)} records from {args.input}")

    # Auto-subfolder: human_validation/{model}/{run_ts}
    model  = data.get("model", "unknown").replace("/", "-")
    run_ts = data.get("run_ts", "unknown")
    out_dir = os.path.join(args.output_dir, model, run_ts)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output dir: {out_dir}")

    font       = load_font(FONT_SIZE)
    font_small = load_font(SMALL_FONT)

    for i, record in enumerate(results):
        canvas = render_frame(record, font, font_small)

        fname = (f"{record['participant']}_{record['scenario_prefix']}"
                 f"_wz{record['wz_id']}_{record['timestamp']:.3f}.jpg")
        out_path = os.path.join(out_dir, fname)
        canvas.save(out_path, quality=args.quality)
        print(f"  [{i+1}/{len(results)}] → {fname}")

    print(f"\nDone. {len(results)} images saved to {out_dir}")


if __name__ == "__main__":
    main()

# 对我们的图片进行LLM标注，得到文本输出，保存到annotation_outputs下的一个json文件中，供日后处理


import argparse
import io
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import torch
from datasets import load_from_disk
from PIL import Image
from transformers import pipeline

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from map_model_path import mapModelPath
from Prompt import Prompt as LLMPrompt

THIS = os.path.dirname(__file__)
DEFAULT_DATASET_DIR = os.path.join(
    PROJECT_ROOT,
    "WARM_UP_TASK",
    "vlm",
    "dataset",
    "front_camera_hf",
)
DEFAULT_OUTPUT = os.path.join(THIS, "annotation_outputs", "llm_annotation_results.json") # 输出文件路径
NUM_SAMPLES = 1 # 样本处理数量，为了快速测试，我们设为1，日后可以修改
NUM_WORKERS = 8 # 并行读入数据，越高越快，但不能太高


def parse_args():
    parser = argparse.ArgumentParser(description="Run LLM annotator on HF driving dataset.")
    parser.add_argument("--dataset-dir", type=str, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--split", type=str, default="validate", choices=["train", "validate"])
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_hf_split(dataset_dir: str, split: str):
    # 读取HF格式的数据集，返回指定split的数据集对象
    dataset_dict = load_from_disk(dataset_dir)
    if split not in dataset_dict:
        raise ValueError(f"Split '{split}' not found. Available splits: {list(dataset_dict.keys())}")
    return dataset_dict[split]


def _extract_image_source(row):
    # 提取图像信息
    image_info = row["image"]
    if isinstance(image_info, dict):
        img_path = image_info.get("path")
        if img_path:
            return img_path
        if image_info.get("bytes"):
            return image_info["bytes"]
        raise ValueError("Image dict does not contain valid path or bytes.")
    return image_info


def collect_samples(dataset, num_samples: int):
    # 从数据集中收集指定数量的样本，返回一个列表，每个元素包含index, scene_name, timestamp_str, image_source等字段
    limit = min(num_samples, len(dataset))
    rows = []
    for idx in range(limit):
        row = dataset[idx]
        rows.append(
            {
                "index": idx,
                "scene_name": row.get("scene_name", ""),
                "timestamp_str": row.get("timestamp_str", ""),
                "image_source": _extract_image_source(row),
            }
        )
    return rows


def _load_and_preprocess_one(sample, image_size=None):
    # 预处理图像
    image_source = sample["image_source"]
    if isinstance(image_source, Image.Image):
        img = image_source.convert("RGB")
        image_path = getattr(image_source, "filename", "")
    elif isinstance(image_source, (bytes, bytearray)):
        img = Image.open(io.BytesIO(image_source)).convert("RGB")
        image_path = ""
    else:
        img = Image.open(image_source).convert("RGB")
        image_path = str(image_source)

    if image_size is not None:
        img = img.resize((image_size, image_size), Image.Resampling.BILINEAR)
    out = dict(sample)
    out["image_path"] = image_path
    out["image"] = img
    return out


def preprocess_images(samples, num_workers: int, image_size=None):
    # 构建信息图像预处理的线程池，使用线程池来避免在进程间传递PIL图像时的重度IPC序列化问题。
    # Use threads to avoid heavy IPC serialization of PIL images between processes.
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        return list(executor.map(lambda s: _load_and_preprocess_one(s, image_size), samples))


def format_messages(prompt: LLMPrompt, image: Image.Image):
    # 根据prompt和图像构建LLM输入的消息列表，返回一个包含system message和user message的列表
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": prompt.system_message}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt.user_message},
            ],
        },
    ]


def extract_text_from_output(output):
    # 解析返回的输出，提取文本内容，返回一个字符串
    generated = output[0].get("generated_text", "")
    if isinstance(generated, str):
        return generated
    if isinstance(generated, list) and generated:
        last_item = generated[-1]
        if isinstance(last_item, dict):
            return str(last_item.get("content", ""))
        return str(last_item)
    return str(generated)


def run_mode(pipe, mode_name: str, prompt: LLMPrompt, preprocessed_samples, max_new_tokens: int):
    # 主要的推理函数，接受模型管道、模式名称、prompt对象、预处理后的样本列表和生成文本的最大长度，返回一个包含每个样本的index、mode和response的列表，以及推理耗时
    mode_outputs = []
    start = time.time()
    for sample in preprocessed_samples:
        messages = format_messages(prompt, sample["image"])
        output = pipe(text=messages, max_new_tokens=max_new_tokens)
        mode_outputs.append(
            {
                "index": sample["index"],
                "mode": mode_name,
                "response": extract_text_from_output(output),
            }
        )
    elapsed = time.time() - start
    return mode_outputs, elapsed


def merge_outputs(samples, freedom_outputs, structured_outputs):
    # 合并两种提示模式下的输出，为保存为一个文件做准备，返回一个包含每个样本的index、scene_name、timestamp_str、image_path、freedom_response和structured_response的列表
    freedom_map = {item["index"]: item["response"] for item in freedom_outputs}
    structured_map = {item["index"]: item["response"] for item in structured_outputs}

    merged = []
    for sample in samples:
        merged.append(
            {
                "index": sample["index"],
                "scene_name": sample["scene_name"],
                "timestamp_str": sample["timestamp_str"],
                "image_path": sample["image_path"],
                "freedom_response": freedom_map.get(sample["index"], ""),
                "structured_response": structured_map.get(sample["index"], ""),
            }
        )
    return merged


def main():
    # 先明白调控参数
    args = parse_args()

    print(f"Loading dataset from: {args.dataset_dir}")
    dataset = load_hf_split(args.dataset_dir, args.split)
    samples = collect_samples(dataset, args.num_samples)
    print(f"Collected {len(samples)} sample(s) from split '{args.split}'.")

    preprocess_start = time.time()
    preprocessed_samples = preprocess_images(
        samples=samples,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )
    preprocess_elapsed = time.time() - preprocess_start
    print(f"Preprocessed {len(preprocessed_samples)} image(s) in {preprocess_elapsed:.3f}s.")

    model_dir = mapModelPath("gemma-3-4b-it")
    print(f"Loading model from {model_dir}...")
    # 用pipeline完成推理，指定模型路径、设备、数据类型等参数
    pipe = pipeline(
        "image-text-to-text",
        model=model_dir,
        device="cuda",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    # 构建两种提示模式的Prompt对象，分别为自由式提示和结构化提示
    freedom_prompt = LLMPrompt(seed="FREEDOM")
    structured_prompt = LLMPrompt(seed="STRUCTURED")
    # 分别运行两种提示模式的推理，得到每个样本在两种模式下的输出和推理耗时
    freedom_outputs, freedom_elapsed = run_mode(
        pipe=pipe,
        mode_name="freedom",
        prompt=freedom_prompt,
        preprocessed_samples=preprocessed_samples,
        max_new_tokens=args.max_new_tokens,
    )
    structured_outputs, structured_elapsed = run_mode(
        pipe=pipe,
        mode_name="structured",
        prompt=structured_prompt,
        preprocessed_samples=preprocessed_samples,
        max_new_tokens=args.max_new_tokens,
    )
    # 合成与保存
    merged = merge_outputs(preprocessed_samples, freedom_outputs, structured_outputs)
    results = {
        "config": {
            "dataset_dir": args.dataset_dir,
            "split": args.split,
            "num_samples": args.num_samples,
            "num_workers": args.num_workers,
            "max_new_tokens": args.max_new_tokens,
            "image_size": args.image_size,
        },
        "timing": {
            "preprocess_seconds": preprocess_elapsed,
            "freedom_seconds": freedom_elapsed,
            "structured_seconds": structured_elapsed,
        },
        "results": merged,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved results to: {args.output}")


if __name__ == "__main__":
    main()
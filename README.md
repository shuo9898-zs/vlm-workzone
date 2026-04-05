# VLM Workzone Annotation Pipeline

## English

### Overview
This project is a **Vision-Language Model (VLM) annotation pipeline** for autonomous driving scenes, with a focus on work zone detection and driver attention analysis. It uses locally-deployed models (e.g., **Qwen2.5-VL**) to automatically annotate front-camera driving images with structured or free-form text descriptions.

### File Structure
| File | Description |
|------|-------------|
| `Prompt.py` | Defines the `Prompt` class with **freedom** (open-ended) and **structured** (JSON output) prompt templates, inspired by the VLM-AD paper. |
| `LLMAnnotation.py` | Basic single-frame annotation pipeline. Loads a HuggingFace dataset and annotates each frame. |
| `LLMAnnotation_temporal.py` | Temporal annotation pipeline. Annotates frames using a sliding window of context frames (default: 10 frames), optionally fusing vehicle telemetry (speed, steering, brake, etc.). |
| `LLMAnnotation_workzone.py` | Work-zone-aware annotation pipeline. Uses an Excel file (`workzone driving data.xlsx`) to identify work zone event windows per participant/scenario, then annotates only the relevant frames. |
| `annotation_outputs/` | Directory where all annotation results (`.json`, `.jsonl`) are saved. |

### Key Features
- **Two prompt modes**: `freedom` (descriptive narrative) and `structured` (strict JSON with fixed fields)
- **Temporal context**: sliding-window multi-frame input for richer scene understanding
- **Work zone event filtering**: timestamps from Excel are used to focus annotation on relevant driving segments
- **Parallel image loading**: `ThreadPoolExecutor` for fast preprocessing
- **Checkpoint support**: `.jsonl` checkpoints allow resuming interrupted runs

### Requirements
- Python 3.9+
- `transformers`, `torch`, `Pillow`, `datasets`, `openpyxl`
- A local VLM model (e.g., Qwen2.5-VL-3B-Instruct)

### Usage
```bash
# Basic single-frame annotation
python LLMAnnotation.py --split validate --num-samples 100

# Temporal annotation
python LLMAnnotation_temporal.py --participants P11 --scenarios S3

# Work zone focused annotation
python LLMAnnotation_workzone.py --participants P11 --scenarios S3
```

---

## 中文说明

### 概述
本项目是一个面向**自动驾驶场景**的**视觉语言模型（VLM）标注流水线**，专注于施工区（Work Zone）检测与驾驶员注意力分析。使用本地部署的模型（如 **Qwen2.5-VL**），对车载前置摄像头图像自动生成结构化或自由格式的文本标注。

### 文件说明
| 文件 | 描述 |
|------|------|
| `Prompt.py` | 定义 `Prompt` 类，包含**自由格式**（开放描述）和**结构化**（JSON 输出）两种 Prompt 模板，设计参考 VLM-AD 论文。 |
| `LLMAnnotation.py` | 基础单帧标注脚本。读取 HuggingFace 格式数据集，逐帧进行 LLM 标注。 |
| `LLMAnnotation_temporal.py` | 时序标注脚本。以滑动窗口方式（默认 10 帧）输入多帧图像，并可融合车辆遥测数据（速度、转向、刹车等）。 |
| `LLMAnnotation_workzone.py` | 施工区感知标注脚本。从 Excel 文件（`workzone driving data.xlsx`）中读取施工区事件时间窗口，仅对相关片段进行标注。 |
| `annotation_outputs/` | 所有标注结果（`.json`、`.jsonl`）的输出目录。 |

### 主要特性
- **两种 Prompt 模式**：`freedom`（自由叙述）和 `structured`（固定字段 JSON）
- **时序上下文**：多帧滑动窗口输入，提供更丰富的场景理解
- **施工区事件过滤**：利用 Excel 时间戳，聚焦于相关驾驶片段
- **并行图像加载**：使用 `ThreadPoolExecutor` 加速预处理
- **断点续跑**：`.jsonl` 格式检查点支持中断后恢复

### 环境依赖
- Python 3.9+
- `transformers`、`torch`、`Pillow`、`datasets`、`openpyxl`
- 本地 VLM 模型（如 Qwen2.5-VL-3B-Instruct）

### 使用方法
```bash
# 基础单帧标注
python LLMAnnotation.py --split validate --num-samples 100

# 时序标注
python LLMAnnotation_temporal.py --participants P11 --scenarios S3

# 施工区专项标注
python LLMAnnotation_workzone.py --participants P11 --scenarios S3
```

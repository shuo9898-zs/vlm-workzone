# LLM Workzone Annotation Pipeline

> GPT-based large-scale annotation of workzone driving frames with gaze overlay.  
> 基于 GPT 的工作区驾驶场景帧自动标注流程（含视线叠加图像）。

---

## File Structure / 文件结构

```
LLMAnnotation/
├── gpt_wz_api.py            # Main pipeline / 主流程脚本
├── visualize_annotations.py # Human validation visualization / 人工验证可视化
├── Prompt.py                # Prompt templates / 提示词模板
├── LLMAnnotation.py         # Local model pipeline (Gemma) / 本地模型流程
├── LLMAnnotation_workzone.py# Local model workzone pipeline / 本地工作区流程
└── annotation_outputs/
    ├── workzone_gpt/        # Raw GPT JSON outputs / GPT 原始输出
    └── human_validation/    # Visualization images / 可视化图片
        ├── gpt-4o/
        ├── gpt-4o-mini/
        └── gpt-5.4-mini/
```

---

## 1. GPT API Setup / 配置 GPT API

### 1.1 Create API Key / 创建 API Key

1. Go to [platform.openai.com](https://platform.openai.com) → **API Keys** → **Create new secret key**  
   前往 [platform.openai.com](https://platform.openai.com) → **API Keys** → **Create new secret key**

2. Add billing credit at Dashboard → **Billing**  
   在 Dashboard → **Billing** 充值

3. Set the key as an environment variable (**never hardcode it**):  
   将 Key 设为环境变量（**不要写入代码**）：

```cmd
# Windows CMD
set OPENAI_API_KEY=sk-proj-xxxx...

# Windows PowerShell
$env:OPENAI_API_KEY = "sk-proj-xxxx..."
```

### 1.2 Install Dependencies / 安装依赖

```cmd
pip install openai pillow openpyxl
```

### 1.3 Recommended Models / 推荐模型

| Model | Quality | Input $/1M | Output $/1M | Recommended For |
|-------|---------|-----------|------------|----------------|
| `gpt-5.4` | ★★★★★ | $2.50 | $15.00 | Highest quality full run |
| `gpt-5.4-mini` | ★★★★☆ | $0.75 | $4.50 | Best cost/quality ratio |
| `gpt-5.4-nano` | ★★★☆☆ | $0.20 | $1.25 | Large-scale budget run |
| `gpt-4o` | ★★★★☆ | $2.50 | $10.00 | Legacy high quality |
| `gpt-4o-mini` | ★★★☆☆ | $0.15 | $0.60 | Legacy budget |

---

## 2. Running the Pipeline / 运行流程

The pipeline has two modes: **real-time API** (fast) and **Batch API** (50% cheaper, slower).  
流程支持两种模式：**实时 API**（快）和 **Batch API**（5折优惠，较慢）。

### Mode A: Real-time / 实时模式（推荐测试用）

```cmd
# Single participant + scenario (test)
# 单人单 scenario 测试
python gpt_wz_api.py run --participants P10 --scenarios S4 --model gpt-5.4-mini

# Full dataset
# 全量数据
python gpt_wz_api.py run --model gpt-5.4-mini
```

### Mode B: Batch API / 批量模式（推荐全量用）

```cmd
# Step 1: Build request file locally (no API cost)
# 第一步：本地生成请求文件（不花钱）
python gpt_wz_api.py build --participants P10 --scenarios S4 --model gpt-5.4-mini

# Step 2: Upload and submit batch job
# 第二步：上传并提交任务
python gpt_wz_api.py submit --batch-input annotation_outputs\workzone_gpt\batch_requests_XXXX.jsonl

# Step 3: Poll and download results (auto-polls every 60s)
# 第三步：轮询下载结果（每 60s 自动查询）
python gpt_wz_api.py collect --batch-id batch_XXXX --participants P10 --scenarios S4
```

### Cost Estimate / 费用估算（34 participants × 6 scenarios × 2 laps）

| Model | Real-time | Batch (50% off) |
|-------|-----------|----------------|
| `gpt-5.4` | ~$237 | ~$119 |
| `gpt-5.4-mini` | ~$40 | ~$20 |
| `gpt-4o-mini` | ~$10 | ~$5 |

---

## 3. Prompt Design / 提示词设计

Two annotation modes are used, defined in `Prompt.py`:  
使用两种标注模式，定义在 `Prompt.py` 中：

### FREEDOM mode / 自由描述模式
Open-ended scene description covering workzone elements, traffic, gaze, risk, and driver attention.  
对工作区元素、交通状况、视线、风险和驾驶员注意力进行开放式描述。

### STRUCTURED mode / 结构化模式
Forces a JSON output with fixed fields for downstream processing:  
强制输出固定字段的 JSON，便于下游处理：

```json
{
  "workzone_present": "yes | no",
  "workzone_type": "none | lane closure | worker activity | merging zone | mixed | unclear",
  "traffic_condition": "free flow | following vehicle | dense traffic | intersection | traffic light | unclear",
  "primary_hazard": "none | worker | cone | vehicle | traffic light | mixed | unclear",
  "gaze_target": "road center | worker | cone | vehicle | traffic light | workzone area | uncertain | irrelevant",
  "attention_alignment": "good | partial | poor",
  "risk_level": "low | medium | high",
  "recommended_action": "continue | slow down | prepare to stop | stop | prepare lane change",
  "reasoning": "<one sentence, max 20 words>"
}
```

---

## 4. Human Validation / 人工验证

After annotation, generate visual validation images to verify GPT output quality:  
标注完成后，生成可视化图片验证 GPT 输出质量：

```cmd
python visualize_annotations.py --input annotation_outputs\workzone_gpt\workzone_gpt_rt_XXXX.json
```

**Output location / 输出位置：**
```
annotation_outputs/human_validation/{model}/{run_ts}/
  P10_S4_wz1_1765554501.621.jpg
  P10_S4_wz1_1765554502.121.jpg
  ...
```

**Each image contains / 每张图包含：**
- Original dashboard camera frame with gaze overlay / 原始仪表盘图像（含视线叠加）
- Risk level badge (green/yellow/red) / 风险等级徽标（绿/黄/红）
- All structured fields in a two-column layout / 结构化字段双列显示
- Freedom response (first 3 lines) / 自由描述（前3行）

Multiple models can be compared side-by-side since each run saves to its own subfolder.  
不同模型的结果保存在独立子目录，方便并排对比。

---

## 5. Output Format / 输出格式

Results are saved as JSON:  
结果保存为 JSON 文件：

```json
{
  "run_ts": "20260410_135514",
  "model": "gpt-5.4-mini",
  "total": 60,
  "results": [
    {
      "participant": "P10",
      "scenario_prefix": "S4",
      "wz_id": 1,
      "t_start": 1765554503.62,
      "t_end": 1765554508.621,
      "timestamp": 1765554501.621,
      "image_path": "...",
      "freedom_response": "...",
      "structured_response": { ... }
    }
  ]
}
```

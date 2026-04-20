# New 4090 Machine Setup Guide

## 1. 创建 Conda 环境

把整个项目文件夹复制/同步到新机器后：

```powershell
# 进入项目目录
cd D:\VLM\BC\vanilla_bc

# 从 environment.yml 一键创建环境
conda env create -f environment.yml

# 激活
conda activate bc_carla
```

如果 `environment.yml` 因网络/版本冲突失败，用 fallback 方式：

```powershell
conda create -n bc_carla python=3.8 -y
conda activate bc_carla

# 安装 PyTorch（CUDA 11.8，适配 4090）
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118

# 安装其余依赖
pip install -r requirements.txt
```

---

## 2. 最大化榨取 4090

### ① 验证 GPU 可见

```powershell
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### ② config_bc.py 调参

| 参数 | 现值 | 4090 建议值 | 说明 |
|---|---|---|---|
| `BATCH_SIZE` | 64 | **256–512** | 4090 有 24 GB，尽量堆大 |
| `NUM_WORKERS` | 8 | **12–16** | 匹配 CPU 核数 |
| `DEVICE` | `"cuda"` | 保持不变 | ✓ |

### ③ 开启混合精度（AMP）

在 `train_bc.py` 训练循环里替换 forward/backward：

```python
scaler = torch.cuda.amp.GradScaler()

# forward
with torch.cuda.amp.autocast():
    outputs = model(images, speeds)
    loss = compute_bc_loss(outputs, labels)

# backward
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

> 通常提升吞吐量 **1.5–2×**，显存占用减半。

### ④ 固定显存碎片（设置环境变量）

```powershell
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### ⑤ 监控 GPU 利用率

```powershell
nvidia-smi dmon -s u   # 实时看 GPU util / 显存
```

> 目标：GPU-Util 持续 **>90%**。若偏低说明 DataLoader 是瓶颈，增大 `NUM_WORKERS`。

---

## 3. 同步项目文件夹

### 方法 A — robocopy（本地/局域网，推荐）

```powershell
robocopy D:\VLM\BC\vanilla_bc \\新机器IP\D$\VLM\BC\vanilla_bc /MIR /XD checkpoints logs results __pycache__
```

- `/MIR` 镜像同步（新增/修改/删除均同步）
- `/XD` 排除 checkpoints/logs/results 等大文件夹（按需调整）

### 方法 B — xcopy 只同步代码

```powershell
xcopy D:\VLM\BC\vanilla_bc \\新机器IP\D$\VLM\BC\vanilla_bc /S /Y /EXCLUDE:exclude.txt
```

### 方法 C — Git（推荐长期版本管理）

```powershell
# 当前机器
git init
git add -A
git commit -m "sync"
git remote add origin <你的repo>
git push

# 新机器
git clone <你的repo> D:\VLM\BC\vanilla_bc
```

> 在 `.gitignore` 中加入以下内容避免推大文件：
> ```
> checkpoints/
> logs/
> results/
> __pycache__/
> ```

---

## 4. 更新 config_bc.py 路径

新机器上路径若不同，只需修改 `config_bc.py` 开头的路径变量：

```python
TRAIN_ANNOTATION_PATH = r"D:\VLM\BC\vanilla_bc\annotations\train.json"
VAL_ANNOTATION_PATH   = r"D:\VLM\BC\vanilla_bc\annotations\val.json"
CHECKPOINT_DIR        = r"D:\VLM\BC\vanilla_bc\checkpoints"
LOG_DIR               = r"D:\VLM\BC\vanilla_bc\logs"
RESULT_DIR            = r"D:\VLM\BC\vanilla_bc\results"
```

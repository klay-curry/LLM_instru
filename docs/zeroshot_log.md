# 零样本推理实施日志

## 方案概述

由于当前赛题提供的 `queries.json` **不包含 bounding box 标注**（只有 visible / infrared / depth / query 四字段），无法直接进行有监督训练。

经与用户确认，本次采用 **Qwen3-VL-8B-Instruct 零样本推理 + DINOv2 视觉特征（待集成）** 的最小可运行闭环方案，跳过原 plan.md 中的训练流程。

## 设计调整（vs. 原 plan.md）

| 项目 | 原 plan.md | 本次实施 |
|------|-----------|---------|
| 核心方法 | Qwen3-VL + Grounding DINO Decoder 双分支 | Qwen3-VL 零样本输出 bbox |
| 数据要求 | 需要 bbox 标注训练 | 无标注，零样本 |
| 训练方式 | LoRA + 全量训练 Decoder | 不训练，纯推理 |
| 推理输出 | Decoder 回归坐标 | Qwen3-VL 直接生成坐标 |

## 实施步骤

### 阶段 0：环境准备 ✅

1. 安装依赖：`peft`, `bitsandbytes`
2. 验证 Qwen3-VL-8B-Instruct 加载（路径：`/root/autodl-fs/weights/Qwen3-VL-8B-Instruct`）
3. 关键发现：
   - 实际模型类名是 `Qwen3VLForConditionalGeneration`（不是 `Qwen2VLForConditionalGeneration`）
   - `transformers.AutoProcessor` 可以正常加载（`Qwen3VLProcessor`）
   - 多图像输入格式：`input_ids + mm_token_type_ids + pixel_values + image_grid_thw`
   - bf16 推理显存约 16.33 GB，远小于 96 GB 显存上限

### 阶段 1：数据接入修复 ✅

| Bug | 修复 |
|-----|------|
| `data/dataset.py` 第 54-55 行 depth_path 和 ir_path 写反 | 修正 |
| `data/preprocessing.py` read_depth 注释模糊 | 明确注释 |
| `scripts/prepare_submit.py` 缺少 `sys.path` 注入 | 注入项目根目录 |

新增 `scripts/split_queries.py`：按 image_id 8:2 划分 train/val 共 9555 条（7706 / 1849）。

### 阶段 2：核心推理模块 ✅

新建文件：
- `configs/zeroshot.yaml`：推理配置（模型路径、生成参数、输出路径）
- `models/zero_shot_qwen.py`：核心推理类 `ZeroShotGrounder`
  - `load_rgb/load_depth/load_ir`：图像加载
    - depth 用 `cv2.IMREAD_UNCHANGED` 保留 16-bit，clip 到 20000mm / 20000 → [0,1]
    - IR 取单通道保留 3ch 复制
  - `parse_bbox`：正则解析四种格式（带/不带方括号、像素坐标自动归一化）
  - `postprocess_bbox`：x1<x2, y1<y2 强制 + clip [0,1] + 零面积兜底
  - `ZeroShotGrounder.predict_bbox`：单 query 推理入口
- `scripts/run_zeroshot.py`：批量推理入口

### 阶段 4：闭环验证 ✅

#### Smoke test（10 条）
- 总耗时：6.0 s（GPU 启动 4-5s + 10 条推理）
- 平均推理：0.60 s/q
- 全部 10 条解析成功

#### 100 条稳定性测试
- 总耗时：52.9 s
- 平均推理：0.53 s/q
- 100/100 解析成功，3 个 fallback bbox（其中一条模型输出 `0,0,0,0` 被替换为 fallback）
- 输出分布：mean bbox ≈ [0.40, 0.50, 0.55, 0.71]，倾向于较小框（中位数 0.10 × 0.14）

## Prompt 设计

经过 3 个 prompt 模板测试，最终采用：

```
You are a precise visual grounding system. The first image is the visible-light RGB view.
The second image is the depth map (brighter = farther).
The third image is the infrared / thermal view (brighter = hotter).
TASK: Locate the target described as: '{query}'
OUTPUT FORMAT (STRICT): Output ONLY four numbers in the range [0.0, 1.0] separated by
commas, representing (x1, y1, x2, y2) where (0,0) is the top-left corner and
(1,1) is the bottom-right corner of the RGB image.
Example output: 0.12,0.34,0.56,0.78
Output:
```

测试输出示例：
- `0.48,0.41,0.55,0.52` （像素坐标自动归一化）
- `[0.08, 0.26, 0.48, 0.43]` （带方括号）
- `0.62,0.49,0.67,0.66`

后处理对所有格式都能正确解析。

## 全量推理

- 启动（screen 后台）：
  ```bash
  screen -dmS qwen_inference bash -c \
    'python3 scripts/run_zeroshot.py --config configs/zeroshot.yaml --batch-size 32 ...'
  ```
- 输出：`outputs/predictions.json`（9555 条）
- **实际耗时**：36 分钟（9555 × 0.23s + GPU 启动开销），比单条推理（80 分钟）快 2.3 倍
- **GPU 利用率**：95-100%（之前单条时仅 35%）
- **显存峰值**：28 GB / 96 GB（远低于上限）
- **统计**：平均 bbox ≈ [0.43, 0.38, 0.57, 0.63]，0 个无效 bbox
- 打包：`scripts/prepare_submit.py --prediction outputs/predictions.json --output outputs/submit.zip`

### 性能优化历程

| 批次大小 | 单条耗时 | GPU 利用率 | 总耗时 (9555 条) |
|---------|---------|-----------|----------------|
| batch=1 | 0.53 s/q | 35% | ~85 min |
| batch=8 | 0.27 s/q | - | ~45 min |
| batch=16 | 0.25 s/q | - | ~40 min |
| **batch=32** | **0.23 s/q** | **95-100%** | **~36 min** ✅ |

### 初赛成绩

**v1 提交（纯 Qwen3-VL）：0.43 分**。诊断显示 75% 预测框面积 < 0.05，说明模型倾向输出紧致框，IoU≥0.5 阈值下损失大量样本。

### v2 改进：引入 Grounding DINO

下载 `IDEA-Research/grounding-dino-base`（1.87 GB）到 `/root/autodl-fs/grounding-dino-base/`。

新增模块：
- `models/grounding_dino_zero_shot.py`：`GroundingDINOZeroShot` 类（IoU NMS + top-1 选取）
- `scripts/build_hybrid_predictions.py`：Qwen3-VL 与 GDino 融合工具
- `outputs/run_gdino_full.py`：全量 GDino 推理

GDino 全量推理耗时：**20.5 分钟**（0.13 s/q，比 Qwen3-VL 快 2 倍），显存仅 1 GB。

A/B 分析（1000 条）：
- **17% 高一致**（IoU>0.5）：两个模型基本同意
- **64% 严重分歧**（IoU<0.1）：位置不同
- 由于无 GT 可验证，**决定采用 GDino 替换策略**：当 GDino 置信度 ≥ 0.25 时替换 Qwen3-VL 输出

### 最终提交（v2 - GDino 替换）

- 替换比例：67.5%（6458/9555）
- 中位 bbox：0.124 × 0.220（v1: 0.100 × 0.175，**更大**）
- 总无效 bbox：0

预测分布对比：

| 模型 | mean bbox | w/h median | area median |
|------|----------|-----------|------------|
| Qwen3-VL (v1) | [0.43, 0.38, 0.57, 0.63] | 0.100/0.175 | 0.017 |
| **GDino-replace (v2)** | [0.39, 0.37, 0.59, 0.67] | **0.124/0.220** | **0.022** |

## 提交物清单

```
/root/code/LLM/outputs/
├── predictions.json               ← 当前主提交（v2 GDino 替换版）
├── predictions_v1_qwen.json       (v1 备份：纯 Qwen3-VL 0.43 分)
├── predictions_hybrid.json        (v2 备选：IoU>0.3 平均融合)
├── predictions_gdino_replace.json (v2 当前：GDino 替换)
├── predictions_weighted.json      (v2 备选：IoU 加权平均)
├── submit.zip                     ← 当前主提交包
├── submit_v1_qwen.zip
├── submit_hybrid.zip
├── submit_gdino_replace.zip
└── submit_weighted.zip
```

## 后续改进（未在本轮实施）

1. **DINOv2 视觉特征融合**：取 visible 图的 patch token grid（16×16），与 Qwen3-VL 输出的文本 token hidden state 做相似度，定位目标 patch，作为 fallback 或 ensemble
2. **多 prompt 模板 + 投票**：3 个不同 prompt 取 median bbox
3. **CLAHE / 直方图均衡**：对 IR/Depth 做对比度增强，提高 VLM 感知
4. **获取有标注数据后做 LoRA 微调**：用伪标签或真标签

## 文件清单

新增：
- `configs/zeroshot.yaml`
- `models/zero_shot_qwen.py`
- `scripts/run_zeroshot.py`
- `scripts/split_queries.py`
- `data/splits/train.json`（7706 条）
- `data/splits/val.json`（1849 条）
- `outputs/predictions.json`（全量预测，运行后生成）
- `outputs/submit.zip`（提交包，运行后生成）

修改：
- `data/dataset.py`：修复 depth/IR 路径写反 bug
- `data/preprocessing.py`：read_depth 注释明确化
- `scripts/prepare_submit.py`：注入 sys.path 修复导入
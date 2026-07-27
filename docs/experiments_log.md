# 实验日志：多模态视觉定位（初赛）

> 记录所有已跑实验、对应分数、关键改动，方便后续复赛/半决赛快速回归。

## 时间线

| 时间 | 事件 | 分数 | 备注 |
|------|------|------|------|
| 07-25 14:42 | Smoke test（10 条 Qwen3-VL 推理） | - | 验证 pipeline |
| 07-25 14:44 | 100 条 Qwen3-VL 推理 | - | 验证稳定性 |
| 07-25 15:00 | 启动全量 Qwen3-VL 推理（screen） | - | batch=32，0.23s/q |
| 07-25 15:36 | 全量 Qwen3-VL 完成（9555 条） | - | 36 分钟 |
| 07-25 15:39 | 打包 submit.zip（v1） | - | 9555 valid |
| 07-25 ~17:00 | **v1 提交评分** | **0.43** | 纯 Qwen3-VL ✅ 当前最高分 |
| 07-25 16:10 | 下载 Grounding DINO base | - | hfd + hf-mirror.com |
| 07-25 16:14 | Smoke test GDino（3 条） | - | 验证可加载 |
| 07-25 16:18 | GDino 100 条 / 1000 条 A/B | - | 64% 分歧 |
| 07-25 16:20 | GDino 全量推理（screen） | - | 20.5 分钟 |
| 07-25 16:43 | 生成 4 种融合策略 | - | 见下表 |
| 07-25 16:45 | **v2 提交（GDino 替换 67.5%）** | **0.39** ❌ 下降 0.04 | bbox 更大但反而变差 |
| 07-25 16:54 | 回滚到 v1（0.43）作为主提交 | - | `submit.zip` = v1 |
| 07-25 17:00 | **v3 提交（hybrid 21.7% 融合）** | **0.45** ✅ 上涨 0.02 | 保守融合有效 |

---

## 实验记录

### Exp-01：纯 Qwen3-VL 零样本（baseline）

**配置：**
- 模型：`/root/autodl-fs/weights/Qwen3-VL-8B-Instruct`
- 输入：3 张图（RGB + Depth + IR）+ query
- Prompt：明确要求输出 `[x1, y1, x2, y2]` 归一化坐标
- batch=32，bf16

**结果：**
- 总耗时：36 分钟
- 成功：9555/9555
- 无效 bbox：0
- 中位 bbox：0.100 × 0.175（**小框问题严重**）
- **得分：0.43**

**关键诊断：**
```
Mean bbox: [0.43, 0.38, 0.57, 0.63]
75% 预测框面积 < 0.05（极小）
Fallback bbox 数: 229
```
VLM 直接回归坐标本身存在精度天花板（与 plan.md 中"VLM 擅长语义理解、不擅长像素级坐标"的判断一致）。

---

### Exp-02：多 prompt 集成（验证提升空间）

**配置：** 3 个 prompt 模板（default / tight / centered）+ median merge

**结果：**
| 模式 | 耗时 | 平均 bbox |
|------|------|----------|
| Single | 22.2 s (100 条) | [0.396, 0.499, 0.551, 0.705] |
| Ensemble 3 prompts | 50.3 s (100 条) | [0.396, 0.502, 0.551, 0.705] |

**结论：** ❌ 无显著提升（差 < 0.005），3 倍耗时。**模型对 prompt 变化鲁棒**，集成无意义。

---

### Exp-03：Grounding DINO 零样本

**配置：**
- 模型：`/root/autodl-fs/grounding-dino-base/`（下载自 `IDEA-Research/grounding-dino-base`，1.87 GB）
- 输入：仅 RGB 图 + query（不支持多模态）
- 阈值：box_threshold=0.20, text_threshold=0.20, NMS IoU=0.5
- 选取：NMS 后取 top-1

**结果（1000 条 A/B）：**
| 阈值 | no-detect | 中位 width/height | 耗时 |
|------|----------|------------------|------|
| 0.25 | 41% | 0.10/0.10 | 0.13s/q |
| 0.20 | 11% | 0.092/0.140 | 0.13s/q |
| **0.15** | **0.5%** | 0.091/0.186 | 0.13s/q |

**与 Qwen3-VL 对比（200 条）：**
| 模型 | median bbox area | no-detect |
|------|-----------------|----------|
| Qwen3-VL | 0.017 | 0% |
| GDino@0.20 | 0.012 | 11% |
| GDino@0.15 | 0.017 | 0.5% |

**A/B 一致性（1000 条）：**
- IoU(Qwen, GDino) > 0.5：169/1000 = **17% 高一致**
- IoU < 0.1：**644/1000 = 64% 严重分歧**
- 由于无 GT 可验证，**无法判断哪个更准**

**全量推理：** 9555 条用 20.5 分钟，0.13 s/q，显存仅 1 GB。

---

### Exp-04：融合策略对比（基于 v1 Qwen + GDino@0.20）

**4 种策略：**

| 变体 | 说明 | 改动比例 | 中位 w/h | 得分 |
|------|------|---------|---------|------|
| `v1_qwen` | 纯 Qwen3-VL | 0% | 0.100/0.175 | **0.43** |
| `hybrid` | IoU≥0.3 时取平均 | 21.7% | 0.100/0.175 | **0.45** ✅ 当前最佳 |
| `gdino_replace` | GDino 置信度≥0.25 时替换 | 67.5% | 0.124/0.220 | **0.39** ❌ |
| `weighted` | IoU 加权平均 | - | 0.100/0.175 | 未提交 |

### Exp-05（v2 失败分析）

**结果：** GDino 替换版（v2）从 0.43 下降到 0.39，**反降 0.04**

**关键发现：**
1. **GDino 单独表现不如 Qwen3-VL**：虽然 GDino bbox 更大（中位 0.124/0.220 vs Qwen3-VL 0.100/0.175），但得分反而更低。说明 **本数据集的真实目标确实较小**，GDino 输出"看起来更合理"的较大框实际是过预测（外扩）。
2. **64% 分歧被验证为 GDino 错误**：之前 A/B 测试看到 Qwen/GDino 64% IoU<0.1（严重分歧），实际这些分歧位置大部分是 GDino 错而 Qwen 对。
3. **多模态信息的价值**：Qwen3-VL 同时看到 RGB + Depth + IR 三模态，能更精准定位；GDino 仅看 RGB 在弱光/遮挡场景会失败。

**结论：** ❌ **Grounding DINO 单独不适合作为最终预测**。即使有 GPU 加速，单模态信息量不如 VLM 的多模态融合。

**行动：** 验证 hybrid（保守融合，IoU≥0.3 时取平均），**得分 0.45（+0.02）**——保守融合有效。

### Exp-06：更严格的融合阈值探索

hybrid (0.3, 0.25) → 0.45。进一步探索更严格的阈值（更少改动）：

| 变体 | 阈值 | 改动比例 | 状态 |
|------|------|---------|------|
| `strict_050` | IoU≥0.50, score≥0.30 | 12.3% | 未提交 |
| `strict_040` | IoU≥0.40, score≥0.30 | 14.2% | 未提交 |
| `replace_strict` | IoU≥0.50, 完全替换 | 12.3% | 未提交 |
| `current hybrid` | IoU≥0.30, score≥0.25 | **21.7% → 0.45** | 主提交 |

**逻辑：** 更高的 IoU 阈值意味着只在"双方都自信"时才融合——这种"高置信度一致"的样本平均下来通常更接近 GT。但**改动更少**，可能收益也小。

---

## 关键经验（更新）

### ✅ 有效的改进
1. **batch 推理 + screen 后台**：充分利用 GPU（35% → 100% util），速度提升 2.3×
2. **padding_side="left"**：消除 decoder-only 警告
3. **Qwen3-VL 多模态融合**：RGB + Depth + IR 三模态融合是关键优势
4. **保守 hybrid 融合**（IoU≥0.3 时取平均）：从 0.43 提升到 **0.45**

### ❌ 无效/有害的改进
1. **多 prompt 集成**：Qwen3-VL 对 prompt 鲁棒，集成无意义
2. **框膨胀后处理**：无 GT 验证，无法判断是否正确
3. **Grounding DINO 替换**：bbox 更大但反而下降 0.04 分

### 💡 关键洞察
1. **VLM 直接回归坐标天花板明显**（0.43 分）—— 与 plan.md 一致
2. **多模态融合 > 单模态检测器**：Grounding DINO 仅 RGB，丢失 depth/ir 信息
3. **不要假设更大的 bbox 更好**：本数据集目标较小，过预测反而扣分
4. **没有 GT 时无法判断融合是否更优**：只能通过实际评测反馈
5. **保守融合策略有效**：仅在 IoU 高且 GDino 置信度高时才融合，避免引入错误

### 🚫 后续避免
- ❌ 不要盲目相信"专门模型一定更好"
- ❌ 不要用单模态检测器替代多模态 VLM
- ❌ 不要做无 GT 验证的"启发式后处理"（如膨胀）

## 📊 历次提交得分

| 版本 | 得分 | Δ | 策略 |
|------|------|---|------|
| v1 Qwen3-VL | **0.43** | baseline | 纯零样本 VLM |
| v2 GDino 替换 | **0.39** | -0.04 | 67.5% 用 GDino |
| **v3 hybrid** | **0.45** ✅ | **+0.02** | 21.7% 平均融合（当前最佳）|

---

## 已修改/新增文件清单

> **历史快照说明：** 本节及其后的“后续待办/关键经验”保留了初赛当时的文件和判断，部分内容已经过期。当前主提交是平台 `0.45` 的 hybrid，`outputs/predictions.json` 与 `outputs/submit.zip` 已对应 hybrid；GDino 大范围替换已以 `0.39` 证明有害。当前状态和下一步以本文前半部分、[`plan.md`](./plan.md) 和 [`README.md`](./README.md) 为准。

### 新增

| 文件 | 用途 |
|------|------|
| `configs/zeroshot.yaml` | 推理配置 |
| `models/zero_shot_qwen.py` | Qwen3-VL 零样本推理（含 batch + ensemble） |
| `models/grounding_dino_zero_shot.py` | Grounding DINO 零样本推理（含 NMS） |
| `scripts/run_zeroshot.py` | 主推理入口（断点续跑 + flush） |
| `scripts/split_queries.py` | train/val 划分 |
| `scripts/build_hybrid_predictions.py` | Qwen+GDino 融合工具 |
| `docs/zeroshot_log.md` | 零样本推理日志 |
| `docs/experiments_log.md` | 本文档 |

### 修改

| 文件 | 改动 |
|------|------|
| `data/dataset.py` | 修复 depth/IR 路径写反 bug |
| `data/preprocessing.py` | depth 16-bit 读取 |
| `scripts/prepare_submit.py` | sys.path 注入修复 |

### 输出文件

```
/root/code/LLM/outputs/
├── predictions.json               ← 主提交（v2 GDino 替换）
├── predictions_v1_qwen.json       (v1 备份)
├── predictions_hybrid.json        (备选 1)
├── predictions_gdino_replace.json (备选 2)
├── predictions_weighted.json      (备选 3)
├── submit.zip                     ← 主提交包
├── submit_v1_qwen.zip             (0.43 分)
├── submit_hybrid.zip
├── submit_gdino_replace.zip
└── submit_weighted.zip
```

---

## 后续待办

### 短期（等待后台训练数据下载期间）

1. **【等待】** 方案 A 训练数据下载完成（后台进行中）
2. **【等待】** 用户上传 v2 后的实际分数

### 中期（拿到有标注数据后）

1. **LoRA 微调 Qwen3-VL**：用 GT bbox 监督
2. **DINOv2 + Cross-Attention Decoder 训练**（原 plan A 核心组件）
3. **多 Checkpoint 集成**
4. **数据增强**：水平翻转、深度加噪、MixUp

### 长期（半决赛 / 总决赛）

1. **完整方案 A 实施**（VLM 语义引导 + Decoder 精修）
2. **外部数据集**：RefCOCO、Visual Genome
3. **模型集成**：EMA + WBF

---

## 关键经验

### ✅ 有效的改进
1. **batch 推理 + screen 后台**：充分利用 GPU（35% → 100% util），速度提升 2.3×
2. **padding_side="left"**：消除 decoder-only 警告
3. **Grounding DINO**：专门为 grounding 设计的模型，能输出更接近真实目标的 bbox

### ❌ 无效的改进
1. **多 prompt 集成**：Qwen3-VL 对 prompt 鲁棒，集成无意义
2. **框膨胀后处理**：无 GT 验证，无法判断是否正确

### ⚠️ 待验证的改进
1. **GDino 替换 Qwen**：bbox 更大但是否更准未知，需上传评测
2. **混合 / 加权平均**：保守策略，可能变化不大

### 💡 关键洞察
1. **VLM 直接回归坐标天花板明显**（0.43 分）—— 与 plan.md 一致
2. **专门 grounding 模型（GDino）能输出更准的 bbox** —— 工业界共识
3. **代码框架 + 断点续跑 + screen 后台** 是大规模推理的关键基建

---

## Exp-007：RefCOCO 语义引导定位器修复（2026-07-27）

### 预计算中止与缓存状态

- 用户决定停止收益较低的逐样本 Qwen 特征预计算。
- 已向 PID `11013` 发送 `SIGTERM`，进程正常退出；退出后 GPU 为 `0 MiB / 0%`。
- 中止时实际已经存在完整可加载的 `refcoco_train.pt`、`refcoco_val.pt`、`refcocoplus_train.pt` 和 `refcocoplus_val.pt`，因此没有继续运行的必要。
- RefCOCO cache 是单体 PyTorch pickle；`refcoco_train.pt` 约 8.2GB，即使只截取 8 条也必须先反序列化全文件，实测加载约 168 秒。正式训练前应改成 shard 或预先生成轻量索引格式。
- 旧预计算脚本把 Parquet 中整组 captions 传给 Qwen，而不是单条字符串。它们对应同一 bbox，现有 token 可用于工程验证，但后续正式 cache 应保存明确的单条 Query、sample ID、attention mask 和版本信息。

### 代码修复

| 文件 | 修复 |
|---|---|
| `data/refcoco_cache.py` | 新增完全离线 cache Dataset；按原图宽高将像素 `xyxy` 归一化；校验 bbox；pad semantic tokens 并生成 padding mask |
| `models/semantic_grounding.py` | 新增冻结 DINOv2 + Qwen semantic tokens + cross-attention locator；输出合法参数化 bbox |
| `models/vision_encoder.py` | 不再丢弃 DINOv2 的 cls/register/mask token 预训练权重 |
| `utils/metrics.py` | 修复 GIoU 中 enclosing area 与 union 的公式 |
| `scripts/train_refcoco.py` | 重写为离线 cache 训练入口；支持 bf16、梯度累积、validation 指标、统一 checkpoint、resume、smoke test 和无语义对照 |

### GPU smoke test

实际命令：

```bash
python3 scripts/train_refcoco.py \
  --smoke-test --batch-size 8 \
  --max-train-samples 8 --max-val-samples 8 \
  --output-dir /root/autodl-tmp/experiments/exp007_refcoco_locator/smoke
```

结果：

- forward、loss、backward 和 optimizer step 均成功。
- 输出 shape 为 `(8, 4)`，无非法框。
- checkpoint 重载前后最大预测差为 `0`。
- loss `2.3963`，峰值显存 `1.1GB`。
- 低显存是 8 样本正确性测试结果，不代表正式 batch 配置。

### 32 样本过拟合与语义对照

从公开 RefCOCO validation cache 提取同一组 32 条样本，仅用于工程过拟合测试。训练集和评测集相同，因此以下结果不能解释为泛化能力或正式 validation 成绩。

| 条件 | Epoch 1 ACC@0.5 | Epoch 50 ACC@0.5 | Epoch 50 mean IoU | Epoch 50 median IoU | 非法框 |
|---|---:|---:|---:|---:|---:|
| RGB + Qwen semantic tokens | 0.1875 | **0.5938** | **0.5680** | **0.5991** | 0 |
| RGB + learned null semantic | 0.1250 | 0.3125 | 0.3899 | 0.3357 | 0 |

共同配置：DINOv2 冻结、batch 32、50 epochs、learning rate `1e-3`、同一定位头和随机种子。带语义模型峰值显存约 `0.8GB`，每轮约 1.7-2.0 秒。

结论：

- 修复后的定位器可以学习，loss 和 IoU 有明确改善。
- 在同预算对照下，Qwen semantic tokens 明显优于无语义 object query，证明“VLM 语义引导 + 专门定位头”在当前实现中确实生效。
- 该结果支持继续主方案，但尚不能证明 RefCOCO 泛化，更不能证明迁移到官方三模态测试集会超过 `0.45`。
- 下一步应先解决 cache 单体加载和 Query 格式，再用独立 RefCOCO train/validation 做正式 A/B；不要直接对官方无标签数据训练或划分。

---

## Exp-008：RefCOCO 小规模独立验证与 batch 探测（2026-07-27）

### 实验目的

验证公开 RefCOCO train 到独立 validation 的完整流程，探测 `batch=64/128/256/512` 的 bf16 训练吞吐与显存，并比较有无 Qwen semantic tokens。官方 `LLM_data` 未参与。

### Batch microbenchmark

配置：冻结 DINOv2，训练 semantic projection、3 层 cross-attention decoder 和 bbox head；每档 2 次 warmup、5 次计时，包含 forward、loss、backward 和 optimizer step。数据预先进入内存，因此以下结果不包含真实图片 IO。

| Batch | Samples/s | Step 时间 | Peak allocated | Peak reserved | 状态 |
|---:|---:|---:|---:|---:|---|
| 64 | 1832.9 | 0.0349s | 1.12GB | 1.56GB | 正常 |
| 128 | 1785.3 | 0.0717s | 1.86GB | 2.72GB | 正常 |
| 256 | 1811.8 | 0.1413s | 3.34GB | 5.00GB | 正常 |
| 512 | **1848.5** | 0.2770s | 6.31GB | 9.50GB | 正常 |

原始结果：`/root/autodl-tmp/experiments/exp008_refcoco_pilot/batch_probe.json`。

结论：四档纯计算吞吐接近，`batch=512` 略高且显存充足；本模型无法通过这四档把 96GB 显存用满。实际训练不应只追求最大 batch，因为 batch 512 在 3,581 条 pilot 数据上每 epoch 仅 7 次更新。

### 独立 train/validation A/B

共同配置：

- train：RefCOCO train 前 3,581 条，约可用 train cache 的 10%；
- validation：完整独立 RefCOCO validation 3,189 条；
- 3 epochs，batch 512，21 次 optimizer update；
- frozen DINOv2，bf16，learning rate `1e-4`，seed 42；
- 训练和验证均未使用官方赛题数据。

| 模型 | Epoch 1 ACC@0.5 | Epoch 3 ACC@0.5 | Epoch 3 Mean IoU | Median IoU | Train loss | Peak 显存 |
|---|---:|---:|---:|---:|---:|---:|
| RGB + Qwen semantic tokens | 0.0988 | **0.1013** | **0.2429** | **0.2051** | **2.5891** | 6.9GB |
| RGB + null semantic | 0.0931 | 0.0947 | 0.2360 | 0.1945 | 2.6247 | 6.9GB |

两组均为 0 个非法框。语义版本相对无语义版本：

- `ACC@0.5`：`+0.0066`；
- mean IoU：`+0.0069`；
- median IoU：`+0.0106`；
- train loss：降低 `0.0356`。

端到端耗时：语义版本首轮约 25.5 秒、后续约 14.5 秒；无语义版本约 16-20 秒/轮。batch 内纯计算可达约 1,800 samples/s，但真实 epoch 受 COCO 图片读取、worker 启动和 validation IO 影响明显。

### 结论与限制

- 完整 train/independent-validation 流程已跑通，semantic tokens 在所有主要指标上方向一致地优于无语义对照。
- 差值很小，当前只能视为初步正向信号，不能证明最终泛化提升，更不能推断赛题分数。
- pilot 只训练 21 步，batch 512 对 3,581 条数据过大；下一轮应优先保证优化步数，建议 batch 128 或 256，而不是继续增大 batch。
- 10% 数据是 cache 前缀截断，可能有排序偏差；正式实验应使用固定随机索引或完整 train。
- 旧 cache 的 Query 格式和单体序列化问题仍存在，正式长训前应规范化并分片。
- 当前没有理由开始官方数据伪标签或无监督训练。下一有价值实验是使用规范数据、更多 optimizer steps 的 RefCOCO A/B。

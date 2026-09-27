# 多模态视觉定位 · Multimodal Visual Grounding

> RGB + Depth + IR + English Query → RGB 坐标系归一化 bbox，指标 **ACC@0.5**。

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-research--only-lightgrey)]()
[![Status](https://img.shields.io/badge/status-reproduced-success)]()

基于强专门基座 + 推理侧置信度的多模态视觉定位项目；从通用 VLM 零样本 **0.43** 起步，最终达到复赛平台分 **0.708**（提升 65%）。所有数字均来自本项目实际训练与平台实测。

---

## 一句话

> **找一个在"检测 / 接地"上专门训练过的 3B 模型做基座，再用训练免费的注意力置信度做低置信回退——比所有自研/微调方案都强。**

## 关键数字

| 阶段 | 方案 | 平台分 |
|---|---|---|
| 基线 | Qwen3-VL-8B 直接坐标生成 | 0.43 |
| 候选融合 | Qwen3-VL + Grounding DINO 保守 hybrid | 0.45 |
| 自研定位器 | DINOv2 + Qwen 语义 + Decoder | 0.12（域迁移失败）|
| **强基座** | **Rex-Omni-3B 零样本** | **0.6926** |
| **推理侧** | **+ MTLA 低置信回退** | **0.6984**（历史最高）|
| 后训练 | SFT-v2（183k 官方风格合成数据）| **0.68** |
| 后训练 | SFT-v4（291k +VG/VLM-rewrite/LVIS）| RGBDT500 0.7875→0.8125 |
| **复赛主提交** | **Rex-Omni 后训练权重** | **0.708** |

> 详细实验记录：[`docs/experiments_log.md`](docs/experiments_log.md)

## 项目架构

```
LLM/
├── configs/           # 训练与推理配置（13 个 yaml/json）
├── scripts/           # 训练、推理、数据构建脚本
├── models/            # 模型定义（融合、GDINO、Candidate Scorer 等）
├── engine/            # 训练引擎（trainer + evaluator）
├── data/              # 数据集加载与预处理
├── utils/             # 工具（metrics、postprocess、MTLA 提取）
├── reference/MTLA/    # Propose and Attend 的官方实现（submodule）
├── docs/              # 详细文档（见下）
└── tests/             # 单元测试
```

## 核心技术栈

- **基座**：[Rex-Omni](https://arxiv.org/abs/2510.12798)（Qwen2.5-VL-3B + 1000 坐标 token，22M 监督 SFT+GRPO）
- **推理侧置信度**：[MTLA - Propose and Attend](https://github.com/TalRemez/MTLA.git)（Amazon，零训练注意力置信度）
- **候选检测器**：[Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)（IDEA Research，开放集检测）
- **融合策略参考**：[Thermo-VL](https://arxiv.org/abs/2605.21882)（门控残差注入）、[CFT](https://arxiv.org/abs/2111.00273)（跨模态注意力）、[RDTTrack](https://arxiv.org/abs/2509.24741)（RGBDT 三模态跟踪）

## 快速上手

```bash
# 1. 准备依赖（详见 requirements.txt）
pip install -r requirements.txt

# 2. 准备官方数据（按 docs/dataset_download.md 的目录结构）
# 注意：官方测试集无标注，仅用于推理

# 3. 复现 Rex-Omni 零样本基线
python scripts/predict.py --config configs/zeroshot.yaml

# 4. 复现 MTLA 低置信回退
python scripts/predict_with_mtla.py --config configs/mtla_fallback.yaml
```

## 核心方法论（七条教训）

1. **专门检测训练 × 数据域匹配 > 模型规模** —— Rex-Omni 3B > Qwen3-VL 8B
2. **推理侧改进 > 微调**（官方域无标注时）—— MTLA 0.6984 > 所有 LoRA / 融合
3. **公开数据验证不迁移官方域** —— 唯一真值是平台分
4. **训练分布与目标分布的距离**决定微调成败
5. **模型输入契约必须逐字对齐**（hint 缩放、prompt 格式、类别词）
6. **数据质量 > 数据数量**（程序化规则会放大缺陷）
7. **工程铁律**：optimizer 参数打印、权重指纹、装配一致性

## 文档导航

| 文档 | 内容 |
|---|---|
| [`docs/project_report.md`](docs/project_report.md) | **工程文档**：赛题 + 数据 + 方案演进 + 效果（建议先读）|
| [`docs/work.md`](docs/work.md) | 官方规则与数据契约 |
| [`docs/experiments_log.md`](docs/experiments_log.md) | 实验日志（含平台实测数字）|
| [`docs/summary.md`](docs/summary.md) | 版本化复盘总结（v1 ~ Exp027）|
| [`docs/improvement_ideas_classified.md`](docs/improvement_ideas_classified.md) | 改进方案分类归档 |
| [`docs/tech_share_presentation.pdf`](docs/tech_share_presentation.pdf) | **技术分享 PPT**（37 页，含 Rex-Thinker 与 Disclaimer）|
| [`docs/learning/`](docs/learning/) | 7 册学习方法论文笔记（详细）|

## 关键贡献

- **方法学**：系统对比 7 种 grounding 范式（自研定位器 / GDINO / Rex-Omni / Rex-Thinker / Thermo-VL / CFT / DUALVISION / RDTTrack / MTLA / GRPO），给出可复用的方法选择框架
- **工程证据**：公开验证高 ≠ 官方有效——`RefCOCO val 0.8986 → 官方 RGB-only 0.1237` 的域迁移失败，给出 6 次证伪方向
- **实战代码**：
  - `engine/trainer.py` —— DDP 训练 + modality dropout + 权重指纹 + 装配一致性
  - `utils/rexomni_vision_attn.py` —— MTLA 注意力提取（monkey-patch + KV cache 切片）
  - `models/candidate_scorer.py` —— GDINO 候选 + ROI + Qwen 文本重排序

## 数据合规性

- 官方测试集**无 bbox**，仅用于推理（无训练、无伪标签、无模型选择）
- 公开数据（RefCOCO / SUN RGB-D / LLVIP / RGBDT500）仅用于验证机制
- 平台分仅在该竞赛初赛 / 复赛测试集上有效
- **本评测不构成对各方法的充分测评**（不同数据集 / query 风格下数字可能显著变化）

## 致谢

- IDEA Research（Rex-Omni / Grounding DINO）
- Amazon（MTLA / Propose and Attend）
- Johns Hopkins University（Thermo-VL）
- Tsinghua University（CFT）
- DeepSeek-AI（GRPO）

#multimodal #grounding #MLLM #GRPO #attention-confidence
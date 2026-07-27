---
name: multimodal-grounding-project
description: 项目级多模态视觉定位工作规程。处理 /root/code/LLM、LLM_data、RGB/Depth/IR referring grounding、Qwen3-VL、Grounding DINO、RefCOCO、训练、推理、评测或实验规划时都应使用，即使用户没有明确提到本技能。先核验项目事实和本地资源，再提出或执行方案，并将计划与结果写入 Markdown。
---

# 多模态视觉定位项目规程

## 1. 开始任务前

先读取与任务相关的项目事实，不凭旧对话或全局技能中的项目状态做决定：

1. 读 `docs/work.md` 获取官方规则、数据和提交契约。
2. 读 `docs/experiments_log.md` 获取已经执行的实验和平台分数。
3. 读 `docs/plan.md` 获取当前方案、代码阻塞和下一步。
4. 需要运行实验时，检查实际代码、配置、进程、日志和产物；文档中的“计划”不等于已经实现。
5. 冲突时按“官方规则 > 实际产物与平台结果 > 当前计划 > 历史日志 > 全局技能”的顺序判断。

`docs/experiment_log.md` 和 `docs/zeroshot_log.md` 是历史记录。它们与上述事实源冲突时，不用其旧结论覆盖新状态。

## 2. 项目与赛题背景

- 项目根目录：`/root/code/LLM`。
- 官方题目说明：`/root/code/LLM/docs/work.md`。
- 输入：空间对齐的 RGB、Depth、Infrared 图像和英文 Query。
- 输出：Query 唯一指向目标在 RGB 坐标系中的归一化 `xyxy` 边界框。
- 主指标：`ACC@0.5`，即 `IoU >= 0.5` 的 Query 比例。
- 提交必须覆盖全部 Query，只能替换 JSON 的 `bbox`，不能修改图像路径和 Query。

官方 `LLM_data` 是无 bbox 的测试集，共 9,555 条 Query、2,000 组三模态图像。项目中的 `data/splits/train.json` 和 `val.json` 只是按 image ID 做的无标签分块，不是监督训练集，也不能计算真实 validation `ACC@0.5`。

遵守以下边界：

- 不使用官方测试图像、Query、bbox 派生信息或人工标注训练模型。
- 不把测试集预测、伪标签或模型间一致性称为 ground truth。
- 可以使用合法公开训练数据，但在实验记录中写明来源、用途、许可和预处理。
- 没有真实标签时，不用主观可视化或预测间一致性宣称方法提升；平台分数只能证明对应已提交产物。

## 3. 当前可信状态

- Qwen3-VL 三模态直接生成坐标的平台 baseline 是 `0.43`。
- Qwen3-VL 与 RGB-only Grounding DINO 的保守 hybrid 是当前最高平台结果 `0.45`。
- Grounding DINO 大范围替换得到 `0.39`，已经证明有害。
- 多 Prompt 坐标中位数没有显著变化且约增加三倍耗时，不作为优先方向。
- 保留 `0.45` hybrid 作为提交基线和回退方案。
- Qwen3-VL 直接生成坐标只用于 baseline 或诊断，不作为主研究路线。

主路线的职责分工是：

```text
VLM/LLM：理解 Query、目标属性和场景关系，提供语义条件
视觉定位器：使用有 bbox 监督的空间特征和定位损失预测连续坐标
```

当前 RefCOCO 语义引导定位闭环已经通过以下工程验收：离线 cache 读取、bbox 归一化、semantic padding mask、单 batch forward/backward、checkpoint 重载、32 样本过拟合和独立 validation A/B。关键实现是 `data/refcoco_cache.py`、`models/semantic_grounding.py`、`scripts/train_refcoco.py` 和 `scripts/benchmark_refcoco_batches.py`。

这不等于模型已经达到可提交水平。现阶段的主要阻塞是旧 Qwen cache 的单体序列化和 Query 格式、冻结 DINOv2 特征尚未缓存、正式 RefCOCO 训练步数不足，以及 RGB 公开数据到赛题三模态数据的域迁移。

## 4. 本地资源与离线约束

环境无法可靠访问公网。加载资源前按以下顺序检查：

1. `/root/autodl-fs`，它实际指向 `/autodl-fs/data`，用于本地预训练权重、公开数据集和持久资源。
2. `/root/autodl-tmp`，用于完整赛题数据副本、实验缓存、日志和训练产物。
3. 项目内的 `configs/`、`outputs/` 和本机模型缓存。

已知本地资源包括 Qwen3-VL-8B-Instruct、Grounding DINO base、DINOv2 ViT-B/14、RefCOCO/RefCOCO+/RefCOCOg、部分 COCO train2014、SAM2 和部分预计算特征。具体路径和完整性每次使用前重新检查，不把“目录存在”当作“数据完整”。特别注意：

- 完整赛题数据当前位于 `/root/autodl-tmp/data/LLM_data`。
- `/root/autodl-fs/datasets/LLM_data` 的解压副本曾缺少绝大多数 infrared 文件。
- `/root/autodl-fs/LLM_data.zip` 是已校验的完整原始包。
- COCO train2014 和 NYU Depth V2 当前不完整，使用前必须重新核验。

禁止静默联网、自动下载或把远程模型名当作可用路径。缺少资源时，先告知用户：

- 资源名称与公开来源；
- 实验用途和为什么现有资源不能替代；
- 预计下载大小、解压空间和许可要求；
- 建议存放路径；
- 没有该资源时可执行的替代方案。

获得用户确认和资源就绪后再继续。

## 5. 96GB GPU 使用规则

本机 GPU 是约 96GB 的 RTX PRO 6000 Blackwell。优化目标是有效吞吐和实验周转，不是单纯占满显存。

正式训练或大规模预计算前：

1. 先用代表性 batch 跑 forward/backward smoke test。
2. 探测 batch size、精度模式和 DataLoader worker 数，记录 peak allocated/reserved memory 与 samples/s。
3. 对适合扩大 batch 的任务，通常将稳定显存目标设为约 60-80GB，并为验证和临时张量留余量。
4. 使用 bf16/AMP、`pin_memory`、`persistent_workers`、合理预取和批处理；确认配置中的 gradient accumulation 确实生效。
5. 启动后检查 GPU utilization、显存、CPU/IO 和吞吐。利用率低时先定位数据加载、逐样本推理或 padding 瓶颈。

若模型本身很轻，低显存不必然是问题。优先扩大 batch、并行预处理、分桶、缓存或运行相互独立且不会争用 IO 的任务。不要为了看起来“充分利用”而无意义增大模型。

未经说明，不启动预计数小时但只占约 16GB、且没有吞吐基准或资源利用理由的正式实验。确需低显存长跑时，先在计划中解释限制和已尝试的优化。

## 6. 实验工作流

### 6.1 立项

运行实验前，先在 `docs/plan.md` 或用户指定的 Markdown 中记录：

```markdown
### Exp-XXX：实验名称

- 假设：
- 与哪个 baseline 对比：
- 数据及合法来源：
- 唯一主要变量：
- 训练/验证划分：
- 主指标与辅助指标：
- 资源预算：预计时间、显存、磁盘：
- 通过标准：
- 提前停止条件：
- 产物目录：
```

没有真实可比较指标、没有明确对照或不能改变下一步决策的实验，不直接启动长跑。

### 6.2 分级验收

按成本从低到高推进，不跳级：

1. 静态数据、坐标契约和 shape 检查。
2. 单 batch forward、loss、backward、optimizer step。
3. checkpoint 保存、重载和预测一致性。
4. 约 32 个样本同集过拟合，确认模型和标签契约能学习。
5. 在同一小集做关键变量消融，例如 semantic tokens 与 null semantic；同集结果只证明机制生效，不代表泛化。
6. 使用公开 train 和独立 validation 做小规模 A/B，固定样本、随机种子和训练预算。
7. 探测 batch、DataLoader 和精度模式，再运行完整实验。

任一阶段失败就先修根因，不通过增加 epoch、扩大模型或运行全量掩盖错误。独立 validation 前，不把过拟合结果写成模型效果；平台提交前，不把公开数据 validation 提升写成赛题提升。

### 6.3 Batch 与资源决策

不要按“占用显存最多”选择 batch。至少同时记录和比较：

- mean step time 与 samples/s；
- peak allocated 和 reserved memory；
- 真实端到端 epoch 时间，包括图片 IO、预处理和 validation；
- 每 epoch optimizer updates 与计划总 updates；
- validation 指标和达到目标指标所需 wall-clock 时间。

如果 batch 扩大后 step time 同比例增加、samples/s 几乎不变，就没有显著速度收益。此时较小 batch 可能以相近时间提供更多 optimizer updates，但不能预先宣称精度更高；需要固定 epoch 和固定 optimizer steps 两种公平口径验证。大 batch 应配套评估 learning rate、warmup 和 scheduler，而不是只复用小 batch 配置。

冻结编码器且语义特征已预计算时，定位头显存很低是合理现象。优先预计算冻结 DINOv2 spatial features、规范 Qwen cache、优化 shard 和 IO，不为了占满 96GB 显存而扩大无收益的 batch 或模型。

Batch microbenchmark 必须和真实端到端训练区分：前者可用于测 GPU 计算上限，后者才反映图片读取、worker 启动、CPU-GPU 传输和 validation 成本。

### 6.5 每日收尾与 Handoff

用户暂停当天工作或准备交给后续 agent 时：

1. 确认训练、预计算和 benchmark 进程是否仍运行，记录 PID；未经用户要求不把临时任务留在后台。
2. 确认 GPU 状态、实验产物、日志、参数和 checkpoint 是否存在。
3. 将已执行结果追加到 `docs/experiments_log.md`，将下一决策更新到 `docs/plan.md`。
4. 创建或更新 `docs/handoffs/YYYY-MM-DD.md`，写明可信结论、失败项、产物路径、当前阻塞和下一步的第一项任务。
5. 更新 `docs/README.md` 指向最新 handoff。
6. 后续 agent 开始工作时先读最新 handoff，再核验实际进程和文件；不要仅凭 handoff 假设状态未变化。

Handoff 至少包含：

```markdown
# YYYY-MM-DD Handoff

## 今日完成
## 可信结果
## 代码与产物
## 已知限制
## 明日第一任务
## 暂不执行
## 运行状态
```

### 6.6 结果留存

实验结束或中止后，立即追加到 `docs/experiments_log.md`：

```markdown
### Exp-XXX：实验名称

- 状态：完成 / 中止 / 失败
- 代码与配置：
- 实际命令：
- 数据版本与样本数：
- 起止时间与耗时：
- GPU：峰值显存、平均利用率、samples/s：
- 指标：
- 产物路径：
- 异常或偏离计划：
- 结论：支持或否定了什么：
- 下一步：
```

如实记录失败、跳过项和未经平台验证的结果。不得把未提交变体写成已提升，也不得用“看起来更合理”代替指标。

## 7. 当前推荐顺序

1. 保留并校验 `0.45` hybrid 主提交，不在新模型有平台证据前替换。
2. 将现有 Qwen RefCOCO 单体 cache 转换为规范 shards，保存单条 Query、sample ID、image ID、normalized bbox、token mask 和 manifest；不重新运行 Qwen。
3. 对冻结 DINOv2 spatial features 做一次性预计算，消除每个 epoch 重复图片解码和 backbone 前向。
4. 使用完整 RefCOCO train 和独立 validation，优先以 `batch=128` 或 `256`、足够 optimizer updates 做 semantic/null semantic 正式 A/B。
5. 根据 validation 指标和达到指标的 wall-clock 时间选择 batch、学习率、warmup 和训练轮数，不按显存占用选择。
6. 只有语义版本在独立公开 validation 稳定优于对照后，才构建官方全部 9,555 条 Query 的推理入口；官方数据不划分、不训练。
7. 只有合法公开的三模态 grounding 数据就绪后，才训练和比较 RGB、RGB+Depth、RGB+IR、RGB+Depth+IR。
8. 最终通过平台结果与 `0.45` baseline 比较，未提交结果不宣称赛题提升。

当前第一任务以 `docs/README.md` 指向的最新 handoff 为准。当前第一技术目标是规范并分片缓存，而不是继续扩大 batch、重跑 Qwen cache 或搜索无标签 hybrid 阈值。

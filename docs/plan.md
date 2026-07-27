# 多模态视觉定位方案规划

> 本文是当前可执行方案，不把历史设想、已完成实验和未实现代码混在一起。所有新模型都必须和现有 baseline 做对照；没有验证结果，不提前宣称方案有效。

## 一、任务与目标

任务输入是四个模态：

```text
RGB 可见光图像 + Depth 深度图 + IR 热红外图像 + 英文 Query
```

任务输出是 Query 所指目标在 RGB 图像中的归一化边界框：

```text
[x1, y1, x2, y2]
```

坐标范围为 `[0, 1]`，其中 `(x1, y1)` 是左上角，`(x2, y2)` 是右下角。评测指标为 `ACC@0.5`：预测框与真实框的 IoU 大于等于 0.5 才计为正确。

正式提交时只能替换 JSON 中的 `bbox` 字段，必须保留 `visible`、`infrared`、`depth` 和 `query` 字段。坐标不能越界、反向、为空或包含 NaN。

详细任务契约见 [`work.md`](./work.md)。

## 二、当前状态

### 2.1 已完成基线

当前真正跑通的是 Qwen3-VL 零样本推理链路：

```text
RGB + Depth + IR + Query
          ↓
      Qwen3-VL
          ↓
   文本生成 bbox 坐标
```

已有结果：

- Qwen3-VL 直接输出坐标：平台分数约 `0.43`。
- Qwen3-VL 与 Grounding DINO 的保守 hybrid：文档记录最高约 `0.45`。
- batch 推理已经在零样本链路中验证过，`batch=32` 时 GPU 利用率明显高于单样本。

这条链路必须保留为回归基线和最终回退方案。新模型没有稳定证据超过它之前，不替换它。

### 2.2 当前 Exp002 的真实定位

Exp002 原计划是：

```text
RefCOCO/RefCOCO+/RefCOCOg
          ↓
预计算 Qwen3-VL semantic tokens
          ↓
训练视觉定位 Decoder + BBox Head
```

它是一个定位器验证实验，不是完整的赛题三模态端到端方案。目前仍存在以下断点：

- 预计算脚本逐样本调用 Qwen，速度和显存利用率不理想。
- `train_refcoco.py` 尚未真正读取预计算结果，仍会重新计算 Qwen。
- RefCOCO 的 COCO bbox 是像素坐标，当前代码没有可靠完成 `[x,y,w,h]` 到归一化 `xyxy` 的转换。
- 当前 RefCOCO 训练实际只使用 RGB，Depth 和 IR 没有进入视觉分支。
- 当前 DINO/FPN、Decoder、配置字段和 checkpoint 接口之间还有不一致。

因此，Exp002 的文件不能直接视为已经验证的训练方案。

### 2.3 本方案的核心判断

VLM 直接生成坐标可以作为 baseline，但不作为主路线。原因是语言模型生成的四个数字不是专门的几何回归任务，容易出现：

- 框偏小或偏向图像中心；
- 坐标顺序和边界不稳定；
- 文本生成误差直接变成位置误差；
- 视觉 token 中的精确空间结构没有被专门的定位损失充分利用。

新的主路线不修改 Qwen3-VL 或 Transformers 的底层实现，而是改变项目自己的职责分工：

```text
Qwen3-VL：理解 Query 和场景，提供语义条件
定位网络：融合语义与三模态空间特征，直接回归 bbox
```

## 三、主方案：语义编码器 + 三模态定位器

### 3.1 整体结构

```text
RGB ───────┐
Depth ─────┼──► 三模态空间分支 ──► 多尺度视觉特征
IR ────────┘                              │
                                         │
RGB + Depth + IR + Query ─► Qwen3-VL ────┤
                              语义 tokens │
                                         ▼
                              Cross-Attention Locator
                                         │
                                         ▼
                              BBox Head: cx, cy, w, h
                                         │
                                         ▼
                              归一化 xyxy bbox
```

### 3.2 Qwen3-VL 的职责

第一版冻结 Qwen3-VL：

- 沿用已经验证的三张 PIL 图加 Query 的 processor 输入方式；
- 使用 `output_hidden_states=True` 提取语义特征；
- 根据 Processor 实际返回的 token mask 提取视觉/语义 token，不通过图片尺寸猜 token 数量；
- 将输出通过固定 projection 映射到 Decoder 维度，例如 `4096 -> 256`；
- 训练阶段优先读取离线 cache，不在每个 batch 中重复加载或调用 Qwen。

第一版不启用 LoRA。只有冻结 Qwen + 定位器已经证明有效后，才单独评估 LoRA 是否有收益。

### 3.3 三模态空间分支

空间分支必须真正同时消费三种视觉模态：

```text
RGB   ─► RGB stem
Depth ─► Depth stem
IR    ─► IR stem
           ↓
    concat + 1x1 fusion
           ↓
       spatial backbone
           ↓
       multi-scale FPN
```

要求：

- RGB、Depth、IR 使用独立 stem，保留模态差异；
- Depth 保留 16-bit 信息，处理无效深度并裁剪到合理范围；
- IR 使用热强度信息，不把伪彩色当作普通 RGB 语义；
- 所有 resize、crop、flip 必须三路同步，bbox 同步变换；
- 如果使用冻结 DINOv2，必须在其前面加入明确的三模态融合 adapter，不能沿用当前绕过 stem、只输入 RGB 的路径；
- DINO 输入的 ImageNet normalization 与赛题数据的 `[0,1]` 预处理必须分开定义。

448 输入时，空间特征优先保持类似 `32x32、16x16、8x8、4x4` 的尺度。不要把所有层上采样到 `256x256` 后再 flatten，否则每个样本可能产生约 65K 个视觉 token，Cross-Attention 成本过高。

### 3.4 定位 Decoder 与 BBox Head

定位 Decoder 使用一个注册在模型中的 learnable object query，融合两类信息：

- semantic cross-attention：Query 条件对应的 Qwen 语义 token；
- spatial cross-attention：三模态空间分支的多尺度视觉 token。

第一版去掉当前每次 forward 临时创建、不会被 optimizer 更新的 coarse-bbox positional MLP。先验证稳定的 learnable query + cross-attention 路线。

BBox Head 不独立预测四个无约束坐标，而是预测：

```text
cx, cy, width, height
```

其中中心点通过 sigmoid 限制在 `[0,1]`，宽高使用正值参数化并限制最大范围，再转换为：

```text
x1 = cx - width / 2
y1 = cy - height / 2
x2 = cx + width / 2
y2 = cy + height / 2
```

最后统一 clamp 到 `[0,1]`，保证训练和推理中的框具有合法顺序。损失使用：

```text
Loss = L1(pred_xyxy, gt_xyxy) + λ * GIoU(pred_xyxy, gt_xyxy)
```

训练日志必须记录 epoch、loss、learning rate、显存占用、IoU 和 `ACC@0.5`。

## 四、直接三模态实施路线

本路线直接面向 RGB + Depth + IR，不把 RGB-only 结果误称为赛题最终结果。但每一步仍有明确验收门槛。

### Stage 0：数据和标签契约

在任何正式训练前完成：

1. 确认本地是否存在带 bbox 的赛题训练集。
2. 确认 RGB、Depth、IR 文件存在、尺寸可对应、空间确实对齐。
3. 确认 Query、图片 ID、bbox 一一对应。
4. 统一内部 bbox 格式为 RGB 坐标系归一化 `xyxy`。
5. RefCOCO 等 COCO 标注若为像素 `[x,y,w,h]`，先转为像素 `xyxy`，再除以原图宽高。
6. 按 image ID 分组划分训练/验证，避免同一张图的不同 Query 泄漏到两边。
7. 输出数据报告：样本数、尺寸、bbox 分布、缺失文件数、无效 Depth 比例。
8. 抽样可视化 RGB、Depth、IR 和 bbox，确认坐标没有偏移。

硬性规则：官方测试集 Query 没有真实 bbox，只能用于推理和最终提交，不能用于监督训练或人工生成标签。如果当前本地没有带 bbox 的赛题训练集，必须把训练目标明确写为“外部有标注数据预训练/伪标签研究”，不能声称已经完成赛题监督训练。

### Stage 1：三模态模型形状验收

先不跑完整数据，使用少量样本验证：

- 三路输入 tensor shape 和 dtype 正确；
- Qwen cache 与样本 ID、Query、图片路径对应；
- FPN 每个尺度的 channel 和 spatial shape 正确；
- semantic projection、visual projection、Decoder 和 BBox Head shape 正确；
- forward、loss、backward、optimizer.step 都能完成；
- checkpoint 可以保存并重新加载。

这一步失败时不扩大 batch、不增加 epoch，先修数据契约或张量 shape。

### Stage 2：Qwen 三模态特征缓存

Qwen 预计算只执行一次，训练阶段直接读取 cache。

预计算要求：

- 使用批处理，而不是固定逐样本调用；
- 先探测 `batch=2、4、8、16`，以吞吐最大且显存稳定为准；
- 根据视觉 token 长度分桶，减少 padding 浪费；
- 使用 `torch.inference_mode()` 和合适的 bf16；
- 每隔固定样本数保存 shard/checkpoint，支持中断续跑；
- 每个 shard 保存 sample ID、semantic tokens、token mask、Query、image path 和 bbox；
- 训练 dataset 只按 sample ID 读取对应特征，严禁凭列表序号盲目对齐。

目标是把 96GB GPU 的显存和计算资源用于更高吞吐，而不是强行追求 100% 显存占用。应以实际 `samples/s` 和稳定性为准。

### Stage 3：三模态定位器训练

首轮策略：

- Qwen 冻结；
- 空间 backbone 根据显存和数据量决定冻结或只训练 adapter；
- 训练三模态 fusion、FPN、semantic projection、Locator Decoder 和 BBox Head；
- 使用 AMP/bf16；
- 使用 DataLoader 的 `pin_memory`、合理 `num_workers`、`persistent_workers`；
- 设置真正生效的 gradient accumulation；
- 保存 `last`、`best` 以及带 optimizer/scheduler/config 的 checkpoint；
- 验证阶段计算 IoU、ACC@0.5，并保存少量预测框用于可视化。

训练和验证必须使用同一坐标契约。训练 checkpoint 必须由正式预测入口直接加载，不能产生只能由另一个脚本读取的私有格式。

### Stage 4：四组模态消融

必须至少比较：

| 实验 | 输入 | 目的 |
|---|---|---|
| A | RGB | 空间基线 |
| B | RGB + Depth | 验证深度贡献 |
| C | RGB + IR | 验证热红外贡献 |
| D | RGB + Depth + IR | 完整方案 |

所有实验固定训练/验证划分、损失、训练轮数和评测脚本。不能只凭训练 loss 判断模态有效，必须看验证 `ACC@0.5` 和最终平台结果。

### Stage 5：可选 LoRA 与最终推理

只有 Stage 3/4 已经稳定并超过现有 `0.45` baseline 后，才做：

- Qwen attention 层 LoRA；
- 三模态输入 adapter 微调；
- backbone 解冻比例和学习率分组；
- 多 checkpoint 或保守模型融合。

最终推理必须：

1. 读取官方 Query JSON；
2. 读取三模态图片；
3. 通过同一模型加载入口生成 bbox；
4. 只替换 `bbox` 字段；
5. 对 bbox 做 clip、排序、NaN、空框和范围校验；
6. 生成 zip 前统计所有无效预测数量。

## 五、必须先修复的当前代码问题

在正式训练前，以下问题必须关闭：

- 预计算结果没有被 `train_refcoco.py` 消费，训练时重复计算 Qwen；
- RefCOCO bbox 没有可靠完成像素 `[x,y,w,h]` 到归一化 `xyxy` 的转换；
- 当前 RefCOCO 训练只使用 RGB，Depth/IR 参数被忽略；
- DINO/FPN 存在配置字段、输入通道和空间尺度不一致；
- Decoder 的视觉输入通道与 FPN 输出不匹配；
- semantic token 维度与 Decoder 输入维度不匹配；
- token padding 没有传递 attention mask；
- coarse bbox positional MLP 在 forward 中临时创建，参数不受训练；
- BBox Head 的四个独立 sigmoid 坐标不能保证训练框合法；
- 训练入口和预测入口 checkpoint schema 不一致；
- 本地数据路径与默认配置不一致，训练不能依赖运行时联网下载；
- scheduler、AMP、gradient accumulation 和 DataLoader 性能配置需要实际生效并通过 smoke test；
- 当前 `vlm_branch.py` 中 Qwen 类名、processor 调用和三模态 tensor 输入仍是未验证的 skeleton，第一版应复用已验证的三张 PIL 图输入路径。

## 六、验收标准

### 数据验收

- 所有关键图像文件存在；
- 三模态尺寸和空间对应关系通过抽样检查；
- bbox 可视化正确；
- train/validation 不共享 image ID；
- 官方测试数据没有被用于训练。

### 模型验收

- 单 batch forward/backward 成功；
- 所有 feature shape 有明确日志；
- checkpoint 可保存、加载和推理；
- 三路输入确实会改变对应实验的空间特征，而不是被静默忽略。

### 训练验收

- 小规模数据 loss 下降并能过拟合；
- 验证集 `ACC@0.5` 可复现；
- 训练日志包含 loss、LR、GPU memory 和验证指标；
- 没有 OOM、NaN 或反向 bbox。

### 结果验收

- 新模型与现有 `0.45` Qwen hybrid 使用相同输出校验和可比数据评测；
- 新模型只有在验证集和平台结果都有证据提升时才作为主提交；
- 如果新模型不稳定，保留并使用现有 baseline，不因训练成本继续扩大无效实验。

## 七、当前执行顺序

```text
1. 保留 0.45 baseline 和当前输出文件
2. 核验本地赛题训练标签与三模态文件
3. 修复 bbox 坐标、cache 对齐和模型 shape
4. 做三模态单 batch smoke test
5. 批量预计算 Qwen 特征并支持断点续跑
6. 训练冻结 Qwen 的三模态定位器
7. 做 RGB / RGB+D / RGB+IR / RGB+D+IR 消融
8. 与 0.45 baseline 公平比较
9. 只有有效时再尝试 LoRA、解冻和融合
```

这条路线的核心不是把 Qwen3-VL 改造成检测器，而是让它做它擅长的语义理解，把连续空间定位交给有 bbox 监督的专门网络。

## 八、下一阶段执行计划（2026-07-27）

### 8.1 当前决策

下一步不启动新的全量 Qwen 坐标推理，也不继续搜索无标签 hybrid 阈值。保留平台 `0.45` 的保守 hybrid 作为当前主提交和回退基线。

当前第一目标是利用合法公开的 RefCOCO 数据建立一个有真实 bbox、可计算 validation `ACC@0.5`、可保存并重载 checkpoint 的 RGB 定位训练闭环。在这个闭环中先验证专门定位器能否学习，再通过严格对照判断 Qwen semantic tokens 是否真正提供增益。

这一步不能直接证明赛题三模态模型有效，但能回答更基础且必须先回答的问题：当前数据转换、定位损失、Decoder 和评测链路是否正确。

### 8.2 Exp-007：RefCOCO cache 定位基线修复

- 假设：修复数据与模型契约后，冻结视觉/语义编码器、只训练定位模块，能够在公开 RefCOCO validation 上形成明显高于随机预测的可复现 `ACC@0.5`。
- 对照：同一数据划分、同一视觉特征和同一训练预算下，先跑不带 Qwen semantic tokens 的视觉定位器，再跑带 semantic tokens 的模型。
- 数据：本地 RefCOCO/RefCOCO+/RefCOCOg 与可用 COCO train2014 图像；只使用存在图像的合法公开样本，并按 image ID 防止 train/validation 泄漏。
- 主要变量：第一阶段只修正确性，不同时更换 backbone、损失、增强和优化器。
- 主指标：RefCOCO validation `ACC@0.5`；辅助指标为 mean IoU、median IoU、无效框数和 loss。
- 资源预算：先做分钟级 smoke test；完整训练预算只在 smoke test 和过拟合测试通过后确定。
- 通过标准：单 batch 可训练、checkpoint 可重载、32 样本可明显过拟合、完整 validation 指标可复现。
- 提前停止：bbox 可视化错误、cache/sample 不对应、loss 不下降、出现 NaN、无效框或 checkpoint 重载不一致。
- 产物：`/root/autodl-tmp/experiments/exp007_refcoco_locator/`；计划与结果分别写入本文和 `experiments_log.md`。

### 8.3 实施顺序

1. 不打断当前 `precompute_refcoco_features.py`；进程自然结束后核验每个 cache 文件是否可完整加载、样本数是否符合可用图片数量，并记录失败样本。
2. 明确 Parquet bbox 的真实格式，依据原图尺寸转换为归一化 `xyxy`，增加范围、顺序和抽样可视化检查。
3. 实现 cache loader，保存并核验 sample ID、Query、image ID、图像路径、semantic tokens 和 attention mask；不得依赖列表位置盲目对齐。
4. 修复 Decoder/FPN 输入维度、padding mask、合法 bbox 参数化、GIoU、gradient accumulation 和统一 checkpoint schema。
5. 完成单 batch forward、loss、backward、optimizer step，以及保存、重载、同输入预测一致性测试。
6. 在约 32 个样本上过拟合；若模型无法记住小样本，停止全量训练并排查数据、损失和梯度。
7. 先训练不带 Qwen semantic tokens 的视觉定位基线，再在其他条件不变时加入 semantic tokens，量化语义引导贡献。
8. 通过代表性 batch 探测显存和吞吐。对可扩大 batch 的训练，稳定显存目标通常为 60-80GB，并记录 peak memory、GPU utilization 和 samples/s；若显存较低是模型规模所限，说明原因并优先优化 batch、DataLoader 和缓存读取。
9. 完整 RefCOCO validation 通过后，再决定是否扩展到 RefCOCO+/RefCOCOg，以及是否寻找合法公开的 RGB-Depth-IR grounding 数据。

### 8.4 暂不执行

- 不使用官方无标签测试集训练、人工标注或构造被当作真值的 bbox。
- 不把现有 `data/splits/train.json` 和 `val.json` 称为监督训练/验证划分。
- 不重跑多 Prompt 集成或大范围 Grounding DINO 替换；已有平台结果已表明优先级低或有害。
- 不在定位闭环通过前启用 Qwen LoRA、解冻大 backbone、三模态复杂融合或多 checkpoint 集成。
- 不启动缺少吞吐基准、预计数小时且仅使用约 16GB 显存的正式实验；确有结构性限制时先记录原因。

### 8.5 后续决策门槛

只有满足以下条件，才进入赛题方向的模型升级：

1. RefCOCO validation 指标可复现，且预测、评测和 checkpoint 链路一致。
2. 带语义条件的定位器相对不带语义条件的对照有稳定收益。
3. 找到规则允许、许可清晰、具有所需模态与 bbox 的公开训练数据，或明确采用只在 RGB 上预训练再零样本迁移的受限方案。
4. 新模型在官方测试集推理后，通过实际平台结果超过 `0.45`，再替换当前主提交。

### 8.6 2026-07-27 修复进展

已完成：

- 按用户决定中止逐样本 RefCOCO 特征预计算，GPU 已释放；已有 RefCOCO 和 RefCOCO+ train/validation cache 可加载。
- 新增离线 RefCOCO cache Dataset，修复 bbox 像素 `xyxy` 到归一化 `xyxy` 的转换，并生成 semantic padding mask。
- 新增冻结 DINOv2、使用 Qwen semantic tokens 的专门定位模型；bbox 由中心点和尺寸构造，保证输出顺序合法。
- 修复 GIoU 公式、DINOv2 special token 权重加载、训练入口、bf16、梯度累积、统一 checkpoint 和 validation 指标。
- 8 样本 smoke test 已通过 forward、backward、optimizer step 和 checkpoint 重载。
- 32 样本同集过拟合对照中，带 Qwen 语义达到 `ACC@0.5=0.5938 / mean IoU=0.5680`，无语义对照为 `0.3125 / 0.3899`。该结果只证明语义引导生效，不是泛化成绩。

下一步阻塞：

1. 现有 8.2GB 单体 train cache 加载约需 168 秒，应先转换为 shards 或可随机访问格式，避免每次实验重复全量反序列化。
2. 旧 cache 的 Query 是一组 captions 的字符串表示；正式训练前应生成具有单条 Query、sample ID、mask 和 manifest 的规范 cache。
3. 使用独立 RefCOCO train/validation 跑语义/无语义正式 A/B，并做 batch size 吞吐探测。
4. 官方 `LLM_data` 不参与任何训练划分。现有 `data/splits/train.json`、`val.json` 仅作为历史分块文件保留，后续训练和评测入口不读取；模型确定后直接对全部 9,555 条 Query 推理。

### 8.7 Exp-008：RefCOCO 小规模独立验证与 batch 探测

- 目标：跑通公开 RefCOCO train 到独立 validation 的完整流程，并探测 `batch=64/128/256/512` 的训练吞吐和峰值显存。
- 数据：`refcoco_train.pt` 与 `refcoco_val.pt`；官方 `LLM_data` 不参与。
- batch 探测：单次加载 cache 后，在同一进程内对每个 batch 完成 warmup 和多次 forward/backward，记录平均 samples/s、peak allocated/reserved memory；发生 OOM 时如实记录并继续较小 batch。
- 小规模 A/B：使用约 10% RefCOCO train、独立 RefCOCO validation、3 epochs、相同随机种子和优化配置。
- A：RGB DINOv2 + null semantic；B：RGB DINOv2 + Qwen semantic tokens。
- 主指标：validation `ACC@0.5`；辅助指标：mean/median IoU、非法框、训练 loss、耗时、samples/s 和显存。
- 判定：该实验只判断全流程与初步泛化方向，不把 3 epoch 小样本结果当作最终模型成绩。
- 已知限制：现有 cache 是 8.2GB 单体文件，且旧 Query 可能是同一 reference 的 captions 数组字符串；10% 截断样本可能受原始排序影响。
- 产物目录：`/root/autodl-tmp/experiments/exp008_refcoco_pilot/`。

### 8.8 当日收尾与下一任务

2026-07-27 的代码、实验、产物、限制和运行状态已归档到 [`handoffs/2026-07-27.md`](./handoffs/2026-07-27.md)。当天不再启动实验，后台训练、预计算和 benchmark 进程均已结束。

下一次工作从 cache v2 转换开始：先将已有 Qwen 单体 cache 转为可随机访问的规范 shards，并完成样本数、随机 token、路径和归一化 bbox 的一致性检查；不要重新运行 Qwen。完成后再预计算 frozen DINOv2 spatial features，最后进行完整 RefCOCO semantic/null semantic A/B。


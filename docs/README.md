# 文档索引

> 项目级工作规程见 [`../.agents/skills/multimodal-grounding-project/SKILL.md`](../.agents/skills/multimodal-grounding-project/SKILL.md)。处理本项目的训练、推理、评测和实验规划前，先按该规程核验事实、数据合规性和本地资源。

## 最新 Handoff

- [`handoffs/2026-07-27.md`](./handoffs/2026-07-27.md)：今日代码与实验归档、可信结论、运行状态和明日第一任务。后续 agent 开始工作时先读此文件，再核验实际进程与文件状态。

## 当前事实源

| 优先级 | 文件 | 作用 |
|---:|---|---|
| 1 | [`work.md`](./work.md) | 官方规则、数据契约、评测和提交要求；其他文档不得覆盖 |
| 2 | [`experiments_log.md`](./experiments_log.md) | 已执行实验、实际平台分数和产物状态 |
| 3 | [`plan.md`](./plan.md) | 当前技术判断、代码阻塞、下一步和验收标准 |

发生冲突时，按“官方规则 > 实际产物与平台结果 > 当前计划 > 历史记录”判断。代码中的计划性 skeleton 不等于已经跑通。

## 历史记录

| 文件 | 内容 | 使用限制 |
|---|---|---|
| [`zeroshot_log.md`](./zeroshot_log.md) | 零样本推理实施过程 | 用于排查旧 pipeline，不采信其中过期的主提交描述 |
| [`experiment_log.md`](./experiment_log.md) | 较早环境与 Exp001-003 记录 | 仅作历史参考，状态可能落后于 `experiments_log.md` |

## 当前状态摘要

- 官方 `LLM_data` 是无 bbox 测试集，不能用于监督训练或真实 validation 评测。
- Qwen3-VL 三模态直接坐标 baseline 的平台分数为 `0.43`。
- Grounding DINO 大范围替换的平台分数为 `0.39`，已经确认有害。
- Qwen3-VL + Grounding DINO 保守 hybrid 的平台分数为 `0.45`，是当前最好结果和回退基线。
- 当前主研究方向是“VLM/LLM 语义引导 + 有 bbox 监督的专门定位器”，不是继续让 VLM 直接生成坐标。
- RefCOCO 工程闭环、32 样本过拟合和独立 validation pilot 已跑通；语义版本有小幅正向结果，但尚未达到可迁移提交的证据强度。
- 当前工程第一任务是将旧 Qwen 单体 cache 规范化并分片，再预计算 frozen DINOv2 spatial features。

## 建议阅读顺序

1. 读最新 handoff，了解上次收尾状态和第一任务。
2. 读 `work.md`，确认任务和合规边界。
3. 读项目级 `SKILL.md`，确认文件、资源和实验规程。
4. 读 `experiments_log.md`，避免重复已失败实验。
5. 读 `plan.md`，执行当前下一阶段。
6. 只有排查历史实现时再读旧日志。

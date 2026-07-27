# 实验记录

## 环境信息

| 项目 | 详情 |
|:-----|:------|
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition (97 GB) |
| 系统盘 | overlay 30G |
| 数据盘 | autodl-tmp 50G (本地) + autodl-fs 200G (共享) |
| 代码 | `/root/code/LLM/` |

## 资源位置

### 数据集 (`/root/autodl-fs/datasets/`)
| 数据集 | 大小 | 说明 |
|:-------|:----:|:------|
| `LLM_data/` | 9.0G | 赛题数据（RGB+Depth+IR+Query，9555 条） |
| `coco/train2014/` | 11G | COCO train2014 图片（~69K 张） |
| `nyu_depth_v2/` | 2.8G | NYU Depth V2 标注（备用） |
| `refcoco/` | 38M | RefCOCO 标注 |
| `refcocog/` | 44M | RefCOCOg 标注 |
| `refcocoplus/` | 38M | RefCOCOplus 标注 |

### 模型权重 (`/root/autodl-fs/weights/`)
| 模型 | 大小 | 用途 |
|:-----|:----:|:------|
| Qwen3-VL-8B-Instruct/ | ~16G | VLM 语义分支 |
| `dinov2_vitb14_reg4_pretrain.pth` | 331M | 视觉 Backbone |
| `sam2_hiera_tiny.pt` | 149M | 框后处理精修 |
| `swin_base_patch4_window7_224.pth` | 332M | 备选视觉 Backbone |

### 训练产出 (`/root/autodl-tmp/`)
| 路径 | 说明 |
|:-----|:------|
| `outputs/` | 模型权重、日志 |
| `data/` | 数据集缓存、预处理文件 |

---

## 实验列表

### Exp 001: Qwen3-VL 零样本 baseline
- **状态**: ✅ 已完成
- **日期**: 2026-07-25
- **描述**: Qwen3-VL-8B-Instruct 直接输出坐标文本
- **配置**: `configs/zeroshot.yaml`
- **结果**: 9555 条，9331 成功，177 错误
  - 平均推理速度: 0.23s/q
  - 预测框倾向于小框（中位数 0.10×0.14），分布偏图像中心
  - 具体分数待提交平台获取
- **产出**: `outputs/predictions.json`
- **代码**: `models/zero_shot_qwen.py`, `scripts/run_zeroshot.py`

### Exp 002: [准备中] RefCOCO 训练 Decoder (Qwen3-VL 语义 + DINOv2 视觉)
- **状态**: 🏗️ 待预计算 Qwen3-VL 特征后启动
- **目标**: 在 RefCOCO 上训练 Cross-Attention Decoder + BBox Head
  - ✅ Qwen3-VL 语义分支（预计算特征，训练时不加载 VLM）
  - ✅ DINOv2 视觉 Backbone（冻结）
  - ✅ Cross-Attention Decoder + BBox Head（可训练）
- **数据**: RefCOCO/RefCOCO+/RefCOCOg train (共 ~126K query-bbox 对)
- **指标**: RefCOCO val ACC@0.5
- **脚本**: 
  - Step 1: `scripts/precompute_refcoco_features.py`（跑一次，预计算语义特征）
  - Step 2: `scripts/train_refcoco.py`（训练 Decoder）
- **配置**: batch_size=8, lr=1e-4, epochs=20, image_size=448
- **产出**: `/root/autodl-tmp/exp002/`

### Exp 003: [待规划] 赛题零样本推理 (Decoder 版)
- **目标**: 用 Exp 002 训练好的 Decoder 替换 Qwen3-VL 文本输出，在赛题上推理

---

## 方案演进

### 方案 A（当前规划）：VLM 语义引导 + Decoder 定位 ✅
```
VLM 角色 ── 语义理解引擎（读懂 Query，理解多模态场景）
定位角色 ── Cross-Attention Decoder（精确定位 BBox）
```

**架构**:
- Branch 1: 多模态视觉编码（Conv Stem + DINOv2 Backbone + FPN）
- Branch 2: VLM 语义分支（Qwen3-VL, 冻结, LoRA 可选）
- Fusion: Cross-Attention Decoder + BBox Head

### 零样本方案（已跑）：VLM 直接文本输出
```
Qwen3-VL 看三张图 + Query → 文本坐标
精度天花板低，仅作为 baseline
```

---

## 改动记录

| 日期 | 改动 | 文件 | 说明 |
|:----:|:-----|:-----|:------|
| 07-25 | 修复 dataset.py 路径写反 | `data/dataset.py` | depth_path 和 ir_path 互换 |
| 07-25 | 新增零样本推理模块 | `models/zero_shot_qwen.py` | Qwen3-VL 直接输出坐标 baseline |
| 07-25 | 新增零样本推理入口 | `scripts/run_zeroshot.py` | 批量推理 |
| 07-25 | 新增数据集划分 | `scripts/split_queries.py` | 按 image_id 8:2 划分 train/val |
| 07-25 | 下载 COCO train2014 | — | 69K 张到 autodl-fs/datasets |
| 07-25 | 下载 RefCOCO/+/g 标注 | — | 到 autodl-fs/datasets |
| 07-25 | 下载 SAM2 权重 | — | 到 autodl-fs/weights |
| 07-27 | 数据迁移 | — | 数据集→autodl-fs/datasets，权重→autodl-fs/weights |

---

## 待办

- [x] 适配 vision_encoder.py 加载本地 DINOv2 .pth
- [ ] 修复 vlm_branch.py 中 Qwen3-VL 兼容性
- [ ] 实现 RefCOCO 数据加载器（对接 COCO 图片）
- [ ] 在 RefCOCO 上训练 Decoder
- [ ] 集成到赛题推理流程
- [ ] 零样本可视化评估
- [ ] FLIR / NYU 数据集下载（填表）

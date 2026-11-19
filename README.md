# SIA-GPQA 复现项目

> 复现并扩展 [SIA: Self Improving AI with Harness & Weight Updates](https://arxiv.org/abs/2605.27276)
> （arXiv 2605.27276）的完整实验：**Harness 模式**、**Weight 模式（权重更新）**、以及两者的**串联**。

本项目包含：
- **原始 SIA 源码**（`sia-src/`，MIT License，版权归原项目）
- **复现脚本**（`scripts/`：轨迹提取、训练、评测、教师生成）
- **本地推理服务**（`inference/`：Qwen3-8B OpenAI 兼容批处理服务）
- **完整实验记录**（`docs/`：含 6 组 harness + 13 组权重 + 1 组串联实验结果）
- **训练好的 LoRA 模型**（ModelScope: [modelzhang/sia-qwen3-8b-gpqa-lora](https://www.modelscope.cn/models/modelzhang/sia-qwen3-8b-gpqa-lora)）
- **完整轨迹数据集**（ModelScope: [modelzhang/sia-gpqa-harness-trajectories](https://www.modelscope.cn/datasets/modelzhang/sia-gpqa-harness-trajectories)）

---

## 一、核心结果

| 模式 | 底座 / 任务 | 结果 | 判定 |
|---|---|---|---|
| **Harness** | deepseek-v4-flash / GPQA | 89.90% → **94.44%**（+4.5pt） | ✅ 有效 |
| **Harness** | Qwen3-8B / GPQA | 未作答 190 → **16 题** | ✅ 修复工程缺陷 |
| **Weight** | Qwen 2.5-3B / lawbench | 18.95% → **26.83%**（p<0.01） | ✅ 有效 |
| **Weight** | Qwen3-8B / GPQA | 65.00%（12 个配置，均未突破） | ❌ 能力上限 |
| **串联** | Qwen3-8B / GPQA disciplined | 62.50% → **80.00%**（McNemar p=0.12） | ⚠️ 大幅提升，统计待确认 |

**详细报告见 [docs/FINAL_REPORT.md](docs/FINAL_REPORT.md)**（含完整 McNemar 检验与逐题数据）。

---

## 二、项目结构

```
open_source/
├── sia-src/              # 原始 SIA 源码（v0.6.0，MIT）
├── scripts/              # 复现脚本
│   ├── extract_harness_traj.py   # 从 SIA 执行轨迹提取训练数据（自动排除 held-out）
│   ├── build_final.py            # 合并 harness 轨迹 + 教师解法 → 训练集
│   ├── train_selfdist.py         # LoRA SFT（拒绝采样自蒸馏）
│   ├── train_rest.py             # lawbench 版本（rejection sampling）
│   ├── train_law.py              # lawbench GRPO（备选）
│   ├── eval_final.py             # 双 prompt 对照评测（cot + disciplined）
│   ├── eval_single.py            # 单配置评测（便于并行）
│   ├── eval_base_law.py          # lawbench 基线评测
│   ├── gen_teacher_solutions.py  # 教师解法生成（不告知标准答案）
│   ├── teacher_retry.py          # 教师多次尝试（温度递增）
│   └── gen_dcot_v3.py            # D-CoT 偏好数据生成（负结果参考）
├── inference/            # 本地推理服务
│   ├── batch_server2.py  # Qwen3-8B OpenAI 兼容服务（批处理）
│   └── local_server.py   # 简化版（支持 LoRA 热切换）
├── docs/                 # 实验报告
│   ├── FINAL_REPORT.md   # 最终报告（推荐先读这个）
│   ├── REPORT.md         # 全景报告
│   ├── MODEL_CARD.md     # 模型卡片（ModelScope 同步）
│   ├── DATASET_CARD.md   # 数据集卡片（ModelScope 同步）
│   └── ...               # 各分项报告
├── requirements.txt      # Python 依赖
├── LICENSE               # MIT
└── README.md             # 本文件
```

---

## 三、快速开始

### 3.1 环境

```bash
pip install -r requirements.txt
# 需要额外安装：
#   torch>=2.5（CUDA 版）
#   transformers>=4.51（Qwen3 架构支持）
#   peft, trl, datasets, fastapi, uvicorn
```

### 3.2 数据准备

从 ModelScope 下载数据集放入 `data/`：

```bash
# https://www.modelscope.cn/datasets/modelzhang/sia-gpqa-harness-trajectories
# 期望布局：
# data/
# ├── gpqa/data/private/diamond_questions.json   # 评测 ground truth
# ├── selfdist/final_train.json                  # 训练集（109 条）
# └── out/test_ids.json                          # held-out 题号
```

### 3.3 训练

```bash
python scripts/train_selfdist.py \
    --traj data/selfdist/final_train.json \
    --out out/final_harness \
    --epochs 3 --lr 1.5e-5 --bs 1 --accum 4 --max_len 8192
```

### 3.4 评测

```bash
python scripts/eval_final.py out/final_harness/final_adapter out/eval.json 40
```

### 3.5 完整 pipeline（harness → 权重更新）

```bash
# 1. 运行 SIA harness（见 sia-src/README 或 docs/REPRODUCTION.md）
# 2. 提取轨迹
python scripts/extract_harness_traj.py
# 3. 构建训练集
python scripts/build_final.py
# 4. 训练 + 评测（同上）
```

---

## 四、关键发现（诚实总结）

### 4.1 三个重要的正向结果

1. **Harness 模式有效**：SIA 让 LLM 自主诊断执行轨迹中的缺陷并修复代码
   - 例如它自己发现 Python 的 `"" in "ABCD"` 恒为 `True` 陷阱
2. **Weight 模式在合适任务上有效**：lawbench +7.88pt（p<0.01）
3. **串联 + disciplined prompt 突破 base**：80.00% vs 62.50%

### 4.2 两条边界

1. **自我改进对"工程缺陷"有效，对"能力天花板"无效**：
   轨迹仍由同一个模型生成，prompt 改变的是"怎么组织答案"，不是"知道什么"
2. **feedback agent 的改进可能适得其反**：
   Gen 2 的"续写机制"推理合理但实际崩了（147/198 题未作答）

### 4.3 一个重要的机制发现

**训练是"条件激活"而非"内化"**：

| prompt | 标签使用率 | 准确率 |
|---|---|---|
| `cot`（无标签指令） | 0% | 62.50% |
| `disciplined`（显式要求控制标签） | 92.5% | **80.00%** |

必须有 prompt 触发，学到的结构化推理才能生效。

---

## 五、License

- SIA 原始源码：MIT（Copyright 2025 Hexo）
- 本项目复现脚本与文档：MIT（见 [LICENSE](LICENSE)）

---

## 六、引用

```bibtex
@article{hebbar2026sia,
  title   = {SIA: Self Improving AI with Harness \& Weight Updates},
  author  = {Hebbar, Prannay and Manawat, Yogendra and Verboomen, Samuel and others},
  journal = {arXiv preprint arXiv:2605.27276},
  year    = {2026}
}
```

---

## 七、致谢

- 训练硬件：NVIDIA A800-SXM4-80GB
- 教师模型：`deepseek-v4-pro`（LiteLLM OpenAI 兼容中转）
- meta/feedback agent：`deepseek-v4-pro`（pydantic-ai）

*生成于 2026-11-18*
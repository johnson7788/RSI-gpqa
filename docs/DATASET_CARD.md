# SIA-GPQA-Harness-Trajectories

> SIA（Self Improving AI，arXiv 2605.27276）在 GPQA 任务上的**完整轨迹数据集**。
> 包含 harness 自我改进产出的真实推理轨迹、教师补充解法、以及可直接用于 LoRA SFT 的训练集。

配套模型：[modelzhang/sia-qwen3-8b-gpqa-lora](https://www.modelscope.cn/models/modelzhang/sia-qwen3-8b-gpqa-lora)

---

## 一、数据集概述

本数据集记录了 **Qwen3-8B 在 GPQA 上的自我改进全流程**：

1. SIA harness 模式运行 3 代，meta-agent（`deepseek-v4-pro`）逐代修改 target agent 的代码与 prompt
2. Gen 1 产出 **198 条完整推理轨迹**，其中 87 题答对
3. 对失败题引入**教师模型**（`deepseek-v4-pro`）独立求解，仅保留答对的解法
4. 组合成可训练的偏好/自蒸馏数据

### 文件清单

| 文件 | 条数 | 大小 | 说明 |
|---|---|---|---|
| `harness_traj.json` | **69** | 716KB | Gen 1 的**正确**轨迹（已排除 held-out 40 题） |
| `teacher_solutions.json` | **52** | 39KB | 教师对失败题的独立解法（仅保留答案正确者） |
| `final_train.json` | **109** | 776KB | **推荐训练集**（69 harness + 40 教师，去重） |
| `mixed_train.json` | 136 | 1.6MB | 早期版本（84 base 采样 + 52 教师） |
| `eval_results/final_harness_eval.json` | — | — | 模型评测结果（含逐题详情） |

### 代码文件（`code/`）

| 文件 | 作用 |
|---|---|
| `extract_harness_traj.py` | 从 SIA 的 `agent_execution/*.json` 提取轨迹，**自动排除 held-out** |
| `build_final.py` | 合并 harness 轨迹 + 教师解法，去重生成最终训练集 |
| `train_selfdist.py` | LoRA SFT 训练（拒绝采样自蒸馏） |
| `eval_final.py` | 双 prompt 对照评测（cot + disciplined） |
| `eval_single.py` | 单配置评测（便于并行） |
| `gen_teacher_solutions.py` | 教师解法生成（**不告知标准答案**，仅校验结果） |
| `teacher_retry.py` | 教师多次尝试（温度递增，直到答对） |
| `batch_server2.py` | Qwen3-8B 本地 OpenAI 兼容推理服务（带请求批处理） |

---

## 二、数据格式

### 2.1 harness_traj.json / final_train.json

```json
{
  "id": 7,
  "prompt": "Question: Two quantum states with energies E1 and E2...\n\nOptions:\nA. ...\n\nThink step by step. End with a final line 'Answer: X' where X is one of A, B, C, D.",
  "response": "Thinking carefully...\n<TEMP_LOW>\nGivens: ...\n</TEMP_LOW>\n<TEMP_MID>\nCalculation: ...\n</TEMP_MID>\nAnswer: B",
  "source": "harness"
}
```

字段说明：

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | int | GPQA question id |
| `prompt` | str | 完整输入（含题目与选项） |
| `response` | str | 模型推理轨迹（含控制标签） |
| `source` | str | `"harness"` 或 `"teacher"` |
| `gold` | str | 正确答案字母（仅 harness_traj） |

### 2.2 teacher_solutions.json

```json
{
  "id": 13,
  "response": "Let me work through this carefully...\n\nAnswer: C",
  "teacher_pred": "C",
  "gold": "C",
  "ok": true,
  "attempt": 2
}
```

`ok: true` 表示教师答案与标准答案一致（**这是唯一允许进入训练集的条目**）。

---

## 三、数据隔离与防泄漏

### 3.1 held-out 严格隔离

| 集合 | 题数 | 用途 |
|---|---|---|
| GPQA 全部 | 198 | — |
| **held-out** | **40** | **仅用于评测，绝不参与训练** |
| 训练题 | 158 | 产生轨迹 |

提取脚本的统计输出：

```
held-out excluded: 40
stats: {total: 198, no_assistant: 0, empty: 16, correct: 69, incorrect: 74, heldout: 39}
usable trajectories: 69
```

> 注意 `heldout: 39` 是因为 40 题中有 1 题落在 `empty` 里（未作答），先被过滤。

### 3.2 教师解法的防泄漏设计

关键约束：**绝不把标准答案告诉教师**。

- 教师只看题目和选项，独立求解
- 解完后校验其答案是否与 ground truth 一致
- 只有 `ok: true` 的解法才保留

这样教师提供的是"可信的推理过程"，而不是"反推出来的答案"。

---

## 四、数据统计

### 4.1 长度分布

| 集合 | response 平均字符 | 平均 token（估） |
|---|---|---|
| harness_traj | ~17,700 | ~5,500 |
| teacher_solutions | ~8,000 | ~2,500 |

> Qwen3-8B 的思考链很长，这是 GPQA 这类难题的特征。

### 4.2 控制标签使用

| 集合 | 含 `<TEMP_LOW>` | 含 `<TEMP_MID>` |
|---|---|---|
| harness_traj | 高（受 prompt 影响） | 高 |
| teacher_solutions | 低（教师用自然语言推理） | 低 |

**这个差异是关键**：harness 轨迹带控制标签，教师解法不带。混合训练后，
模型学到的结构化推理需要在 prompt 触发时才能激活（详见模型卡片）。

---

## 五、使用方法

### 5.1 加载

```python
import json

traj = json.load(open("harness_traj.json"))
print(f"{len(traj)} trajectories")
print(traj[0]["prompt"][:200])
```

### 5.2 转为训练格式

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

def to_features(rec, max_len=8192):
    p_ids = tokenizer(rec["prompt"], add_special_tokens=False)["input_ids"]
    r_ids = tokenizer(rec["response"] + tokenizer.eos_token,
                      add_special_tokens=False)["input_ids"]
    if len(p_ids) + len(r_ids) > max_len:
        r_ids = r_ids[:max_len - len(p_ids)]
    return {
        "input_ids": p_ids + r_ids,
        "labels": [-100] * len(p_ids) + r_ids,   # 只训练 response
    }
```

### 5.3 训练（参考配置）

```bash
python train_selfdist.py \
    --traj final_train.json \
    --epochs 3 --lr 1.5e-5 --bs 1 --accum 4 --max_len 8192
```

LoRA 配置：`r=16, alpha=32, dropout=0.0`，
target_modules = `[q,k,v,o,gate,up,down_proj]`

---

## 六、数据集的实验价值

### 6.1 这是"负结果"最丰富的数据集之一

基于本数据集，我们做了 **13 个权重配置实验，12 个失败**：

| 方法 | 数据量 | GPQA 准确率 |
|---|---|---|
| base（无训练） | — | 65.00% |
| 老师格式 ORPO（日常场景） | 1200 | 50.00% |
| 老师格式 ORPO（STEM+难度） | 2100 | 32.50% |
| 自蒸馏（base 采样） | 84 | 57.50% |
| 教师混合（base 采样） | 136 | 65.00% |
| **教师混合（harness 轨迹）** | **109** | **80.00%** ✅ |

**失败模式的共同点**：纯老师格式数据会让模型学到"表面格式"，
同时把推理能力压缩掉（平均 token 从 8192 掉到 819-1055）。

### 6.2 为什么混合 harness + 教师有效

| 来源 | 贡献 |
|---|---|
| harness 轨迹（69） | 来自**改进后的 prompt**，结构规范、未作答率低 |
| 教师解法（40） | 提供**原本不会的题**的正确推理 |

单用任一来源都不行：纯 harness 样本太少且无新知识；纯教师格式冲突。

---

## 七、重要提醒

### 7.1 这份数据不能直接用于提升模型知识

harness 轨迹仍由**同一个 Qwen3-8B** 生成。它记录的是"这个模型怎么解题"，
不包含模型原本不知道的知识。

要在 GPQA 上真正突破，需要**注入新知识**（RAG / 更大底座 / 外部数据）。

### 7.2 held-out 已隔离，但请勿二次污染

如果你用本数据集做新实验，请继续使用相同的 40 题 held-out（见 `eval_results/`），
否则结果无法与我们的基线比较。

---

## 八、引用

```bibtex
@article{hebbar2026sia,
  title   = {SIA: Self Improving AI with Harness \& Weight Updates},
  author  = {Hebbar, Prannay and Manawat, Yogendra and Verboomen, Samuel and others},
  journal = {arXiv preprint arXiv:2605.27276},
  year    = {2026}
}
```

---

*生成于 2026-11-18 · 训练硬件 NVIDIA A800-80GB · 教师模型 deepseek-v4-pro*
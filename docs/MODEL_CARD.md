# SIA-Qwen3-8B-GPQA-LoRA

> 复现 [SIA: Self Improving AI with Harness & Weight Updates](https://arxiv.org/abs/2605.27276)
> 的 **Harness + 权重更新串联**实验，在 GPQA 上得到的 LoRA adapter 与完整实验记录。

本仓库包含：
- **LoRA adapter**（基于 Qwen3-8B，r=16，仅 0.53% 参数）
- **完整的 harness 轨迹数据**（SIA 自我改进产出的 198 条真实推理轨迹）
- **训练脚本**与**评测结果**
- **全部负结果记录**（13 个权重配置，12 个失败）

---

## 一、核心结果

### 1.1 一句话结论

> **串联（harness 轨迹 → 权重更新）在 `disciplined` prompt 下达到 80.00%（32/40），
> 相比严格重测的 base 62.50%（25/40）提升 +17.5pt。**
>
> **McNemar 配对检验：11 题由错转对、4 题由对转错，χ²=2.400，p=0.1213。**
> 按常规标准（p<0.05）**统计上尚不显著**，需要在更大样本上验证。

### 1.2 评测结果（GPQA 40 题严格 held-out）

| 配置 | 准确率 | null_rate | 平均 token | 控制标签使用率 |
|---|---|---|---|---|
| base Qwen3-8B + `cot` | 65.00%（26/40） | 2.5% | 8192 | 0% |
| **base Qwen3-8B + `disciplined`（严格重测）** | **62.50%（25/40）** | 5.0% | 5010 | — |
| 本模型 + `cot` | 62.50%（25/40） | 7.5% | 7843 | 0% |
| **本模型 + `disciplined`** | **80.00%（32/40）** | **0.0%** | **6215** | **92.5%** |

### 1.3 逐题验证（不是解析器误差）

同一权重下，`disciplined` 相比 `cot`：

- **修正 9 题**（question_id: 180, 2, 187, 167, 60, 8, 9, 190, 7）
- **退化 2 题**（question_id: 155, 130）
- 净 +7 题

`disciplined` 的 8 道错题是**真实推理错误**（pred ≠ gold），不是解析失败。

### 1.4 关键机制：条件激活，而非内化

| prompt | 标签使用率 | 准确率 |
|---|---|---|
| `cot`（无标签指令） | **0%** | 62.50%（训练效果消失） |
| `disciplined`（明确要求 `<TEMP_LOW>/<TEMP_MID>/<TEMP_HIGH>`） | **92.5%** | **80.00%**（训练效果激活） |

**这是本实验最重要的发现：**

> 训练没有"内化"能力，而是"**条件激活**"了结构化推理模式。
> 必须有 prompt 触发，学到的结构化推理才能生效。

这与 D-CoT 论文（arXiv 2602.21786）主张的"内化后不依赖标签"不同。
我们的结果表明：**prompt 触发是必要条件**。

这也解释了为什么之前的 D-CoT 实验失败（纯老师格式数据 → 只学到表面格式），
而本次成功（harness 轨迹 + 高质量教师解法 + prompt 触发三者配合）。

---

## 二、方法：Harness + 权重更新串联

### 2.1 整体流程

```
┌──────────────────────────────────────────────────────────┐
│ 阶段 1: Harness（SIA 自我改进，改代码/prompt）              │
│   meta-agent: deepseek-v4-pro（经 LiteLLM 中转站）        │
│   target-agent: 本地 Qwen3-8B                             │
│   → Gen 1 产出 198 条真实轨迹（87 正确）                  │
│   → 未作答从早期 agent 的 190 题降至 16 题                 │
├──────────────────────────────────────────────────────────┤
│ 阶段 2: 轨迹提取（严格排除 held-out 40 题）                │
│   → 69 条 Gen 1 正确轨迹                                  │
│   → + 40 条教师解法（deepseek-v4-pro，对失败题）           │
│   → 合计 109 条训练样本                                   │
├──────────────────────────────────────────────────────────┤
│ 阶段 3: LoRA SFT（拒绝采样自蒸馏）                         │
│   → r=16, alpha=32, 7 target modules                     │
│   → 43.6M 可训练参数（0.53%）                             │
│   → 3 epochs, 81 步, lr=1.5e-5                           │
├──────────────────────────────────────────────────────────┤
│ 阶段 4: 评测（40 题 held-out，双 prompt 对照）             │
└──────────────────────────────────────────────────────────┘
```

### 2.2 Harness 阶段结果

| 代 | 正确 | 错误 | 未作答 | evaluator 口径 | **实际准确率(正确/198)** |
|---|---|---|---|---|---|
| **Gen 1** | 87 | 95 | 16 | 47.80% | **43.94%** |
| Gen 2 | 36 | 15 | **147** | 70.59% | **18.18%** |

**学科分布（Gen 1）**：Physics 57.0%（49/86）、Biology 57.9%（11/19）、**Chemistry 29.0%（27/93）**。

> ⚠️ **evaluator 口径陷阱**：SIA 的 GPQA evaluator 用 `accuracy = correct/(correct+incorrect)`，
> 未作答题不计入分母。Gen 2 显示的 70.59% 极具误导性——实际只有 18.18%。
> 必须同时报告 `correct/198`。

### 2.3 Gen 2 的负优化（有价值的失败案例）

feedback agent 自主诊断出了正确根因（*"agent 要求严格 JSON 输出，但 Qwen3-8B 的思考链某条超过
31,000 字符，8192 tokens 全用在思考上，来不及输出 JSON"*），并提出三项修复：

1. 禁用 `enable_thinking`
2. 强制 JSON 模式 + 安全回退
3. **引入续写机制**（`CONTINUATION_MAX_TOKENS`）

**但续写机制在这套硬件上是负优化**：每个被截断的请求变成 2+ 次调用，
速率从 **1.8 题/分钟跌到 0.1 题/分钟**，最终 **147/198 题未作答**。

> **洞察**：feedback agent 的推理是合理的，但它无法预判自己的改进在资源约束下会适得其反。
> **SIA 缺少一个"改进效果验证"环节**——每代改动应先小样本验证再全量跑。

---

## 三、数据集说明

### 3.1 文件清单

| 文件 | 条数 | 说明 |
|---|---|---|
| `data/selfdist/harness_traj.json` | 69 | SIA harness Gen 1 产出的正确轨迹（已排除 held-out） |
| `data/selfdist/teacher_solutions.json` | 52 | `deepseek-v4-pro` 对学生失败题的独立解法（仅保留答案正确的） |
| `data/selfdist/final_train.json` | **109** | **最终训练集**（69 harness + 40 教师，去重后） |
| `data/selfdist/mixed_train.json` | 136 | 早期版本（84 base 采样 + 52 教师） |
| `data/out/final_harness_eval.json` | — | 本模型的评测结果（含逐题详情） |

### 3.2 数据格式

```json
{
  "id": 7,
  "prompt": "Question: ...\n\nOptions:\nA. ...\nB. ...\n\nThink step by step. End with 'Answer: X'...",
  "response": "Think carefully... <TEMP_LOW>...</TEMP_LOW> ... Answer: B",
  "source": "harness"   // 或 "teacher"
}
```

### 3.3 数据来源的关键设计

**为什么混合两种来源？**

| 来源 | 作用 |
|---|---|
| harness 轨迹 | 来自**改进后的 prompt**，结构更规范、未作答率低 |
| 教师解法 | 提供**学生原本不会的题**的正确推理（知识补充） |

单用任一来源都不行：
- 纯 harness（69 条）：样本太少，且仍是同一模型产出，无新知识
- 纯教师（52 条）：格式与模型原生风格冲突（早前 D-CoT 实验验证过，准确率掉到 32.50%）

### 3.4 数据隔离（防泄漏）

- GPQA 198 题中，**40 题作为 held-out**，全程不参与任何训练
- 提取脚本严格排除：`stats: {total: 198, heldout: 39, correct: 69, incorrect: 74, empty: 16}`
- 教师解法同样排除 held-out 题

---

## 四、训练配置

| 项 | 值 |
|---|---|
| 底座模型 | [Qwen/Qwen3-8B](https://www.modelscope.cn/models/Qwen/Qwen3-8B) |
| 方法 | LoRA SFT（拒绝采样自蒸馏，ReST 式） |
| LoRA r / alpha | 16 / 32 |
| LoRA dropout | 0.0 |
| target_modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| 可训练参数 | 43,646,976（占总参数 0.53%） |
| epochs | 3 |
| batch size | 1（梯度累积 4） |
| learning rate | 1.5e-5 |
| max_seq_len | 8192 |
| 优化器 | paged_adamw_8bit |
| 训练步数 | 81 |
| 训练时长 | 6 分 36 秒（A800-80GB） |
| 最终 loss | 0.10–0.16 |

---

## 五、使用方法

### 5.1 加载模型

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B", torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(base, "modelzhang/sia-qwen3-8b-gpqa-lora")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
model.eval()
```

### 5.2 推理（必须使用 disciplined prompt）

> ⚠️ **关键**：本模型的效果**依赖 `disciplined` prompt 触发**。
> 用普通 `cot` prompt 会退化到 62.50%（低于 base 的 65.00%）。

```python
DISCIPLINED_PROMPT = """Question: {question}

Options:
{options}

Use a disciplined, bounded reasoning process with explicit control tags:
1) <TEMP_LOW> list the given quantities / facts and what is being asked (no speculation)
2) <TEMP_MID> do the single needed calculation or recall the needed fact
3) <TEMP_HIGH> ONLY if a second approach is genuinely needed, otherwise skip
Do NOT loop or restate. Hard limit: at most 8 short lines total.
Then output exactly one final line: 'Answer: X' where X is one of A, B, C, D."""

messages = [{"role": "user", "content": DISCIPLINED_PROMPT.format(...)}]
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
enc = tokenizer(text, return_tensors="pt").to(model.device)

with torch.inference_mode():
    out = model.generate(**enc, max_new_tokens=8192, do_sample=False,
                         pad_token_id=tokenizer.pad_token_id)

response = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
```

### 5.3 答案解析

```python
import re

def extract_answer(text):
    """三级回退解析：boxed → Answer: X → 最后一个独立字母"""
    if not text:
        return None
    m = re.findall(r"\\boxed\{([ABCD])\}", text)
    if m:
        return m[-1]
    m = re.findall(r"answer\s*[:：]\s*\(?([ABCD])\)?", text, re.I)
    if m:
        return m[-1].upper()
    m = re.findall(r"\b([ABCD])\b", text)
    return m[-1].upper() if m else None
```

---

## 六、复现步骤

### 6.1 环境

```bash
# 训练环境
torch==2.5.1+cu124
transformers==4.51.3
peft==0.13.2
trl==0.16.1
datasets==3.1.0
```

> `transformers` 必须 ≥ 4.51（Qwen3 架构支持）

### 6.2 完整流程

```bash
# 1. SIA harness 运行（需 SIA + 本地模型服务）
#    产出 runs/run_N/gen_1/agent_execution/*.json
python -m sia run --task gpqa --max_gen 3 --focus harness

# 2. 提取轨迹（自动排除 held-out）
python extract_harness_traj.py

# 3. 构建训练集
python build_final.py    # → final_train.json (109 条)

# 4. 训练
python train_selfdist.py \
    --traj selfdist/final_train.json \
    --out out/final_harness \
    --epochs 3 --lr 1.5e-5 --bs 1 --accum 4 --max_len 8192

# 5. 评测（双 prompt 对照）
python eval_final.py out/final_harness/final_adapter out/eval.json 40
```

### 6.3 评测脚本关键点

- **必须**用 `padding_side="left"`（否则批量生成结果错误）
- **必须**用 `do_sample=False`（确定性评测）
- **必须**同时报告 `correct/total`（不能只报 evaluator 的 accuracy）

---

## 七、诚实的局限说明

### 7.1 统计显著性不足

80.00% vs 62.50%（n=40）：
- **McNemar 配对检验：11 题由错转对、4 题由对转错，χ² = 2.400，p = 0.1213**
- 按常规标准（p<0.05）**尚未达到显著**
- 粗略估算：维持此效应量要达到 p<0.05 需 n≈100–120 题
- GPQA 本身只有 198 题（40 题 held-out），扩大样本需要换评测集

> 这是一个**强烈提示但未确认**的结果，不应作为"已验证的突破"引用。

### 7.2 只在一个任务上有效

本模型只在 **GPQA** 上验证。迁移到其他任务（如 lawbench、MMLU-Pro）未测试。

### 7.3 13 个配置中 12 个失败

本次实验共尝试 13 个权重配置，只有这 1 个突破 base：

| 方法 | 数据量 | GPQA 准确率 |
|---|---|---|
| 老师格式 ORPO（日常场景） | 1200 | 50.00% |
| 老师格式 ORPO（STEM+难度） | 2100 | 32.50% |
| 自蒸馏（base 采样） | 84 | 57.50% |
| 教师混合（base 采样） | 136 | 65.00% |
| 串联 + cot | 109 | 62.50% |
| **串联 + disciplined** | **109** | **80.00%** ✅ |

**失败模式已记录**：纯老师格式数据会让模型学到"表面格式"而牺牲推理能力（token 从 8192 压到 819）。

### 7.4 为什么不能用它突破更大模型

**核心限制**：harness 产出的轨迹仍由同一个 Qwen3-8B 生成。
prompt 能改变它"怎么组织答案"，**不能改变它"知道什么"**。

要在 GPQA 上进一步突破，需要：
- **检索增强（RAG）**：唯一能真正注入新知识的手段
- **更大底座**（32B+）
- **外部数据集**

单纯的自我蒸馏已触及能力边界。

---

## 八、相关工作

- **SIA** (arXiv 2605.27276)：本工作的复现目标，W+H 双更新框架
- **D-CoT** (arXiv 2602.21786)：控制标签抑制 overthinking。本工作发现其"内化"主张需要修正——**prompt 触发是必要的**
- **ReST / 拒绝采样自蒸馏**：本工作采用的核心训练范式

---

## 九、引用

```bibtex
@article{hebbar2026sia,
  title   = {SIA: Self Improving AI with Harness \& Weight Updates},
  author  = {Hebbar, Prannay and Manawat, Yogendra and Verboomen, Samuel and others},
  journal = {arXiv preprint arXiv:2605.27276},
  year    = {2026}
}
```

---

## 十、致谢与工具

- 训练硬件：NVIDIA A800-SXM4-80GB
- 教师模型：`deepseek-v4-pro`（经 LiteLLM 中转站）
- meta/feedback agent：`deepseek-v4-pro`（pydantic-ai impl）

---

*最后更新：2026-11-18*
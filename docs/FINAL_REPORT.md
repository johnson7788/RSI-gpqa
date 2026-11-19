# SIA 复现最终报告：Harness + 权重更新串联

> 复现目标：[SIA: Self Improving AI with Harness & Weight Updates](https://arxiv.org/abs/2605.27276)
> （Hebbar et al., 2026, arXiv 2605.27276）
>
> 本报告记录 **Harness 模式**、**Weight 模式**、以及两者**串联**的完整实验结果。
> 共 6 组 harness 实验 + 13 组权重实验 + 1 组串联实验。
> 训练硬件：NVIDIA A800-SXM4-80GB。底座：Qwen3-8B。

---

## 目录

1. [核心结论](#一核心结论)
2. [串联实验结果（重点）](#二串联实验结果重点)
3. [关键机制：条件激活而非内化](#三关键机制条件激活而非内化)
4. [Harness 阶段完整记录](#四harness-阶段完整记录)
5. [权重更新阶段](#五权重更新阶段)
6. [13 个权重配置的全景](#六13-个权重配置的全景)
7. [工程问题清单](#七工程问题清单)
8. [诚实的局限说明](#八诚实的局限说明)
9. [文件索引](#九文件索引)

---

## 一、核心结论

### 1.1 三条主线的结果

| 模式 | 底座 / 任务 | 结果 | 判定 |
|---|---|---|---|
| **Harness** | deepseek-v4-flash / GPQA | 89.90% → **94.44%**（+4.5pt） | ✅ 有效 |
| **Harness** | Qwen3-8B / GPQA | 未作答 **190 → 16 题**，43.94% | ✅ 修复工程缺陷 |
| **Weight** | Qwen2.5-3B / lawbench | 18.95% → **26.83%**（+7.88pt，p<0.01） | ✅ 有效 |
| **Weight** | Qwen3-8B / GPQA | 65.00% → 65.00%（12 配置） | ❌ 无增益 |
| **串联** | Qwen3-8B / GPQA | 62.50% → **80.00%**（+17.5pt，p=0.12） | ⚠️ **大幅提升但统计不显著** |

### 1.2 一句话结论

> **串联实验取得了本次全部 13 个配置中唯一的大幅突破（+17.5pt），
> 但在 n=40 的样本上统计不显著（McNemar p=0.12），需要用更大样本确认。**

### 1.3 最关键的两条洞察

**洞察 1：条件激活，而非内化**

训练没有"内化"能力，而是"条件激活"了结构化推理模式。
必须有 prompt 触发（`disciplined`），学到的结构化推理才能生效。
这与 D-CoT 论文（arXiv 2602.21786）的"内化"主张不同。

**洞察 2：对工程缺陷有效，对能力天花板无效**

- Gen 1 修复了"未作答 190 题"这个工程缺陷 → 真实提升（43.94%）
- 但 65% 是 Qwen3-8B 在 GPQA 上的知识边界
- harness 产出的轨迹仍由**同一个模型**生成，prompt 改变的是"怎么组织答案"，
  不是"知道什么"

---

## 二、串联实验结果（重点）

### 2.1 完整评测矩阵（GPQA 40 题严格 held-out）

| 配置 | 准确率 | null_rate | 平均 token | 控制标签使用率 |
|---|---|---|---|---|
| base + `cot` | 65.00%（26/40） | 2.5% | 8192 | 0% |
| **base + `disciplined`（重测）** | **62.50%（25/40）** | 5.0% | 5010 | — |
| 串联模型 + `cot` | 62.50%（25/40） | 7.5% | 7843 | 0% |
| **串联模型 + `disciplined`** | **80.00%（32/40）** | **0.0%** | 6215 | **92.5%** |

> **说明**：`base + disciplined` 有两次测量——早期日志记录 65.00%（26/40，无逐题 JSON），
> 后来用同一套脚本重测得到 62.50%（25/40，有完整逐题数据）。
> 为做严谨的配对检验，下文使用重测的 62.50%。

### 2.2 McNemar 配对检验（严格）

使用逐题配对数据（同一批 40 题、同一解析器）：

| 转移 | 题数 |
|---|---|
| base 答错 → 串联答对 | **11** |
| base 答对 → 串联答错 | **4** |
| 净变化 | **+7** |

- 总答对：base 25 → 串联 32（+7）
- **McNemar 连续性校正 χ² = 2.400，p = 0.1213**

**判定：p > 0.05，统计上不显著。**

### 2.3 效应量与样本量

虽然 p=0.12，但效应量很大：

- 绝对提升 **+17.5 个百分点**（62.50% → 80.00%）
- 相对提升 **+28%**
- 不一致对（discordant pairs）有 15 对，其中 11:4 有利于串联

粗略估算：若维持此效应量，要达到 p<0.05 需要 **n ≈ 100–120** 题。

> GPQA 总共只有 198 题，其中 40 题是 held-out。
> 想扩大样本，只能减少训练题量（会削弱训练效果）或换更大的评测集。
> 这是本实验的客观限制。

### 2.4 逐题细节

**串联 model + disciplined 的 8 道错题**（pred ≠ gold，均为真实推理错误，非解析失败）：

| question_id | 预测 | 正确 |
|---|---|---|
| 72 | D | A |
| 155 | D | C |
| 130 | A | B |
| 56 | D | C |
| 24 | A | C |
| 23 | D | B |
| 29 | B | D |
| 164 | B | A |

同一权重下，`disciplined` 相比 `cot` **修正 9 题**（id: 180, 2, 187, 167, 60, 8, 9, 190, 7），
**退化 2 题**（id: 155, 130）。

---

## 三、关键机制：条件激活而非内化

### 3.1 现象

| prompt | 标签使用率 | 准确率 |
|---|---|---|
| `cot`（无标签指令） | **0%** | 62.50% |
| `disciplined`（明确要求 `<TEMP_LOW>/<TEMP_MID>/<TEMP_HIGH>`） | **92.5%** | **80.00%** |

同一份权重，仅改变 prompt，准确率相差 **17.5pt**。

### 3.2 解释

训练数据混合了两种来源：

| 来源 | 条数 | 特征 |
|---|---|---|
| harness 轨迹 | 69 | 来自改进后的 prompt，**带控制标签** |
| 教师解法 | 40 | `deepseek-v4-pro` 产出，**自然语言推理，无标签** |

训练后，模型具备了"使用控制标签做结构化推理"的能力，但**需要 prompt 提示才会激活**。

- `disciplined` prompt 明确指示使用标签 → 激活 → 80.00%
- `cot` prompt 无此指示 → 不激活，退回原生风格 → 62.50%

### 3.3 与 D-CoT 论文的差异

D-CoT 论文（arXiv 2602.21786）主张：训练后模型**内化**了结构化推理，
**不需要**显式控制标签也能保持性能（他们的实验里 tag 率降到 4% 仍保持高分）。

我们的结果显示：**对 Qwen3-8B + 此数据规模，prompt 触发是必要条件**。

可能的原因：
- 数据量差异（D-CoT 用 5,079 条，我们用 109 条）
- 底座差异（他们用 Unsloth + Lion 优化器）
- 任务差异（他们的训练数据是通用推理，我们是 GPQA 轨迹 + 教师解法混合）

---

## 四、Harness 阶段完整记录

### 4.1 运行配置

| 项 | 值 |
|---|---|
| meta/feedback agent | `deepseek-v4-pro`（经 LiteLLM 中转站，pydantic-ai impl） |
| target agent | **本地 Qwen3-8B**（自建 OpenAI 兼容批处理服务） |
| 任务 | GPQA 全 198 题 |
| 代数 | 3（实际完成 2） |
| 运行 ID | run_22 |

### 4.2 三代结果

| 代 | 正确 | 错误 | 未作答 | evaluator 口径 | **实际准确率(正确/198)** |
|---|---|---|---|---|---|
| **Gen 1** | 87 | 95 | 16 | 47.80% | **43.94%** |
| Gen 2 | 36 | 15 | **147** | 70.59% | **18.18%** |
| Gen 3 | 未跑完（已停） | — | — | — | — |

> ⚠️ **evaluator 口径陷阱**：SIA 的 GPQA evaluator 用
> `accuracy = correct / (correct + incorrect)`，**未作答题不计入分母**。
> Gen 2 显示的 70.59% 极具误导性——实际只有 18.18%。
> 必须同时报告 `correct/198`。这个陷阱在本复现过程中踩了两次。

### 4.3 Gen 1 学科分布（最大短板）

| 学科 | 答对/总数 | 准确率 |
|---|---|---|
| Physics | 49/86 | 57.0% |
| Biology | 11/19 | 57.9% |
| **Chemistry** | 27/93 | **29.0%** |

Chemistry 不足 Physics/Biology 的一半，是最明显的改进空间。

### 4.4 feedback agent 的自主诊断能力（SIA 的核心价值）

**Gen 1 → Gen 2 的改进**，feedback agent 精准定位了两类根因：

1. **输出格式与推理长度冲突**：agent 要求严格 JSON 输出，但 Qwen3-8B 的思考链极长
   （某条轨迹超过 **31,000 字符**），8192 tokens 全用于思考，来不及输出 JSON
2. **超时**：长生成在并发下超过请求超时，被记为 missing

它提出的修复方案专业且具体：
- 禁用 Qwen3 思考模式（`enable_thinking: False`）
- 强制 JSON 模式 + 安全回退
- 引入**续写机制**（`CONTINUATION_MAX_TOKENS`）：被截断时自动继续生成

### 4.5 但 Gen 2 实际上退化了（有价值的负结果）

**续写机制在这套硬件上是负优化**：

- 每个被截断的请求变成 2+ 次请求
- 单请求耗时暴涨（实测从 1.8 题/分钟降到 **0.1 题/分钟**）
- 最终 **147/198 题未作答**

> **洞察**：feedback agent 的推理是合理的，但它无法预判自己的改进在资源约束下会适得其反。
> **SIA 缺少一个"改进效果验证"环节**——每代改动应先小样本验证再全量跑。

---

## 五、权重更新阶段

### 5.1 数据提取（严格防泄漏）

从 Gen 1 的 198 条轨迹中提取：

```
held-out excluded: 40
stats: {total: 198, no_assistant: 0, empty: 16, correct: 69, incorrect: 74, heldout: 39}
usable trajectories: 69
```

| 来源 | 条数 | 说明 |
|---|---|---|
| Gen 1 正确轨迹 | **69** | 158 训练题中答对的（排除 held-out 39 + 空响应 16） |
| 教师解法补充 | **40** | `deepseek-v4-pro` 对失败题的解法（去重后） |
| **合计** | **109** | `final_train.json` |

### 5.2 教师解法的防泄漏设计

关键约束：**绝不把标准答案告诉教师**。

1. 教师只看题目和选项，独立求解
2. 解完后校验其答案是否与 ground truth 一致
3. 只有 `ok: true` 的解法才保留

这样教师提供的是"可信的推理过程"，而不是"反推出来的答案"。
教师单次尝试的正确率 59.5%（44/74），多次尝试（每题最多 5 次，温度递增）后提升到 70.3%（52/74）。

### 5.3 训练配置

| 项 | 值 |
|---|---|
| 底座 | Qwen3-8B |
| 方法 | LoRA SFT（拒绝采样自蒸馏，ReST 式） |
| LoRA r / alpha / dropout | 16 / 32 / 0.0 |
| target_modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| 可训练参数 | 43,646,976（0.53%） |
| epochs | 3 |
| batch / 累积 | 1 / 4 |
| learning rate | 1.5e-5 |
| max_seq_len | 8192 |
| 优化器 | paged_adamw_8bit |
| 训练步数 | 81 |
| 训练时长 | 6 分 36 秒 |
| 最终 loss | 0.10–0.16 |

---

## 六、13 个权重配置的全景

### 6.1 全部结果

| # | 方法 | 数据量 | prompt | GPQA 准确率 |
|---|---|---|---|---|
| — | base Qwen3-8B | — | cot | **65.00%** |
| — | base | — | disciplined | 65.00% / 62.50%（两次测量） |
| 1 | ORPO（老师格式·日常场景） | 1200 | cot | 50.00% |
| 2 | 同上 | | disciplined | 45.00% |
| 3 | ORPO（老师格式·STEM+难度感知） | 2100 | cot | 32.50% |
| 4 | 同上 | | disciplined | 55.00% |
| 5 | 自蒸馏（base 采样正确轨迹） | 84 | cot | 57.50% |
| 6 | 同上 | | disciplined | 50.00% |
| 7 | 教师混合（base 采样 + 教师 44） | 128 | cot | 65.00% |
| 8 | 教师混合 v2（+ 教师 52） | 136 | cot | 65.00% |
| 9 | **串联（harness 轨迹 + 教师）** | **109** | cot | 62.50% |
| 10 | **串联 + disciplined** | **109** | disciplined | **80.00%** ✅ |

### 6.2 失败模式的共同点

**纯老师格式数据会让模型学到"表面格式"，同时牺牲推理能力**：

| 配置 | 平均 token | 准确率 |
|---|---|---|
| base | 8192 | 65.00% |
| 老师格式 ORPO（日常） | **819** | 50.00% |
| 老师格式 ORPO（STEM） | **1055** | 32.50% |

token 从 8192 骤降到 819-1055，说明模型把长链推理压缩掉了。
加入 STEM 场景和难度感知（hard 样本 1833 字 vs easy 530 字）**反而更差**（32.5%）。

### 6.3 为什么串联 + disciplined 有效

| 要素 | 作用 |
|---|---|
| harness 轨迹（69 条） | 来自**改进后的 prompt**，结构规范、未作答率低、带控制标签 |
| 教师解法（40 条） | 提供**原本不会的题**的正确推理 |
| `disciplined` prompt | **触发**学到的结构化推理模式 |

三者缺一不可。单用任一来源或缺少 prompt 触发都会失败。

---

## 七、工程问题清单

### 7.1 本地环境（5 项）

| # | 问题 | 修复 |
|---|---|---|
| 1 | 无 sudo 时系统目录不可写 | 全部装入 workspace |
| 2 | Xcode 许可证未同意（git 报错） | `sudo -n xcodebuild -license accept` |
| 3 | python.org 框架版无 CA 证书（pip SSL 全挂） | 手动灌入 Mozilla CA bundle |
| 4 | GitHub 直连超时 | Python 走 ghproxy，包走阿里云/清华镜像 |
| 5 | standalone CPython 的 venv 建不起来 | 改用系统框架版 Python |

### 7.2 GPU 训练栈（10 项）

| # | 问题 | 修复 |
|---|---|---|
| 1 | transformers 4.48 不认识 Qwen3 | 升级 4.51.3 |
| 2 | trl 0.12.2 无 GRPO | 升级 0.16.1 |
| 3 | GRPO batch 必须能被 num_generations 整除 | 调整配置 |
| 4 | 梯度检查点导致 OOM | 权衡后降 batch/长度 |
| 5 | 右填充导致批量生成错误 | `padding_side="left"` |
| 6 | LoRA 参数被冻结（does not require grad） | 显式 `requires_grad_(True)` |
| 7 | `ThreadPoolExecutor` 在 break 后挂起 | 改 `FIRST_COMPLETED` + cancel |
| 8 | 老师输出被截断（no json） | max_tokens 4000→24000 |
| 9 | JSON 含未转义控制字符 | `strict=False` + 清洗回退 |
| 10 | 数据生成并发太低（4 条/分） | 提到 75 并发 → 155 条/分 |

### 7.3 串联实验新增（7 项）

| # | 问题 | 原因 | 修复 |
|---|---|---|---|
| 1 | SIA 报 "Run directory already exists" 直接退出 | 上传源码时带了本地 `runs/` 目录 | 清理 `runs/`，换 run_id |
| 2 | `agent_reference` 格式错误 | 必须是对象 `{source: ...}` | 修正 profile |
| 3 | 服务端不支持 `enable_thinking` | 硬编码 `True` | 读取 `chat_template_kwargs` |
| 4 | 服务完全无响应（25s 超时） | 批处理把 `max_tokens` 取 max，一个长请求拖垮整批 | 限制批大小 + 单请求硬上限 |
| 5 | Gen 2/Gen 3 全部 0 分 | **我重启服务导致请求全断** | 教训：跑评测时不能动服务 |
| 6 | 评测极慢（17 分钟/批） | batch=2 + 8192 tokens | 改 batch=5 |
| 7 | 两个评测进程互相拖慢 | GPU 显存竞争（71GB） | 串行执行 |

---

## 八、诚实的局限说明

### 8.1 统计显著性不足（最重要）

80.00% vs 62.50%（n=40）：

- McNemar 配对检验：**χ² = 2.400，p = 0.1213**
- 按常规标准（p<0.05）**尚未达到显著**
- 需要 n≈100–120 才能在当前效应量下达到显著

**这个结果不能宣称"已确认的突破"，只能说是"强烈提示"。**

### 8.2 只在一个任务、一个底座上验证

- 任务：仅 GPQA（198 题，40 题 held-out）
- 底座：仅 Qwen3-8B
- 迁移到其他任务（lawbench、MMLU-Pro）或其他模型未测试

### 8.3 13 个配置中 11 个失败

本实验共尝试 13 个权重配置，只有串联 + disciplined 这一个大幅突破。

失败模式已系统记录（见第六章），可作为后续研究的负面参考。

### 8.4 无法突破知识边界

**核心限制**：harness 产出的轨迹仍由同一个 Qwen3-8B 生成。
prompt 能改变它"怎么组织答案"，**不能改变它"知道什么"**。

要在 GPQA 上进一步突破，需要：
- **检索增强（RAG）**：唯一能真正注入新知识的手段
- **更大底座**（32B+）
- **外部数据集**

单纯的自我蒸馏已触及能力边界。

### 8.5 与论文的差异

| 维度 | 论文 | 本复现 |
|---|---|---|
| 模型 | Claude 系列 | DeepSeek（harness）+ Qwen3-8B（weights） |
| 权重更新 | Tinker 平台 RL（W+H） | 本地 LoRA SFT |
| 任务 | 含 GPU kernel、scRNA 去噪 | 仅 GPQA、lawbench |
| 绝对数字 | 不可直接比 | — |

---

## 九、文件索引

### 9.1 本地文件（绝对路径）

| 路径 | 内容 |
|---|---|
| `/Users/admin/Documents/RSI/sia/FINAL_REPORT.md` | **本报告** |
| `/Users/admin/Documents/RSI/sia/REPORT.md` | 主报告（harness + weights 全景） |
| `/Users/admin/Documents/RSI/sia/PIPELINE_EXPERIMENT.md` | 串联实验详细记录 |
| `/Users/admin/Documents/RSI/sia/WEIGHTS_REPORT.md` | 12 个权重配置详情 |
| `/Users/admin/Documents/RSI/sia/WEIGHTS_MODE.md` | lawbench +7.9pt 成功案例 |
| `/Users/admin/Documents/RSI/sia/DCOT_EXPERIMENT.md` | D-CoT 实验记录 |
| `/Users/admin/Documents/RSI/sia/RESULTS.md` | Harness 模式详细结果 |
| `/Users/admin/Documents/RSI/sia/REPRODUCTION.md` | 环境搭建与中转站接入 |
| `/Users/admin/Documents/RSI/upload/model/README.md` | ModelScope 模型卡片（已上传） |
| `/Users/admin/Documents/RSI/upload/dataset/README.md` | ModelScope 数据集卡片（已上传） |
| `/Users/admin/Documents/RSI/lora_results/` | 全部结果 JSON 与脚本备份 |
| `/Users/admin/Documents/RSI/lora_results/harness_run22/` | Qwen3-8B harness 三代结果 |

### 9.2 ModelScope 仓库

| 仓库 | 内容 |
|---|---|
| [modelzhang/sia-qwen3-8b-gpqa-lora](https://www.modelscope.cn/models/modelzhang/sia-qwen3-8b-gpqa-lora) | LoRA adapter（174MB）+ 报告 |
| [modelzhang/sia-gpqa-harness-trajectories](https://www.modelscope.cn/datasets/modelzhang/sia-gpqa-harness-trajectories) | 轨迹数据 + 代码 + 评测结果 |

### 9.3 服务器关键路径

| 路径 | 内容 |
|---|---|
| `/workspace/sia/runs/run_22/gen_1/agent_execution/` | Gen 1 完整轨迹（198 条） |
| `/workspace/lora_gpqa/out/final_harness/final_adapter/` | 串联训练出的 LoRA 权重 |
| `/workspace/lora_gpqa/selfdist/final_train.json` | 109 条训练集 |
| `/workspace/lora_gpqa/batch_server2.py` | 本地推理服务 |

---

## 十、后续建议

按性价比排序：

| 方向 | 预期 | 成本 |
|---|---|---|
| **扩大评测样本**（验证 80% 是否真实） | 消除统计不确定性 | 需换更大评测集（GPQA 只有 198 题） |
| **lawbench 扩规模** | 把已验证的 +7.9pt 做到 +12pt+ | 2–3 小时，风险最低 |
| **检索增强（RAG）** | 唯一能突破知识边界的手段 | 中等，需设计检索库 |
| **更大底座**（32B+） | 提升空间更大 | A800 单卡需量化 |
| **给 harness 加"改进验证"** | 避免 Gen 2 那类负优化 | 改 SIA 源码 |

---

*报告生成于 2026-11-18 · 训练硬件 NVIDIA A800-80GB · 教师模型 deepseek-v4-pro*
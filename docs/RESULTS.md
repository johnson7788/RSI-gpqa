# SIA 复现结果

> 📖 返回 [文档索引](README.md) · [主 README](../README.md)


实验配置（`runs/run_3`）：

| 项 | 值 |
|---|---|
| 任务 | `gpqa`（198 道 GPQA-diamond 风格选择题） |
| 模式 | **Harness**（默认，只改代码/prompt，不动权重） |
| 代数 | 3 |
| meta / feedback agent | `deepseek-v4-pro`，pydantic-ai impl，经 LiteLLM 中转站 |
| target agent | `deepseek-v4-flash`，经中转站 OpenAI 兼容端点 |
| 温度 | 0 |

## 主结果

| 代 | 答对 | 答错 | 缺失 | 准确率(答对/已作答) | **准确率(答对/198)** | Physics | Chemistry | Biology |
|---|---|---|---|---|---|---|---|---|
| **Gen 1** | 178 | 16 | 4 | 91.75% | **89.90%** | 94.2% (81/86) | 87.1% (81/93) | 84.2% (16/19) |
| **Gen 2** | 180 | 18 | 0 | 90.91% | **90.91%** | 96.5% (83/86) | 87.1% (81/93) | 84.2% (16/19) |
| **Gen 3** | 187 | 11 | 0 | 94.44% | **94.44%** | 97.7% (84/86) | 92.5% (86/93) | 89.5% (17/19) |

**Gen1 → Gen3：答对 178 → 187（+9 题）**，缺失 4 → 0，整体从 89.90% 提升到 **94.44%**。

一代未调优的基线（`runs/run_1`，未调整推理配置）作为对照：
`correct 85 / missing 109`，表面“准确率 95.51%”，但按 198 题计只有 **42.93%**。

## 每代改进内容（由 feedback agent 自主诊断）

### Gen 1 → Gen 2
- **修复 `"" in "ABCD"` 恒为 True 的校验 bug**：空/解析失败的答案被误当作成功，导致该题不会被重试。
- **移除 prompt 中的工具调用诱导**：原 prompt 暗示可读写文件系统，模型会输出 `{"name":"shell","arguments":...}` 这类工具调用而非答案；新 prompt 明确声明无工具、须凭知识作答。
- 用**纠正性重试**替代盲目重发（追加“上次格式无效，请只回复 ...”）。
- 引入带 HTTP 状态码的 `APIError`、指数退避 + 抖动、`json_object` 失败时自动降级为纯文本模式。
- 多候选 JSON 解析（fenced / greedy / 逐 span）、原子写入轨迹文件。

### Gen 2 → Gen 3
- 诊断出 Gen2 的 197/198 条响应里**没有 `reasoning` 字段**——prompt 把推理写成“可选”，模型直接一跳给出答案。
- 改为**强制结构化思维链**：`{"reasoning": "...", "answer": "A"}`，要求逐一评估四个选项、说明干扰项为何错误、复核算术与单位。
- 效果：Chemistry 87.1% → **92.5%**，Biology 84.2% → **89.5%**，Physics 96.5% → **97.7%**。

## 诚实的结论

1. **自我改进机制确实有效且可复现**：三代单调提升（89.90% → 90.91% → 94.44%），每一代的改进点都是 feedback agent 从执行日志中**自主诊断**出来的，而不是我手动提示的。
2. **与论文数字不可直接对比**：论文的 GPQA/MLE-Bench 等结果使用 Claude 等模型及 W+H（权重+harness）双更新；本次是 harness-only、DeepSeek 模型、自有中转站。可比的是**趋势**（逐代上升），不是绝对分值。
3. **两个必须警惕的度量陷阱**（见 `REPRODUCTION.md` 第 4 节）：
   - 推理模型的 `max_tokens` 必须给足，否则难题会在思考阶段被截断。
   - GPQA evaluator 的 accuracy 分母只含“已作答”题目，missing 被静默排除。
4. **未复现的部分**：论文的 GPU kernel（TriMul Triton）与 scRNA 去噪任务不在内置任务中，需自建 `--task_dir`；Weights 模式需 `TINKER_API_KEY` + Modal，本次未涉及。

## 复现命令

```bash
cd ~/Documents/RSI/sia
export SIA_RELAY_API_KEY="sk-..."
./run_repro.sh gpqa 3 3          # 任务 代数 run_id
python3 summarize.py 3           # 汇总逐代结果
../.venv/bin/sia web --runs-dir ./runs    # 可视化面板
```

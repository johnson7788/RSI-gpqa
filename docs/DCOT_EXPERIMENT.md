# D-CoT 式权重重训记录（Qwen3-8B + ORPO）

> 📖 返回 [文档索引](README.md) · [主 README](../README.md)


这是路线 B 的第二阶段：在 lawbench 拒绝采样 SFT（+7.9pt，见 `WEIGHTS_MODE.md`）之后，
尝试复现 D-CoT（arXiv 2602.21786）——用**控制标签 strucutred CoT + ORPO** 抑制小模型的
overthinking。

## 1. 基线（Qwen3-8B，GPQA 40 题 held-out）

| 配置 | 准确率 | Null Rate | 平均 token |
|---|---|---|---|
| base + `cot`（8192 预算） | **65.00%** (26/40) | 2.5% | 8192（顶满） |
| base + `disciplined`（控制标签指令） | **65.00%** (26/40) | 0.0% | 6525（−20%） |

两个重要事实：
1. 早期测的 "20.00% / Null 30%" 是 `max_new_tokens=1536` 太小导致的**测量错误**，不是模型问题。
2. **纯 prompt 的控制标签指令**已能消除 Null Rate 并省 20% token，但不提升准确率。

## 2. v2 尝试：日常场景数据（1200 条）→ 失败

数据：1200 条偏好对，全部为日常办公场景（定价/排班/调试）。训练：LoRA r=16，lr 4e-6，2 epoch。

| 配置 | 准确率 | token | 标签使用率 |
|---|---|---|---|
| base + cot | 65.00% | 8192 | 0% |
| **v2 tuned + cot** | **50.00%** | **819** | **100%** |
| v2 tuned + disciplined | 45.00% | 839 | 100% |

**结论**：权重确实更新了（标签使用率 0→100%，token 降 90%），但**准确率掉 15–20pt**。
根因：训练数据全是短回答场景，模型学成了"对什么都简短回答"，把 GPQA 需要的
长链推理压没了（avg_tok 819 就是证据）。

## 3. v3 尝试：STEM + 难度感知数据（2100 条）

修正两点：
1. **45% STEM 场景**（物理/化学/生物/数学/工程，自造题目，非 GPQA 题库）
2. **难度感知**：每条标注 easy/hard
   - easy：LOW→MID，约 530 字
   - hard：LOW→多段 MID（3–7 段显式计算）→HIGH，约 1833 字
   让模型学"按难度伸缩推理"，而非一律压缩

| 维度 | 数值 |
|---|---|
| 总条数 | 2100（去重后） |
| 难度分布 | hard 1231 / easy 869 |
| STEM 占比 | 935 条（45%） |
| 去污染 | 13-gram 重叠检查，dropped=0 |

训练配置：LoRA r=32（含 lm_head），lr 8e-6，5 epoch，1310 步，max_len 4096，9228 万可训练参数。

> 训练结果见第 4 节（完成后补）。

## 4. 数据生成工程细节

生成器 `gen_dcot_v3.py` 用中转站 `deepseek-v4-pro` 作为老师模型并行生成，踩过：

1. **`max_tokens` 太小导致输出截断**（`no json` 错误）→ 提到 24000
2. **`ThreadPoolExecutor` 在 break 后等待所有 future** 导致挂起 → 改用 `FIRST_COMPLETED` + `cancel_futures`
3. **JSON 中未转义控制字符** → 加 `strict=False` + 清洗回退
4. **并发太低**（14 → 生成速率 4 条/分钟）→ 提到 75 并发后达 **155 条/分钟**
5. **边生成边落盘**（`.partial`），避免进程被杀丢数据

## 5. 评测协议

- 题目：GPQA 40 题 held-out（`test_ids.json`，从未参与训练）
- 解析器：三次尝试（`\boxed{X}` → `Answer: X` → 最后一个独立字母）
- 双 prompt 对照：`cot`（无标签指令）vs `disciplined`（含标签指令）
- **同时报告 accuracy / null_rate / avg_tok / tag_rate**，避免"missing 不进分母"的陷阱

## 文件

```
<训练服务器路径>
├── gen_dcot_v3.py        数据生成（STEM + 难度感知）
├── train_orpo.py         ORPO 训练
├── eval_final.py         双配置评测
├── compare_all.py        三方对比
├── dcot_data/pairs_v3_final.json   2100 条偏好对
└── out/dcot_orpo3/       训练输出
```

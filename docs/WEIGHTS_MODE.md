# 路线 B 复现记录：LoRA 权重更新

> 📖 返回 [文档索引](README.md) · [主 README](../README.md)


与 harness 模式（只改代码/prompt）不同，这里是**真正修改模型参数**。

## 服务器

| 项 | 值 |
|---|---|
| 机器 | uhost-1w9delcv23hk（`YOUR_SERVER_IP:23`） |
| GPU | NVIDIA A800-SXM4-80GB（驱动 595.71.05 / CUDA 13.2） |
| 系统 | Ubuntu 22.04 / Python 3.10 / 236 GB 内存 / 164 GB 空闲磁盘 |
| 环境 | `<训练服务器路径>`：torch 2.5.1+cu124、transformers 4.48.3、peft 0.13.2、trl 0.16.1 |
| 底座模型 | `/model/ModelScope/Qwen/Qwen2.5-3B-Instruct`（另有 7B、Qwen3-4B/8B、Llama-3.1-8B 等） |

> 模型实际在 `/model`（不是 `/models`）。服务器原本无 pip，用 ensurepip 自举后用清华镜像装依赖。

## 关键结论：任务选择决定一切

我先在 **GPQA** 上训练，失败了，原因值得记录：

| 模型 | GPQA 准确率（40 题 held-out） | 随机基线 |
|---|---|---|
| Qwen2.5-3B | 32.50% | ~25% |
| Qwen2.5-7B | **25.00%** | ~25% |

**7B 反而比 3B 差，且双双贴着随机线** —— GPQA 是研究生级别科学题，对 3B/7B 不可学。
GRPO 的组内奖励几乎恒为 0，`reward_std ≈ 0`，没有任何梯度信号。这是**任务-模型能力不匹配**，不是代码问题。

因此改用 **lawbench**（中文罪名分类，191 类，自带 5333 条带标签训练数据）：
- 随机基线 1/191 ≈ 0.5%
- 3B 实测约 19–33%，**远高于随机** → 可学习，奖励稠密

## 训练方法：拒绝采样自蒸馏（ReST）

直接跑 GRPO 时奖励方差仍然很小（多数样本 0.1 基座分），因为模型很少答对。
改用**拒绝采样 + SFT**：

1. 让 base 模型对训练集每题采样 4 次（400 题 × 4 = 1600 个 rollout）
2. 用可验证奖励筛选：**只保留答案与标签逐字一致的轨迹** → 采样到 398 条，去长后 356 条
3. 在这些成功轨迹上做 LoRA SFT（r=16，2993 万可训练参数，150 步）
4. 在**从未参与训练的** 533 条验证集上评测 base 与 tuned 模型

筛选统计：`total=1600, parsed=961, correct=398`。

## 结果（held-out 533 条验证集）

| 模型 | 准确率 | 解析率 | 修正题数 | 退化题数 |
|---|---|---|---|---|
| **Base（Qwen2.5-3B-Instruct）** | 101/533 = **18.95%** | 51.4% | — | — |
| **LoRA 微调后** | 143/533 = **26.83%** | **62.9%** | +63 | −21 |

- **准确率 +7.88 个百分点（相对 +41.6%）**
- 两比例 z 检验 **z = 3.06，p < 0.01，统计显著**
- 净收益 +42 题（修正 63，退化 21），不是靠“多答对几题、又错掉更多”换来的表面提升

## 复现中踩的坑（都已修复）

1. **右填充导致 batch 生成错误** → tokenizer 必须 `padding_side="left"`
2. **LoRA 参数被冻结**（`does not require grad`）→ `get_peft_model` 后显式 `requires_grad_(True)`
3. **GRPO OOM**（80GB 也不够）→ 关闭梯度检查点会爆，保留 + 降 batch/长度
4. **GPQA 训练不可学** → 换任务（最重要的一条）
5. **模型自创罪名**（不在 191 类里）→ prompt 强调逐字复制 + 解析器加最长匹配兜底

## 与论文的关系

论文报告的是 **W+H 双更新**（weights + harness），在 LawBench 上达到 70.1% Top-1。
本次复现的是 **weights 单更新**、Qwen2.5-3B 底座、约 356 条自采样轨迹，与论文条件差异很大，**绝对数字不可直接比较**。
可复现的核心命题是：**用可验证奖励从模型自身成功轨迹中学习，能稳定提升 held-out 表现**（+7.9pt，p<0.01）。

若要更接近论文：增加采样量（当前仅 400 题 × 4 次）、换更大底座（7B/8B）、或接入 Tinker 走论文原路径。

## 远端文件

```
<训练服务器路径>
├── train_rest.py          # 拒绝采样 + SFT + 评测 主流程
├── train_law.py           # lawbench GRPO（备选）
├── train_rl.py            # GPQA GRPO（对照，证明不可学）
├── eval_model.py          # GPQA 评测
├── eval_base_law.py       # lawbench base 评测
└── out/
    ├── base_law_3b.json          # base 533 题结果
    ├── rest_law_3b/eval_result.json  # 微调后结果
    ├── rest_law_3b/val.csv       # 验证集
    └── rest_law_3b/reject_stats.json
```

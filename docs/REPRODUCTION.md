# SIA 论文复现记录（本地）

复现目标：[SIA: Self Improving AI with Harness & Weight Updates](https://arxiv.org/abs/2605.27276) 的 **Harness 模式**自我改进循环。

## 1. 本机环境（如何搭起来的）

无 sudo 时的约束：`/opt`、`/usr/local`、`/Library` 早期不可写，`python3`/`git` 只是 Xcode shim。
后来用户配置 `NOPASSWD: ALL`，完成系统级安装。

| 组件 | 版本 | 位置 |
|---|---|---|
| git | 2.50.1 (Apple Git-155) | `/usr/bin/git` |
| Python | 3.12.10 | `/usr/local/bin/python3` → `/Library/Frameworks/Python.framework/Versions/3.12` |
| SIA | sia-agent 0.6.0 (editable) | `../.venv` |
| pydantic-ai | 2.68.0 | `../.venv` |

关键过程中修复的问题：

1. **Xcode 许可未同意** → `sudo -n xcodebuild -license accept`
2. **python.org 框架版无 CA 证书**（pip SSL 全挂）→ 用 curl 拉 Mozilla CA bundle 放到
   `/Library/Frameworks/Python.framework/Versions/3.12/etc/openssl/cert.pem`
3. **pypi.org 太慢** → `~/.pip/pip.conf` 配阿里云镜像
4. **standalone CPython 建 venv 时 ensurepip 崩溃(SIGABRT)** → 改用系统框架版 Python

## 2. 接入自有中转站

中转站 `https://YOUR_RELAY_HOST` 是 LiteLLM 网关，**同时支持 OpenAI 与 Anthropic 两种协议**，
因此 target agent 与 meta agent 都不需要改 SIA 源码。

```
providers/infox-openai.json    # OpenAI 兼容，给 meta/feedback agent（pydantic-ai impl）
providers/infox-anthropic.json # Anthropic 兼容，给 target agent（参考 agent 用 anthropic SDK 风格）
profiles/infox-meta.json       # agent_impl=pydantic-ai, model=deepseek-v4-pro
profiles/infox-target.json     # model=deepseek-v4-flash, agent_reference=../seeds/gpqa_seed.py
```

`api_key_env` 统一用 `SIA_RELAY_API_KEY`，**不要把 key 写进仓库**。

### 为什么需要自定义 seed

GPQA 任务自带的 `reference_target_agent.py` 硬编码了 Tinker 端点
（`https://tinker.thinkingmachines.dev/...`）和 `TINKER_API_KEY`。
`agent_reference: "default"` 会用这份文件作为 meta-agent 的改进起点，
于是第一代生成器会直接索要 `TINKER_API_KEY` 而失败。

因此 `seeds/gpqa_seed.py` 是同一份逻辑、但端点/模型/凭证改为环境变量驱动的版本：

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `SIA_RELAY_API_KEY` | — | 中转站 key（必填） |
| `TARGET_BASE_URL` | `https://YOUR_RELAY_HOST/v1` | 目标模型端点 |
| `TARGET_MODEL` | `deepseek-v4-flash` | 目标模型 |
| `TARGET_MAX_TOKENS` | `60000` | **关键**：见下方“推理模型陷阱” |
| `TARGET_CONCURRENCY` | `16` | 并发 |
| `TARGET_TIMEOUT` | `600` | 单请求超时（推理题可能耗数分钟） |

## 3. 运行

```bash
cd ~/Documents/RSI/sia
export SIA_RELAY_API_KEY="sk-..."
./run_repro.sh gpqa 3 1     # 任务、代数、run_id
```

产物在 `runs/run_<id>/gen_<n>/`：`target_agent.py`、`improvement.md`、`results.json`、
`agent_execution/`、各类日志。

## 4. 复现中发现的两个陷阱

### 陷阱一：推理模型的 max_tokens

`deepseek-v4-flash` 是推理模型，难题的思考链极长。实测 GPQA 中部分题目需要
**7000–27000 reasoning tokens** 才能输出答案。原参考 agent 的 `max_tokens=1000`
导致大量题在思考阶段被截断、`content` 为空。

### 陷阱二：GPQA evaluator 的 accuracy 定义

`sia/tasks/gpqa/data/public/evaluate.py`：

```python
attempted = results["correct"] + results["incorrect"]
results["accuracy"] = results["correct"] / attempted
```

即 **accuracy = 答对 / 已作答**，`missing` 不计入分母。
第一代曾出现“Correct 85 / Missing 109，Accuracy 95.51%”这种误导性读数。
**对比论文数字时应同时报告 `correct/198`**，否则无法反映实际覆盖率。

## 5. 结果

见 `RESULTS.md`。

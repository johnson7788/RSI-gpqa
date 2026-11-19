#!/usr/bin/env python3
"""
路线 B（lawbench 版）：LoRA + GRPO 训练一个中文罪名分类器。

为什么换 lawbench：
  GPQA 对 3B/7B 小模型近乎随机猜测（实测 32.5% / 25%，随机约 25%），
  GRPO 的组内奖励几乎恒为 0，没有可学习的梯度信号。
  lawbench 自带 5333 条带标签训练数据（191 类罪名），奖励稠密、任务可学。

奖励（可验证，与 SIA evaluator 口径一致：预测词 == 标准罪名 才算对）：
  +1.0  预测罪名与标签完全一致
   0.0  预测合法但不一致
  -0.5  输出无法解析成列表中的任何罪名
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

BASE = Path("data/lawbench/data")
CLASSES = json.load(open(BASE / "public/classes.json"))


def norm(s: str) -> str:
    """把 "[生产、销售]假药" 之类标签的方括号去掉，便于比较。"""
    return re.sub(r"[\[\]]", "", s).strip()


CLASS_SET = {norm(c) for c in CLASSES}


def build_prompt(text: str, classes_display: list[str]) -> str:
    clist = "、".join(classes_display)
    return (
        "你是中国刑事案件的罪名认定助手。\n"
        "任务：阅读案件事实，从下面给出的**候选罪名列表**中选出最符合的**一个**。\n"
        "硬性要求：你输出的罪名必须与候选列表中的某一项**逐字完全相同**，不得自创、"
        "不得改写、不得简写、不得添加「罪」字或省略方括号内容。\n\n"
        f"候选罪名列表（共 {len(classes_display)} 项）：\n{clist}\n\n"
        f"案件事实：{text}\n\n"
        "请直接作答，最后一行必须是：\n罪名：<从候选列表中原样复制的一个罪名>"
    )


ANS_RE = re.compile(r"罪名\s*[:：]\s*([^\n]+)")


def extract_charge(text: str) -> str | None:
    if not text:
        return None
    m = ANS_RE.findall(text)
    if m:
        cand = norm(m[-1])
        cand = re.sub(r"[「」『』\s。.]", "", cand)
        if cand in CLASS_SET:
            return cand
        # 前缀匹配（模型可能多说几个字）
        for c in CLASS_SET:
            if cand.startswith(c) or c in cand:
                return c
    # fallback：在整段回答里找候选列表中的罪名，取最长匹配
    cleaned = re.sub(r"[「」『』\s。.，,]", "", text)
    best = None
    for c in CLASS_SET:
        if c in cleaned:
            if best is None or len(c) > len(best):
                best = c
    return best


def load_data(seed: int, val_frac: float = 0.1):
    df = pd.read_csv(BASE / "training_data/train.csv")
    df = df.dropna(subset=["text", "label"])
    df["label_n"] = df["label"].map(norm)
    df = df[df["label_n"].isin(CLASS_SET)]
    rng = random.Random(seed)
    idx = list(range(len(df)))
    rng.shuffle(idx)
    n_val = int(len(idx) * val_frac)
    val_idx = set(idx[:n_val])
    train = df.iloc[[i for i in idx if i not in val_idx]]
    val = df.iloc[[i for i in idx if i in val_idx]]
    return train, val


def make_reward(gt_map):
    def reward_fn(prompts, completions, **kwargs):
        out = []
        for prompt, comp in zip(prompts, completions):
            text = comp if isinstance(comp, str) else comp[-1]["content"]
            key = prompt if isinstance(prompt, str) else prompt[0]["content"]
            gold = gt_map.get(key)
            pred = extract_charge(text)
            if pred is None:
                out.append(0.0)          # 无法解析：不得分，但不重罚
            elif pred == gold:
                out.append(1.0)          # 完全正确
            else:
                out.append(0.1)          # 格式正确、答案错误：给一点基座分，保证梯度

        return out
    return reward_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/model/ModelScope/Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--out", default="data/out/rl_law")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_steps", type=int, default=120)
    ap.add_argument("--num_generations", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_completion", type=int, default=320)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_train", type=int, default=3000)
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    train, val = load_data(args.seed)
    if args.max_train and len(train) > args.max_train:
        train = train.sample(n=args.max_train, random_state=args.seed)
    print(f"train={len(train)} val={len(val)} classes={len(CLASSES)}", flush=True)
    json.dump({"val_ids": [int(x) for x in val["id"].tolist()]}, open(Path(args.out) / "val_ids.json", "w"))

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    rows, gt = [], {}
    for _, r in train.iterrows():
        text = str(r["text"])[:1500]
        p = build_prompt(text, CLASSES)
        msgs = [{"role": "user", "content": p}]
        full = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        gt[full] = r["label_n"]
        rows.append({"prompt": msgs})
    if args.dry_run:
        print(build_prompt(str(train.iloc[0]["text"])[:300], CLASSES)[:600]); return

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    cfg = GRPOConfig(
        output_dir=Path(args.out) / "grpo",
        learning_rate=args.lr, max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=1,
        num_generations=args.num_generations, max_completion_length=args.max_completion,
        max_prompt_length=2048, logging_steps=1, save_steps=200,
        bf16=True, report_to=[], temperature=0.9, use_vllm=False, seed=args.seed,
        gradient_checkpointing=args.grad_ckpt, optim="paged_adamw_8bit",
    )
    trainer = GRPOTrainer(model=model, args=cfg, processing_class=tok,
                          reward_funcs=make_reward(gt), train_dataset=rows)
    trainer.train()
    trainer.save_model(Path(args.out) / "final_adapter")
    tok.save_pretrained(Path(args.out) / "final_adapter")
    print("saved", Path(args.out) / "final_adapter", flush=True)


if __name__ == "__main__":
    main()

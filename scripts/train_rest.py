#!/usr/bin/env python3
"""
路线 B（lawbench, 拒绝采样自蒸馏 / ReST）：

为什么不用 GRPO 直接做：
  3B 模型在 191 类罪名上初测正确率约 26-33%，且约 20% 的回答自创了不在候选
  列表里的罪名。GRPO 的组内奖励方差极小（多数为 0.1 的基座分），梯度信号弱。

本脚本采用拒绝采样 + SFT（ReST 式自蒸馏）：
  1. 让 base 模型对训练集每题采样 K 次；
  2. 只保留**答案与标签完全一致**的样本 —— 这就是可验证奖励筛出来的"好轨迹"；
  3. 在这些成功轨迹上做 LoRA 监督微调。
  这样模型学到的是"如何正确地从候选列表选出罪名"，且格式由成功样本保证。

评测在 held-out 验证集上进行，base 与 SFT 后模型使用同一个 evaluator。
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, Trainer, TrainingArguments)

BASE = Path("data/lawbench/data")
CLASSES = json.load(open(BASE / "public/classes.json"))


def norm(s: str) -> str:
    return re.sub(r"[\[\]]", "", s).strip()


CLASS_SET = {norm(c) for c in CLASSES}


def build_prompt(text: str, classes_display: list[str]) -> str:
    clist = "、".join(classes_display)
    return (
        "你是中国刑事案件的罪名认定助手。\n"
        "任务：阅读案件事实，从下面给出的**候选罪名列表**中选出最符合的**一个**。\n"
        "硬性要求：你输出的罪名必须与候选列表中的某一项**逐字完全相同**。\n\n"
        f"候选罪名列表（共 {len(classes_display)} 项）：\n{clist}\n\n"
        f"案件事实：{text}\n\n"
        "请先简要分析事实要点，最后一行必须是：\n罪名：<从候选列表中原样复制的一个罪名>"
    )


ANS_RE = re.compile(r"罪名\s*[:：]\s*([^\n]+)")


def extract_charge(text: str) -> str | None:
    if not text:
        return None
    m = ANS_RE.findall(text)
    if m:
        cand = re.sub(r"[「」『』\s。.]", "", norm(m[-1]))
        if cand in CLASS_SET:
            return cand
        for c in sorted(CLASS_SET, key=len, reverse=True):
            if cand.startswith(c) or c in cand:
                return c
    cleaned = re.sub(r"[「」『』\s。.，,]", "", text)
    best = None
    for c in CLASS_SET:
        if c in cleaned and (best is None or len(c) > len(best)):
            best = c
    return best


def load_split(seed: int, val_frac: float = 0.1):
    df = pd.read_csv(BASE / "training_data/train.csv").dropna(subset=["text", "label"])
    df = df.assign(label_n=df["label"].map(norm))
    df = df[df["label_n"].isin(CLASS_SET)]
    rng = random.Random(seed)
    idx = list(range(len(df)))
    rng.shuffle(idx)
    n_val = int(len(idx) * val_frac)
    val_idx = set(idx[:n_val])
    return df.iloc[[i for i in idx if i not in val_idx]].reset_index(drop=True), \
           df.iloc[[i for i in idx if i in val_idx]].reset_index(drop=True)


@torch.no_grad()
def generate(model, tok, prompts, max_new=256, do_sample=True, k=1, batch=8):
    """批量生成。返回 list[list[response]]，外层每题 K 个。"""
    outs = []
    for i in range(0, len(prompts), batch):
        chunk = prompts[i:i + batch]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=2048).to(model.device)
        n = enc["input_ids"].shape[1]
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=do_sample,
                             temperature=0.9 if do_sample else None,
                             top_p=0.95 if do_sample else None,
                             num_return_sequences=k if do_sample else 1,
                             pad_token_id=tok.pad_token_id)
        dec = tok.batch_decode(gen[:, n:], skip_special_tokens=True)
        if k > 1:
            for j in range(len(chunk)):
                outs.append(dec[j * k:(j + 1) * k])
        else:
            outs.extend([[d] for d in dec])
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/model/ModelScope/Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--out", default="data/out/rest_law")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=8, help="每题采样次数")
    ap.add_argument("--max_pool", type=int, default=600, help="用于采样的题数")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--stage", default="all", choices=["all", "sample", "train", "eval"])
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    train, val = load_split(args.seed)
    print(f"train={len(train)} val={len(val)}", flush=True)
    val.to_csv(out / "val.csv", index=False)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tok.padding_side = "left"   # 批量生成必须左填充
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)

    # ---------------- 阶段 1：拒绝采样 ----------------
    if args.stage in ("all", "sample"):
        pool = train.sample(n=min(args.max_pool, len(train)), random_state=args.seed)
        prompts = []
        for _, r in pool.iterrows():
            msgs = [{"role": "user", "content": build_prompt(str(r["text"])[:1500], CLASSES)}]
            prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        print(f"sampling {len(prompts)} x {args.k} ...", flush=True)
        model.eval()
        gens = generate(model, tok, prompts, k=args.k)
        kept, stats = [], {"total": 0, "parsed": 0, "correct": 0}
        for (_, r), plist, glist in zip(pool.iterrows(), prompts, gens):
            for resp in glist:
                stats["total"] += 1
                pred = extract_charge(resp)
                if pred is None:
                    continue
                stats["parsed"] += 1
                if pred == r["label_n"]:
                    stats["correct"] += 1
                    kept.append({"prompt": plist, "response": resp, "label": r["label_n"]})
        print(f"rejection stats: {stats} | kept={len(kept)}", flush=True)
        json.dump(kept, open(out / "kept.json", "w"), ensure_ascii=False)
        json.dump(stats, open(out / "reject_stats.json", "w"))

    # ---------------- 阶段 2：在成功轨迹上 SFT ----------------
    if args.stage in ("all", "train"):
        kept = json.load(open(out / "kept.json"))
        if not kept:
            raise SystemExit("没有采样到任何正确轨迹，无法训练")
        def to_feat(ex):
            ids_p = tok(ex["prompt"], add_special_tokens=False)["input_ids"]
            ids_r = tok(ex["response"] + tok.eos_token, add_special_tokens=False)["input_ids"]
            labels = [-100] * len(ids_p) + ids_r
            return {"input_ids": ids_p + ids_r, "labels": labels}
        ds = Dataset.from_list(kept).map(to_feat, remove_columns=list(kept[0].keys()))
        ds = ds.filter(lambda x: len(x["input_ids"]) <= 2048)
        print(f"SFT samples: {len(ds)}", flush=True)
        model.config.use_cache = False
        mdl = get_peft_model(model, LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM"))
        # 关键：确保 LoRA 参数可训练（之前因顺序问题被冻结）
        for n_, p_ in mdl.named_parameters():
            if "lora_" in n_:
                p_.requires_grad_(True)
        n_train = sum(p_.numel() for p_ in mdl.parameters() if p_.requires_grad)
        print(f"trainable params: {n_train}", flush=True)
        assert n_train > 0, "no trainable params"
        targs = TrainingArguments(
            output_dir=str(out / "sft"), max_steps=args.steps,
            per_device_train_batch_size=2, gradient_accumulation_steps=4,
            learning_rate=args.lr, bf16=True, logging_steps=5,
            save_strategy="no", report_to=[], seed=args.seed,
            gradient_checkpointing=True, optim="paged_adamw_8bit",
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        Trainer(model=mdl, args=targs, train_dataset=ds,
                data_collator=DataCollatorForSeq2Seq(tok, padding=True)).train()
        mdl.save_pretrained(out / "final_adapter")
        tok.save_pretrained(out / "final_adapter")
        print("saved adapter", flush=True)

    # ---------------- 阶段 3：held-out 评测 ----------------
    if args.stage in ("all", "eval"):
        from peft import PeftModel
        adapter = out / "final_adapter"
        if adapter.exists():
            model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
        model.eval()
        prompts = []
        for _, r in val.iterrows():
            msgs = [{"role": "user", "content": build_prompt(str(r["text"])[:1500], CLASSES)}]
            prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        gens = generate(model, tok, prompts, do_sample=False, k=1)
        correct = parsed = 0
        details = []
        for (_, r), g in zip(val.iterrows(), gens):
            pred = extract_charge(g[0])
            if pred is not None:
                parsed += 1
            if pred == r["label_n"]:
                correct += 1
            details.append({"id": int(r["id"]), "gold": r["label_n"], "pred": pred})
        n = len(val)
        res = {"n": n, "correct": correct, "parsed": parsed,
               "accuracy": 100 * correct / n, "parse_rate": 100 * parsed / n,
               "details": details}
        json.dump(res, open(out / "eval_result.json", "w"), indent=1)
        print(f"EVAL n={n} correct={correct} acc={res['accuracy']:.2f}% parse_rate={res['parse_rate']:.1f}%", flush=True)


if __name__ == "__main__":
    main()

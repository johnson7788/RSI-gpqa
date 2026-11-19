#!/usr/bin/env python3
"""
阶段 2：在模型自己**答对**的轨迹上做 SFT（拒绝采样自蒸馏）。

与之前 D-CoT 实验的本质区别：
  - 训练内容是模型**自己**产生的推理（不是老师写的）
  - 不添加任何控制标签，保持原生推理风格
  - 只用 status=="correct" 的轨迹

训练时 prompt 部分 labels=-100（不学 prompt），只学 response。
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, Trainer, TrainingArguments)

MODEL = "/model/ModelScope/Qwen/Qwen3-8B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", default="data/selfdist/trajectories_train.json")
    ap.add_argument("--out", default="data/out/selfdist")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=8192)
    ap.add_argument("--max_kept", type=int, default=0, help="0=全部")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    recs = json.load(open(args.traj))
    if recs and "source" in recs[0]:
        # 混合格式（方案B）：已包含 prompt/response/source
        kept = [r for r in recs if r.get("prompt") and r.get("response", "").strip()]
        import collections
        print(f"mixed dataset rows={len(kept)} sources={dict(collections.Counter(r['source'] for r in kept))}", flush=True)
    else:
        kept = [r for r in recs if r["status"] == "correct" and r["response"].strip()]
        print(f"trajectories={len(recs)} | correct(kept)={len(kept)}", flush=True)
    if args.max_kept and len(kept) > args.max_kept:
        random.Random(args.seed).shuffle(kept)
        kept = kept[:args.max_kept]
    if not kept:
        raise SystemExit("no correct trajectories to learn from")

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    def feat(r):
        p_ids = tok(r["prompt"], add_special_tokens=False)["input_ids"]
        r_ids = tok(r["response"] + tok.eos_token, add_special_tokens=False)["input_ids"]
        if len(p_ids) + len(r_ids) > args.max_len:
            # 截断 response，保留完整 prompt
            r_ids = r_ids[: max(1, args.max_len - len(p_ids))]
        return {"input_ids": p_ids + r_ids, "labels": [-100] * len(p_ids) + r_ids}

    ds = Dataset.from_list(kept).map(feat, remove_columns=list(kept[0].keys()))
    ds = ds.filter(lambda x: len(x["input_ids"]) <= args.max_len)
    print(f"SFT rows={len(ds)}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    model.config.use_cache = False
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, task_type=TaskType.CAUSAL_LM,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    for n_, p_ in model.named_parameters():
        if "lora_" in n_: p_.requires_grad_(True)
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=str(out / "sft"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        bf16=True, logging_steps=1, save_strategy="no", report_to=[],
        seed=args.seed, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      data_collator=DataCollatorForSeq2Seq(tok, padding=True))
    trainer.train()
    model.save_pretrained(str(out / "final_adapter"))
    tok.save_pretrained(str(out / "final_adapter"))
    print("saved", out / "final_adapter", flush=True)


if __name__ == "__main__":
    main()

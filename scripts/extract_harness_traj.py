#!/usr/bin/env python3
"""从 SIA harness 的 Gen 1 产出中提取轨迹，用于权重更新（自蒸馏）。

严格排除 held-out 40 题，只用训练题的**正确**轨迹。
"""
import json, glob, os
from pathlib import Path

RUN = "runs/run_22/gen_1/agent_execution"
OUT = "data/selfdist/harness_traj.json"
priv = {x["id"]: x for x in json.load(open('data/gpqa/data/private/diamond_questions.json'))}
test_ids = set(json.load(open('data/out/test_ids.json')))
print(f"held-out excluded: {len(test_ids)}", flush=True)

recs, stats = [], {"total": 0, "no_assistant": 0, "empty": 0, "correct": 0, "incorrect": 0, "heldout": 0}
for f in sorted(glob.glob(RUN + "/execution_q*.json")):
    try:
        msgs = json.load(open(f))
    except Exception:
        continue
    if not isinstance(msgs, list):
        continue
    stats["total"] += 1
    # 找 system+user 和 assistant
    user_msg, asst = None, None
    for m in msgs:
        if isinstance(m, dict):
            if m.get("role") == "user" and user_msg is None:
                user_msg = m.get("content")
            elif m.get("role") == "assistant":
                asst = m
    if asst is None:
        stats["no_assistant"] += 1
        continue
    qid = asst.get("question_id")
    content = (asst.get("content") or "").strip()
    if not content:
        stats["empty"] += 1
        continue
    if qid in test_ids:
        stats["heldout"] += 1
        continue
    gold = (priv.get(qid, {}).get("correct_answer_letter") or "").upper()
    pred = (asst.get("model_answer") or "").upper()
    if pred and pred == gold:
        stats["correct"] += 1
        recs.append({"id": qid, "prompt": user_msg, "response": content,
                     "gold": gold, "source": "harness_correct"})
    else:
        stats["incorrect"] += 1

print("stats:", stats, flush=True)
print(f"usable trajectories: {len(recs)}", flush=True)
Path(OUT).parent.mkdir(parents=True, exist_ok=True)
json.dump(recs, open(OUT, "w"), ensure_ascii=False)
print("saved", OUT, flush=True)

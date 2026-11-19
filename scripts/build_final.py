#!/usr/bin/env python3
"""构建最终训练集：harness 正确轨迹 + 教师对失败题的解法。"""
import json
from pathlib import Path

priv = {x["id"]: x for x in json.load(open('data/gpqa/data/private/diamond_questions.json'))}
harness = json.load(open('data/selfdist/harness_traj.json'))
teacher = [x for x in json.load(open('data/selfdist/teacher_solutions.json')) if x.get("ok")]

def build_prompt(it):
    opts = "\n".join(f"{k}. {v}" for k, v in sorted(it["options"].items()))
    return (f"Question: {it['Question']}\n\nOptions:\n{opts}\n\n"
            "Think step by step. End with a final line 'Answer: X' where X is one of A, B, C, D.")

rows, seen = [], set()
# 1) harness 产出的正确轨迹（保留其原生 prompt）
for r in harness:
    rows.append({"prompt": r["prompt"], "response": r["response"], "source": "harness"})
    seen.add(r["id"])
# 2) 教师对 harness 失败题的解法（不重复同一题）
added = 0
for t in teacher:
    if t["id"] in seen:
        continue
    rows.append({"prompt": build_prompt(priv[t["id"]]), "response": t["response"], "source": "teacher"})
    seen.add(t["id"])
    added += 1

print(f"total={len(rows)} | harness={len(harness)} | teacher_added={added}", flush=True)
Path('data/selfdist/final_train.json').write_text(
    json.dumps(rows, ensure_ascii=False))
print("saved final_train.json", flush=True)

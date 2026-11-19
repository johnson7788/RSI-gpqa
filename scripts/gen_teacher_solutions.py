#!/usr/bin/env python3
"""教师指导：对学生答错的题，让老师独立给出完整解法。

关键：**绝不把标准答案告诉老师**。老师只看题目，自己解；
解完后校验其答案是否与 ground truth 一致，只保留正确解。
"""
from __future__ import annotations
import json, os, re, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, FIRST_COMPLETED, wait

RELAY = os.environ.get("RELAY_BASE", "https://YOUR_RELAY_HOST/v1")
KEY = os.environ["SIA_RELAY_API_KEY"]
TEACHER = os.environ.get("TEACHER_MODEL", "deepseek-v4-pro")
OUT = "data/selfdist/teacher_solutions.json"

priv = {x["id"]: x for x in json.load(open('data/gpqa/data/private/diamond_questions.json'))}
traj = json.load(open('data/selfdist/trajectories_train.json'))
failed = [r for r in traj if r["status"] in ("incorrect", "null")]
print(f"failed questions: {len(failed)}", flush=True)

SYSTEM = """You are an expert scientist solving a graduate-level multiple-choice question.

- Reason carefully and completely. Show every calculation step explicitly.
- Verify units, signs, and magnitudes before committing to an answer.
- Then end with EXACTLY one final line of the form:
Answer: X
where X is a single letter A, B, C, or D.

Do not hedge between options. Commit to one answer."""

ANS = re.compile(r"answer\s*[:：]\s*\(?([ABCD])\)?", re.I)
def extract(t):
    if not t: return None
    m = re.findall(r"\\boxed\{([ABCD])\}", t)
    if m: return m[-1]
    m = ANS.findall(t)
    if m: return m[-1].upper()
    m = re.findall(r"\b([ABCD])\b", t)
    return m[-1].upper() if m else None

def build(it):
    opts = "\n".join(f"{k}. {v}" for k, v in sorted(it["options"].items()))
    return f"Question: {it['Question']}\n\nOptions:\n{opts}\n"

def solve(rec, _retry=2):
    it = priv[rec["id"]]
    user = build(it)
    last = None
    for a in range(_retry + 1):
        try:
            body = json.dumps({
                "model": TEACHER,
                "messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": user}],
                "temperature": 0.3, "max_tokens": 24000,
            }).encode()
            req = urllib.request.Request(RELAY + "/chat/completions", data=body,
                                         headers={"Authorization": f"Bearer {KEY}",
                                                  "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=420) as r:
                d = json.load(r)
            txt = d["choices"][0]["message"].get("content") or ""
            pred = extract(txt)
            if pred is None:
                raise ValueError("no answer letter")
            gold = (it.get("correct_answer_letter") or "").upper()
            return {"id": rec["id"], "response": txt, "teacher_pred": pred,
                    "gold": gold, "ok": pred == gold}
        except Exception as e:
            last = e
            time.sleep(2 * (a + 1))
    return {"id": rec["id"], "error": str(last)[:200], "ok": False}

results, errors = [], 0
ex = ThreadPoolExecutor(max_workers=10)
pending = {ex.submit(solve, r) for r in failed}
try:
    while pending:
        done, pending = wait(pending, return_when=FIRST_COMPLETED)
        for f in done:
            try:
                results.append(f.result())
            except Exception as e:
                errors += 1
        if len(results) % 10 == 0:
            ok = sum(1 for r in results if r.get("ok"))
            print(f"  {len(results)}/{len(failed)} | teacher_correct={ok}", flush=True)
            json.dump(results, open(OUT, "w"), ensure_ascii=False)
finally:
    for f in pending: f.cancel()
    ex.shutdown(wait=False, cancel_futures=True)

json.dump(results, open(OUT, "w"), ensure_ascii=False)
ok = sum(1 for r in results if r.get("ok"))
print(f"DONE: solved={len(results)} teacher_correct={ok} teacher_acc={100*ok/max(1,len(results)):.1f}%", flush=True)

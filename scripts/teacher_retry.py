#!/usr/bin/env python3
"""教师多次尝试：对未解出的题反复采样，直到答对或达到上限。

关键：依然**绝不把标准答案告诉老师**。老师独立解，只保留答对的轨迹。
策略：每题最多 ATTEMPTS 次，温度渐进提高（增加探索多样性）。
"""
from __future__ import annotations
import json, os, re, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, FIRST_COMPLETED, wait

RELAY = os.environ.get("RELAY_BASE", "https://YOUR_RELAY_HOST/v1")
KEY = os.environ["SIA_RELAY_API_KEY"]
TEACHER = os.environ.get("TEACHER_MODEL", "deepseek-v4-pro")
OUT = "data/selfdist/teacher_solutions.json"
ATTEMPTS = int(os.environ.get("ATTEMPTS", "5"))

priv = {x["id"]: x for x in json.load(open('data/gpqa/data/private/diamond_questions.json'))}
existing = json.load(open(OUT))
by_id = {x["id"]: x for x in existing}
unsolved = [x["id"] for x in existing if not x.get("ok")]
print(f"to solve: {len(unsolved)} questions, up to {ATTEMPTS} attempts each", flush=True)

SYSTEM = """You are an expert scientist solving a graduate-level multiple-choice question.

- Reason carefully and completely. Show every calculation step explicitly.
- Verify units, signs, and magnitudes before committing to an answer.
- Actively consider and rule out each distractor option.
- If a calculation yields something implausible, re-derive it a different way.

Then end with EXACTLY one final line of the form:
Answer: X
where X is a single letter A, B, C, or D. Commit to one answer."""

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

def attempt(qid, k):
    it = priv[qid]
    temp = 0.3 + 0.15 * k
    body = json.dumps({
        "model": TEACHER,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": build(it)}],
        "temperature": min(temp, 1.0), "max_tokens": 24000,
    }).encode()
    req = urllib.request.Request(RELAY + "/chat/completions", data=body,
                                 headers={"Authorization": f"Bearer {KEY}",
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=420) as r:
        d = json.load(r)
    txt = d["choices"][0]["message"].get("content") or ""
    pred = extract(txt)
    gold = (it.get("correct_answer_letter") or "").upper()
    return {"id": qid, "response": txt, "teacher_pred": pred, "gold": gold,
            "ok": pred is not None and pred == gold, "attempt": k}

def solve_with_retries(qid):
    last = None
    for k in range(ATTEMPTS):
        try:
            r = attempt(qid, k)
            if r["ok"]:
                return r
            last = r
        except Exception as e:
            last = {"id": qid, "error": str(e)[:150], "ok": False}
        time.sleep(1)
    return last

results = []
ex = ThreadPoolExecutor(max_workers=8)
pending = {ex.submit(solve_with_retries, q) for q in unsolved}
try:
    while pending:
        done, pending = wait(pending, return_when=FIRST_COMPLETED)
        for f in done:
            try: results.append(f.result())
            except Exception as e: results.append({"ok": False, "error": str(e)[:150]})
        if len(results) % 5 == 0:
            ok = sum(1 for r in results if r.get("ok"))
            print(f"  resolved {len(results)}/{len(unsolved)} | newly_correct={ok}", flush=True)
finally:
    for f in pending: f.cancel()
    ex.shutdown(wait=False, cancel_futures=True)

# 合并回原文件
merged = []
for x in existing:
    if x.get("ok"):
        merged.append(x)
    else:
        r = next((y for y in results if y.get("id") == x["id"]), None)
        merged.append(r if (r and r.get("ok")) else x)
for r in results:
    if r.get("ok") and not any(y["id"] == r["id"] and y.get("ok") for y in merged):
        merged.append(r)
json.dump(merged, open(OUT, "w"), ensure_ascii=False)
ok = sum(1 for x in merged if x.get("ok"))
print(f"DONE: total_solved={ok}/{len(merged)} ({100*ok/len(merged):.1f}%)", flush=True)

#!/usr/bin/env python3
"""
v3 数据生成：加入 STEM 推理场景 + 难度感知的"伸缩"推理结构。

上一版（v2）的失败原因：
  1200 条全为日常办公场景（定价/排班/调试），模型学成了"对什么都简短回答"，
  GPQA 平均 token 从 8192 降到 819，难题推理被压没了，准确率 65%→50%。

本版修正：
  1. 场景分布：40% STEM 推理（物理/化学/生物/数学/工程，但**不使用 GPQA 题库**），
     60% 通用推理 —— 让模型知道"科学题需要长推理"
  2. 难度感知：每条样本标注 easy / hard。
     - easy  -> chosen 用"短且收敛"的推理（LOW→MID，快速作答，~150 字）
     - hard  -> chosen 用"充分展开"的推理（LOW→MID→MID 多步计算→HIGH 校验，~500-800 字）
     这样模型学到的是 **按难度伸缩**，而不是一律缩短。
  3. 控制标签语义明确："LOW/ MID / HIGH 是推理阶段的切换，不是字数限制"
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, FIRST_COMPLETED, wait
from pathlib import Path

RELAY = os.environ.get("RELAY_BASE", "https://YOUR_RELAY_HOST/v1")
KEY = os.environ["SIA_RELAY_API_KEY"]
TEACHER = os.environ.get("TEACHER_MODEL", "deepseek-v4-pro")

# 40% STEM（自造题目，不是 GPQA，去污染时校验），60% 通用
STEM_SCENARIOS = [
    ("physics-mechanics", "a projectile or forces problem with several given quantities"),
    ("physics-optics", "a refraction / lens / interference calculation"),
    ("physics-thermo", "a heat capacity / ideal gas / entropy calculation"),
    ("physics-em", "a circuit with resistors and capacitors, or an E/B field problem"),
    ("chemistry-organic", "predicting a reaction product or a stereochemical outcome"),
    ("chemistry-equilibria", "computing pH, solubility, or equilibrium concentrations"),
    ("chemistry-stoichiometry", "a limiting-reagent / yield calculation"),
    ("biology-genetics", "a pedigree or allele frequency reasoning task"),
    ("biology-molecular", "reading a construct, primer, or expression result"),
    ("biology-biochem", "an enzyme kinetics or metabolic pathway question"),
    ("math-algebra", "solving a system or manipulating an expression step by step"),
    ("math-calculus", "an integral, derivative, or a convergence question"),
    ("math-probability", "a combinatorics or conditional-probability question"),
    ("math-linear-algebra", "an eigenvalue / matrix decomposition question"),
    ("engineering-signal", "a filter or sampling-rate reasoning question"),
    ("engineering-thermal", "a heat-transfer or efficiency calculation"),
    ("materials", "a stress/strain or phase-diagram reasoning question"),
    ("astronomy", "an orbital or redshift/luminosity calculation"),
]

GENERAL_SCENARIOS = [
    ("business math", "pricing and break-even calculation"),
    ("logistics", "scheduling with capacity and time-window constraints"),
    ("software debugging", "root-causing a production issue from logs"),
    ("personal finance", "comparing loan or investment offers"),
    ("statistics", "choosing a test and interpreting its output"),
    ("policy writing", "drafting a rule that is unambiguous"),
    ("inventory", "deciding reorder points from historical sales"),
    ("unit conversion", "converting and sanity-checking a measurement"),
    ("travel planning", "an itinerary under a fixed budget and time"),
    ("hardware troubleshooting", "diagnosing why a device fails to start"),
]

FAILURE_MODES = [
    "Token Omission (skips fact-checking, jumps straight into calculation)",
    "Premature Creativity (explores alternatives before establishing the givens)",
    "Deliberation Loop (repeats 'Wait, but...' and never commits)",
    "Architectural Naivety (ignores a hard constraint present in the prompt)",
    "Compliance Hallucination (invents a fact or rule that was not given)",
    "Unchecked Assumption (proceeds on an unverified assumption)",
    "Over-Truncation (a genuinely HARD multi-step problem answered in one rushed line, skipping needed steps)",
    "Over-Expansion (a trivial problem padded with unnecessary loops and restatement)",
]

SYSTEM = """You generate training data for preference optimization of a reasoning model.

The model must learn WHEN to reason at length and WHEN to be brief:
- <TEMP_LOW>  = establish the given facts, constraints, and what is being asked (exactly once)
- <TEMP_MID>  = do the actual computation / recall / application (may be used MULTIPLE times for a hard problem)
- <TEMP_HIGH> = check the result against an alternative approach or sanity bounds (only if useful)

Control tags switch REASONING PHASE. They do NOT mean "answer briefly".
For a HARD problem, the <TEMP_MID> phase should be long, with several explicit
calculation steps shown.

You will be given a SCENARIO and a DIFFICULTY. Produce ONE JSON object with:
  "user_prompt": a realistic, self-contained question. All facts needed must be IN the
     prompt. For HARD, it must genuinely require 3+ distinct reasoning steps.
  "difficulty": echo "easy" or "hard".
  "thought_chosen": <think>...</think> planning, under 100 words.
  "chosen_response": the ideal answer, using the control tags in a sensible order.
     * easy: LOW then MID, total under 160 words.
     * hard: LOW, then MULTIPLE MID segments with explicit step-by-step calculations,
       then optionally HIGH to verify. Total 250-500 words. Do not skip steps.
  "thought_rejected": <think>...</think> planning for a "competent but misguided"
     persona exhibiting the given FAILURE MODE, under 100 words.
  "rejected_response": fluent but flawed, exhibiting that failure mode. Under 300 words.
  "failure_mode": the failure mode you targeted.

Rules:
- Do NOT copy or paraphrase any published benchmark question (GPQA/MMLU/AIME etc).
  Invent fresh numbers and setups.
- Output ONLY the JSON object. No markdown fences.
"""


def _call_once(scenario, difficulty, failure_mode, n, seed):
    user = (f"SCENARIO: {scenario[0]} — {scenario[1]}\n"
            f"DIFFICULTY: {difficulty}\n"
            f"FAILURE MODE for the rejected sample: {failure_mode}\n"
            f"Variation #{n} (seed {seed}). Use concrete, freshly invented numbers.")
    body = json.dumps({
        "model": TEACHER,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": user}],
        "temperature": 0.95,
        "max_tokens": 24000,
    }).encode()
    req = urllib.request.Request(RELAY + "/chat/completions", data=body,
                                 headers={"Authorization": f"Bearer {KEY}",
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=420) as r:
        d = json.load(r)
    txt = d["choices"][0]["message"].get("content") or ""
    txt = re.sub(r"^```(?:json)?|```$", "", txt.strip(), flags=re.M).strip()
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        raise ValueError("no json")
    raw = m.group()
    try:
        o = json.loads(raw)
    except json.JSONDecodeError:
        # 修复常见的控制字符问题（未转义的换行/制表符）
        fixed = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", raw)
        try:
            o = json.loads(fixed, strict=False)
        except Exception:
            o = json.loads(fixed.replace("\n", "\\n").replace("\t", "\\t"), strict=False)
    need = ["user_prompt", "thought_chosen", "chosen_response",
            "thought_rejected", "rejected_response"]
    if any(not o.get(k) for k in need):
        raise ValueError("missing fields")
    o["scenario"] = scenario[0]
    o["difficulty"] = difficulty
    o["failure_mode"] = failure_mode
    return [o]


def call_teacher(scenario, difficulty, failure_mode, n, seed, _retry=2):
    last = None
    for a in range(_retry + 1):
        try:
            return _call_once(scenario, difficulty, failure_mode, n, seed)
        except Exception as e:
            last = e
            time.sleep(1.5 * (a + 1))
    raise last


def decontaminate(samples, gpqa_path, ng=8):
    gpqa = json.load(open(gpqa_path))
    corpus = " ".join(x["Question"] + " " + " ".join(x["options"].values()) for x in gpqa).lower()
    corpus = re.sub(r"\s+", " ", corpus)
    w = corpus.split()
    grams = {" ".join(w[i:i + ng]) for i in range(max(0, len(w) - ng + 1))}
    kept, dropped = [], 0
    for s in samples:
        t = re.sub(r"\s+", " ", (s["user_prompt"] + " " + s["chosen_response"]).lower())
        tw = t.split()
        if any(" ".join(tw[i:i + ng]) in grams for i in range(max(0, len(tw) - ng + 1))):
            dropped += 1
        else:
            kept.append(s)
    print(f"decontamination: dropped={dropped} kept={len(kept)}", flush=True)
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=2000)
    ap.add_argument("--stem_frac", type=float, default=0.4)
    ap.add_argument("--hard_frac", type=float, default=0.6)
    ap.add_argument("--out", default="data/dcot_data/pairs_v3.json")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--save_every", type=int, default=50)
    ap.add_argument("--resume", default=None, help="从已有的 partial 文件继续")
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    jobs = []
    for i in range(int(args.target * 2.2)):
        is_stem = rng.random() < args.stem_frac
        sc = rng.choice(STEM_SCENARIOS if is_stem else GENERAL_SCENARIOS)
        diff = "hard" if rng.random() < args.hard_frac else "easy"
        jobs.append((sc, diff, rng.choice(FAILURE_MODES), i, rng.randrange(10**6)))

    results, errors = [], 0
    if args.resume:
        try:
            results = json.load(open(args.resume))
            print(f"resumed {len(results)} samples from {args.resume}", flush=True)
        except Exception as e:
            print(f"resume failed: {e}", flush=True)
    ex = ThreadPoolExecutor(max_workers=args.workers)
    pending = {ex.submit(call_teacher, *j) for j in jobs}
    try:
        while pending and len(results) < args.target:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for f in done:
                try:
                    results.extend(f.result())
                except Exception as e:
                    errors += 1
                    if errors <= 6:
                        print(f"  error: {type(e).__name__}: {str(e)[:110]}", flush=True)
            if len(results) % args.save_every < args.workers:
                print(f"  generated {len(results)} (errors={errors})", flush=True)
                json.dump(results, open(str(args.out) + ".partial", "w"), ensure_ascii=False)
    finally:
        for f in pending:
            f.cancel()
        ex.shutdown(wait=False, cancel_futures=True)

    print(f"generated={len(results)} errors={errors}", flush=True)
    results = decontaminate(results, "data/gpqa/data/public/diamond_questions.json")
    import collections
    print("difficulty dist:", collections.Counter(s["difficulty"] for s in results), flush=True)
    json.dump(results, open(args.out, "w"), ensure_ascii=False, indent=1)
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""汇总各 run 的逐代结果。"""
import json, sys, glob, os

runs = sys.argv[1:] or ["1", "2", "3"]
print(f"{'run':>4} {'gen':>4} {'correct':>8} {'wrong':>6} {'missing':>8} {'acc(att)':>9} {'acc(198)':>9}")
print("-" * 58)
for r in runs:
    for f in sorted(glob.glob(f"runs/run_{r}/gen_*/results.json")):
        g = f.split("/")[-2]
        d = json.load(open(f))
        n = d.get("total_questions", 198)
        att = d["correct"] + d["incorrect"]
        acc_att = 100 * d["correct"] / att if att else 0
        print(f"{r:>4} {g.replace('gen_','G'):>4} {d['correct']:>8} {d['incorrect']:>6} {d['missing']:>8} "
              f"{acc_att:>8.2f}% {100*d['correct']/n:>8.2f}%")

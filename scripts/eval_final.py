"""最终评测：base vs ORPO-tuned，两种 prompt，同一套解析器。
用法: python eval_final.py <adapter|none> <out_json> [n]
"""
import json, re, sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL="/model/ModelScope/Qwen/Qwen3-8B"
adapter = sys.argv[1]
outpath = sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv)>3 else 40

priv={x["id"]:x for x in json.load(open('data/gpqa/data/private/diamond_questions.json'))}
ids=json.load(open('data/out/test_ids.json'))[:N]

ANS=re.compile(r"answer\s*[:：]\s*\(?([ABCD])\)?", re.I)
def extract(t):
    if not t: return None
    m=re.findall(r"\\boxed\{([ABCD])\}", t)
    if m: return m[-1]
    m=ANS.findall(t)
    if m: return m[-1].upper()
    m=re.findall(r"\b([ABCD])\b", t)
    return m[-1].upper() if m else None

def build(it, mode):
    opts="\n".join(f"{k}. {v}" for k,v in sorted(it["options"].items()))
    head=f"Question: {it['Question']}\n\nOptions:\n{opts}\n\n"
    if mode=="cot":
        return head+"Think step by step. End with a final line 'Answer: X' where X is one of A, B, C, D."
    return head+(
        "Use a disciplined, bounded reasoning process with explicit control tags:\n"
        "1) <TEMP_LOW> list the given quantities / facts and what is being asked (no speculation)\n"
        "2) <TEMP_MID> do the single needed calculation or recall the needed fact\n"
        "3) <TEMP_HIGH> ONLY if a second approach is genuinely needed, otherwise skip\n"
        "Do NOT loop or restate. Hard limit: at most 8 short lines total. "
        "Then output exactly one final line: 'Answer: X' where X is one of A, B, C, D.")

tok=AutoTokenizer.from_pretrained(MODEL); tok.padding_side="left"
if tok.pad_token is None: tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto")
if adapter and adapter!="none":
    from peft import PeftModel
    model=PeftModel.from_pretrained(model, adapter).merge_and_unload()
model.eval()

res={"adapter":adapter,"results":{}}
for mode in ["cot","disciplined"]:
    texts=[tok.apply_chat_template([{"role":"user","content":build(priv[q],mode)}],
            tokenize=False, add_generation_prompt=True, enable_thinking=True) for q in ids]
    c=i=m=0; toks=[]; det=[]
    for b in range(0,len(texts),2):
        enc=tok(texts[b:b+2], return_tensors="pt", padding=True, truncation=True, max_length=2048).to("cuda")
        n=enc["input_ids"].shape[1]
        out=model.generate(**enc, max_new_tokens=8192, do_sample=False, pad_token_id=tok.pad_token_id)
        print(f"  [{mode}] batch {b//2+1}/{-(-len(texts)//2)}", flush=True)
        for j,r in enumerate(tok.batch_decode(out[:,n:], skip_special_tokens=True)):
            tk=out.shape[1]-n; toks.append(tk)
            pred=extract(r); gold=(priv[ids[b+j]].get("correct_answer_letter") or "").upper()
            if pred is None: m+=1; st="null"
            elif pred==gold: c+=1; st="correct"
            else: i+=1; st="incorrect"
            det.append({"id":ids[b+j],"gold":gold,"pred":pred,"status":st,"tokens":tk,
                        "has_tags":"<TEMP_" in r})
    n_=len(ids)
    res["results"][mode]={"n":n_,"correct":c,"incorrect":i,"null":m,
        "acc":100*c/n_,"null_rate":100*m/n_,"avg_tok":sum(toks)/n_,
        "tag_rate":100*sum(1 for d in det if d["has_tags"])/n_,"details":det}
    print(f"RESULT adapter={adapter} mode={mode:12s} n={n_} correct={c} inc={i} null={m} "
          f"| acc={100*c/n_:5.2f}% null_rate={100*m/n_:4.1f}% avg_tok={sum(toks)/n_:.0f} "
          f"tag_rate={res['results'][mode]['tag_rate']:.0f}%", flush=True)

json.dump(res, open(outpath,"w"), indent=1)
print("saved", outpath, flush=True)

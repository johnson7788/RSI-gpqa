"""单模式评测：只跑一个 prompt 配置，便于并行。用法: eval_single.py <adapter> <mode> <out>"""
import json, re, sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
MODEL="/model/ModelScope/Qwen/Qwen3-8B"
adapter, mode, outpath = sys.argv[1], sys.argv[2], sys.argv[3]
priv={x["id"]:x for x in json.load(open('data/gpqa/data/private/diamond_questions.json'))}
ids=json.load(open('data/out/test_ids.json'))
ANS=re.compile(r"answer\s*[:：]\s*\(?([ABCD])\)?", re.I)
def extract(t):
    if not t: return None
    m=re.findall(r"\\boxed\{([ABCD])\}", t)
    if m: return m[-1]
    m=ANS.findall(t)
    if m: return m[-1].upper()
    m=re.findall(r"\b([ABCD])\b", t)
    return m[-1].upper() if m else None
def build(it):
    opts="\n".join(f"{k}. {v}" for k,v in sorted(it["options"].items()))
    head=f"Question: {it['Question']}\n\nOptions:\n{opts}\n\n"
    if mode=="cot":
        return head+"Think step by step. End with a final line 'Answer: X' where X is one of A, B, C, D."
    return head+("Use a disciplined, bounded reasoning process with explicit control tags:\n"
        "1) <TEMP_LOW> list the given quantities / facts and what is being asked\n"
        "2) <TEMP_MID> do the single needed calculation or recall\n"
        "3) <TEMP_HIGH> ONLY if a second approach is needed, otherwise skip\n"
        "Do NOT loop. Then output exactly one final line: 'Answer: X'.")
tok=AutoTokenizer.from_pretrained(MODEL); tok.padding_side="left"
if tok.pad_token is None: tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto")
if adapter and adapter!="none":
    from peft import PeftModel
    model=PeftModel.from_pretrained(model, adapter).merge_and_unload()
model.eval()
texts=[tok.apply_chat_template([{"role":"user","content":build(priv[q])}],tokenize=False,
        add_generation_prompt=True, enable_thinking=True) for q in ids]
c=i=m=0; toks=[]; det=[]
B=int(sys.argv[4]) if len(sys.argv)>4 else 5
for b in range(0,len(texts),B):
    enc=tok(texts[b:b+B], return_tensors="pt", padding=True, truncation=True, max_length=2048).to("cuda")
    n=enc["input_ids"].shape[1]
    with torch.inference_mode():
        out=model.generate(**enc, max_new_tokens=8192, do_sample=False, pad_token_id=tok.pad_token_id)
    print(f"  batch {b//B+1}/{-(-len(texts)//B)}", flush=True)
    for j,r in enumerate(tok.batch_decode(out[:,n:], skip_special_tokens=True)):
        tk=int((out[j,n:]!=tok.pad_token_id).sum()); toks.append(tk)
        pred=extract(r); gold=(priv[ids[b+j]].get("correct_answer_letter") or "").upper()
        if pred is None: m+=1
        elif pred==gold: c+=1
        else: i+=1
        det.append({"id":ids[b+j],"gold":gold,"pred":pred,"status":"correct" if pred==gold else ("null" if pred is None else "incorrect"),"tokens":tk})
n_=len(ids)
res={"mode":mode,"n":n_,"correct":c,"incorrect":i,"null":m,"acc":100*c/n_,"null_rate":100*m/n_,"avg_tok":sum(toks)/n_,"details":det}
json.dump(res,open(outpath,"w"),indent=1)
print(f"RESULT mode={mode} n={n_} correct={c} inc={i} null={m} acc={100*c/n_:.2f}% null_rate={100*m/n_:.1f}% avg_tok={sum(toks)/n_:.0f}", flush=True)

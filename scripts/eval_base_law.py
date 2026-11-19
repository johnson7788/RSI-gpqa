import sys, json, torch, pandas as pd
sys.path.insert(0,'.')
from transformers import AutoTokenizer, AutoModelForCausalLM
from train_rest import build_prompt, CLASSES, extract_charge, norm
m=sys.argv[1]; out=sys.argv[2]
val=pd.read_csv('data/out/rest_law_3b/val.csv')
tok=AutoTokenizer.from_pretrained(m, trust_remote_code=True); tok.padding_side="left"
if tok.pad_token is None: tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(m, torch_dtype=torch.bfloat16, device_map="auto"); model.eval()
prompts=[]
for _,r in val.iterrows():
    msgs=[{"role":"user","content":build_prompt(str(r['text'])[:1500], CLASSES)}]
    prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
correct=parsed=0; details=[]
for i in range(0,len(prompts),8):
    chunk=prompts[i:i+8]
    enc=tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=2048).to("cuda")
    n=enc["input_ids"].shape[1]
    gen=model.generate(**enc, max_new_tokens=256, do_sample=False, pad_token_id=tok.pad_token_id)
    for j,resp in enumerate(tok.batch_decode(gen[:, n:], skip_special_tokens=True)):
        row=val.iloc[i+j]; pred=extract_charge(resp)
        if pred is not None: parsed+=1
        if pred==row['label_n']: correct+=1
        details.append({"id":int(row['id']),"gold":row['label_n'],"pred":pred})
res={"n":len(val),"correct":correct,"parsed":parsed,"accuracy":100*correct/len(val),"parse_rate":100*parsed/len(val),"details":details}
json.dump(res, open(out,'w'), indent=1)
print(f"BASE EVAL n={res['n']} correct={correct} acc={res['accuracy']:.2f}% parse_rate={res['parse_rate']:.1f}%")

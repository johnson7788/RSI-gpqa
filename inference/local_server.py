#!/usr/bin/env python3
"""本地 Qwen3-8B 的 OpenAI 兼容推理服务（供 SIA 的 target agent 调用）。

支持 LoRA adapter 热切换：通过环境变量或 /reload 接口指定 adapter 路径，
这样串联实验中"每代权重更新后的模型"可以被同一个服务用不同 adapter 提供。
"""
import os, json, re, threading, time
from typing import Any
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

MODEL = os.environ.get("SERVE_MODEL", "/model/ModelScope/Qwen/Qwen3-8B")
PORT = int(os.environ.get("SERVE_PORT", "8000"))
MAX_NEW = int(os.environ.get("SERVE_MAX_NEW", "8192"))

app = FastAPI()
lock = threading.Semaphore(int(os.environ.get("SERVE_CONCURRENCY", "4")))
state: dict[str, Any] = {"model": None, "tok": None, "adapter": None}

def load(adapter: str | None):
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    state["model"], state["tok"], state["adapter"] = model, tok, adapter
    return f"loaded adapter={adapter}"

def to_chat(payload: dict) -> str:
    msgs = payload.get("messages") or []
    if len(msgs) == 1 and msgs[0].get("role") == "user":
        return state["tok"].apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    return state["tok"].apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)

@app.post("/v1/chat/completions")
async def chat(request: Request):
    payload = await request.json()
    with lock:
        tok, model = state["tok"], state["model"]
        text = to_chat(payload)
        enc = tok(text, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
        mn = int(payload.get("max_tokens") or MAX_NEW)
        temp = float(payload.get("temperature") or 0.0)
        kw = dict(max_new_tokens=mn, pad_token_id=tok.pad_token_id)
        if temp > 0:
            kw.update(do_sample=True, temperature=temp, top_p=float(payload.get("top_p") or 0.95))
        else:
            kw.update(do_sample=False)
        with torch.inference_mode():
            out = model.generate(**enc, **kw)
        resp_text = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return JSONResponse({
        "id": f"chatcmpl-{int(time.time()*1000)}", "object": "chat.completion",
        "created": int(time.time()), "model": payload.get("model", "qwen3-8b"),
        "choices": [{"index": 0, "message": {"role": "assistant", "content": resp_text},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": int(enc["input_ids"].shape[1]),
                  "completion_tokens": int(out.shape[1] - enc["input_ids"].shape[1]),
                  "total_tokens": int(out.shape[1])},
    })

@app.post("/reload")
async def reload_adapter(request: Request):
    body = await request.json()
    msg = load(body.get("adapter"))
    return {"status": "ok", "info": msg}

@app.get("/v1/models")
async def models():
    return {"data": [{"id": "qwen3-8b-local", "object": "model"}]}

@app.get("/health")
async def health():
    return {"status": "ok", "adapter": state["adapter"]}

if __name__ == "__main__":
    print(load(os.environ.get("SERVE_ADAPTER") or None), flush=True)
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")

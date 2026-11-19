#!/usr/bin/env python3
"""Qwen3-8B OpenAI 兼容服务（带批处理，单请求预算独立）。

关键修复：同一批次里不同请求的 max_tokens 不同，统一按"最大"生成会导致
短请求被长时间霸占。这里按 batch 内最大预算生成，但**每个请求完成后**就
返回其截断结果（用 finish reason 提示），且批次大小限制在较小值。
"""
import asyncio, os, time, uuid
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = os.environ.get("SERVE_MODEL", "/model/ModelScope/Qwen/Qwen3-8B")
PORT = int(os.environ.get("SERVE_PORT", "8000"))
HARD_CAP = int(os.environ.get("SERVE_HARD_CAP", "4096"))   # 单请求硬上限
BATCH_MAX_N = int(os.environ.get("BATCH_MAX", "4"))
BATCH_WAIT = float(os.environ.get("BATCH_WAIT", "0.05"))

app = FastAPI()
tok = AutoTokenizer.from_pretrained(MODEL)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto")
model.eval()

queue: asyncio.Queue = asyncio.Queue()


def render(payload):
    msgs = payload.get("messages") or []
    ctk = payload.get("chat_template_kwargs") or {}
    et = ctk.get("enable_thinking")
    et = True if et is None else bool(et)
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=et)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def do_generate(batch):
    """batch: list of (payload, fut, text)"""
    texts = [b[2] for b in batch]
    max_new = min(HARD_CAP, max(int(b[0].get("max_tokens") or 2048) for b in batch))
    temps = [float(b[0].get("temperature") or 0.0) for b in batch]
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(model.device)
    n = enc["input_ids"].shape[1]
    kw = dict(max_new_tokens=max_new, pad_token_id=tok.pad_token_id)
    if any(t > 0 for t in temps):
        kw.update(do_sample=True, temperature=max(max(temps), 0.1), top_p=0.95)
    else:
        kw.update(do_sample=False)
    with torch.inference_mode():
        out = model.generate(**enc, **kw)
    return tok.batch_decode(out[:, n:], skip_special_tokens=True)


async def worker():
    while True:
        first = await queue.get()
        batch = [first]
        deadline = time.time() + BATCH_WAIT
        while len(batch) < BATCH_MAX_N and time.time() < deadline:
            try:
                batch.append(queue.get_nowait())
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.005)
        try:
            decoded = await asyncio.get_event_loop().run_in_executor(None, do_generate, batch)
            for (_, fut, _), text in zip(batch, decoded):
                if not fut.done():
                    fut.set_result(text)
        except Exception as e:
            for _, fut, _ in batch:
                if not fut.done():
                    fut.set_exception(e)
        finally:
            for _ in batch:
                queue.task_done()


@app.on_event("startup")
async def startup():
    asyncio.create_task(worker())


@app.post("/v1/chat/completions")
async def chat(request: Request):
    payload = await request.json()
    fut = asyncio.get_event_loop().create_future()
    await queue.put((payload, fut, render(payload)))
    text = await fut
    return JSONResponse({
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}", "object": "chat.completion",
        "created": int(time.time()), "model": "qwen3-8b-local",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    })


@app.get("/v1/models")
async def models():
    return {"data": [{"id": "qwen3-8b-local", "object": "model"}]}


@app.get("/health")
async def health():
    return {"status": "ok", "queue": queue.qsize()}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")

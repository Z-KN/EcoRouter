"""Inference server for Qwen3-VL-4B-Instruct on the Hexagon NPU (GenieX/QAIRT).

Exposes an OpenAI-compatible /v1/chat/completions endpoint (streaming and
non-streaming, text + base64 images) plus /health and /metrics. Use this same
running server for both remote API calls and local prompting -- the NPU
context is single-owner, so don't also run `geniex chat` against the same
model while this server is up.

    .venv\\Scripts\\python.exe serve_qwen_vl.py [--host 0.0.0.0] [--port 8000]
"""
from __future__ import annotations

import argparse
import base64
import binascii
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from geniex import AutoModelForCausalLM

MODEL_ID = "ai-hub-models/Qwen3-VL-4B-Instruct"

state: dict[str, Any] = {
    "model": None,
    "lock": None,
    "started_at": 0.0,
    "request_count": 0,
    "total_decode_tokens": 0,
    "total_decode_time_s": 0.0,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio

    print(f"Loading {MODEL_ID} ...")
    t0 = time.perf_counter()
    state["model"] = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    state["lock"] = asyncio.Lock()
    state["started_at"] = time.time()
    print(f"Model loaded in {time.perf_counter() - t0:.2f}s -- serving.")
    yield
    state["model"].close()


app = FastAPI(title="Qwen3-VL-4B-Instruct NPU server", lifespan=lifespan)


class ChatMessage(BaseModel):
    role: str
    content: Any  # str, or OpenAI-style content-block list


class ChatCompletionRequest(BaseModel):
    model: str = MODEL_ID
    messages: list[ChatMessage]
    max_tokens: int = 512
    temperature: float = 0.0
    stream: bool = False


def _extract_text_and_images(content: Any) -> tuple[str, list[str]]:
    """Flatten OpenAI-style content (str or block list) into (text, image_paths).

    Base64 data-URL images are decoded to temp files -- geniex.generate()
    takes file paths, not raw bytes.
    """
    if isinstance(content, str):
        return content, []

    text_parts: list[str] = []
    image_paths: list[str] = []
    for block in content:
        btype = block.get("type")
        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "image_url":
            url = (block.get("image_url") or {}).get("url", "")
            if not url.startswith("data:"):
                raise HTTPException(400, "image_url must be a data: URI (base64) -- remote URL fetch is not supported")
            try:
                header, b64data = url.split(",", 1)
                raw = base64.b64decode(b64data)
            except (ValueError, binascii.Error) as exc:
                raise HTTPException(400, f"invalid base64 image data: {exc}") from exc
            suffix = ".png" if "png" in header else ".jpg"
            tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            tmp.write(raw)
            tmp.close()
            image_paths.append(tmp.name)
    return "\n".join(text_parts), image_paths


def _to_geniex_messages(messages: list[ChatMessage]) -> tuple[list[dict], list[str]]:
    geniex_messages = []
    all_images: list[str] = []
    for m in messages:
        if isinstance(m.content, str):
            geniex_messages.append({"role": m.role, "content": m.content})
            continue
        text, images = _extract_text_and_images(m.content)
        all_images.extend(images)
        content_blocks = [{"type": "text", "text": text}]
        content_blocks.extend({"type": "image", "image": p} for p in images)
        geniex_messages.append({"role": m.role, "content": content_blocks})
    return geniex_messages, all_images


@app.get("/health")
async def health():
    model = state["model"]
    if model is None:
        return JSONResponse({"status": "loading"}, status_code=503)
    return {
        "status": "healthy",
        "model": MODEL_ID,
        "uptime_s": round(time.time() - state["started_at"], 1),
        "requests_served": state["request_count"],
    }


@app.get("/metrics")
async def metrics():
    total_time = state["total_decode_time_s"]
    total_tok = state["total_decode_tokens"]
    return {
        "requests_served": state["request_count"],
        "total_decode_tokens": total_tok,
        "avg_decode_speed_tok_s": round(total_tok / total_time, 1) if total_time > 0 else None,
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    model = state["model"]
    geniex_messages, image_paths = _to_geniex_messages(req.messages)
    prompt = model.tokenizer.apply_chat_template(geniex_messages, add_generation_prompt=True)

    gen_kwargs = dict(max_new_tokens=req.max_tokens, temperature=req.temperature)
    if image_paths:
        gen_kwargs["images"] = image_paths

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    async def _cleanup():
        for p in image_paths:
            Path(p).unlink(missing_ok=True)

    if not req.stream:
        async with state["lock"]:
            out = model.generate(prompt, stream=False, **gen_kwargs)
        await _cleanup()
        state["request_count"] += 1
        state["total_decode_tokens"] += out.profile.generated_tokens
        state["total_decode_time_s"] += out.profile.decode_time / 1_000_000
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": out.text},
                    "finish_reason": "stop" if out.profile.stop_reason == "eos" else "length",
                }
            ],
            "usage": {
                "prompt_tokens": out.profile.prompt_tokens,
                "completion_tokens": out.profile.generated_tokens,
                "total_tokens": out.profile.prompt_tokens + out.profile.generated_tokens,
            },
            "quad_profile": {
                "ttft_ms": out.profile.ttft / 1000,
                "prefill_speed_tok_s": out.profile.prefill_speed,
                "decode_speed_tok_s": out.profile.decode_speed,
                "device": out.profile.device,
                "backend": out.profile.backend,
            },
        }

    async def event_stream():
        import json

        async with state["lock"]:
            streamer = model.generate(prompt, stream=True, **gen_kwargs)
            first = True
            for chunk in streamer:
                delta = {"role": "assistant", "content": chunk} if first else {"content": chunk}
                first = False
                payload = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": req.model,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                }
                yield f"data: {json.dumps(payload)}\n\n"
            out = streamer.output
            state["request_count"] += 1
            if out is not None:
                state["total_decode_tokens"] += out.profile.generated_tokens
                state["total_decode_time_s"] += out.profile.decode_time / 1_000_000
        await _cleanup()
        final = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": req.model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(final)}\n\ndata: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    print(f"Starting on http://{args.host}:{args.port} (reachable on the LAN at this machine's IP)")
    uvicorn.run(app, host=args.host, port=args.port)

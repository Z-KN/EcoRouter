"""Interactive local chat client for the Qwen3-VL-4B-Instruct server.

Talks to the running serve_qwen_vl.py over HTTP -- it does NOT load its own
copy of the model, so it's safe to use alongside remote API callers without
two processes fighting over the NPU/HTP context.

    .venv\\Scripts\\python.exe chat_client.py [--url http://localhost:8000]

Commands inside the chat: `/image <path>` attaches an image to your next
message, `/reset` clears history, `/quit` exits.
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import sys
import urllib.request

messages: list[dict] = []


def build_content(text: str, image_path: str | None) -> object:
    if not image_path:
        return text
    mime = mimetypes.guess_type(image_path)[0] or "image/png"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return [
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        {"type": "text", "text": text},
    ]


def stream_reply(url: str) -> str:
    payload = {"messages": messages, "max_tokens": 512, "stream": True}
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    full = []
    with urllib.request.urlopen(req, timeout=120) as resp:
        for line in resp:
            line = line.decode().strip()
            if not line.startswith("data: "):
                continue
            data = line[len("data: "):]
            if data == "[DONE]":
                break
            chunk = json.loads(data)["choices"][0]["delta"].get("content")
            if chunk:
                print(chunk, end="", flush=True)
                full.append(chunk)
    print()
    return "".join(full)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()

    try:
        with urllib.request.urlopen(f"{args.url}/health", timeout=5) as resp:
            print("Connected:", json.load(resp))
    except Exception as exc:
        print(f"Could not reach {args.url}/health -- is serve_qwen_vl.py running? ({exc})")
        sys.exit(1)

    pending_image: str | None = None
    print("Type your prompt. `/image <path>` attaches an image, `/reset` clears history, `/quit` exits.\n")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text == "/quit":
            break
        if text == "/reset":
            messages.clear()
            print("(history cleared)")
            continue
        if text.startswith("/image "):
            pending_image = text[len("/image "):].strip()
            print(f"(attached {pending_image} -- send your next message)")
            continue

        messages.append({"role": "user", "content": build_content(text, pending_image)})
        pending_image = None
        reply = stream_reply(args.url)
        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()

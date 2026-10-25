"""Manual check: the openai SDK against a running gateway. Calls the real provider; not for CI.

uv run uvicorn app.main:app            # with GROQ_API_KEY / GEMINI_API_KEY in .env
uv run python -m scripts.create_key create smoke           # once; then export the key it prints
GATEWAY_API_KEY=gw-... uv run python scripts/smoke_openai_sdk.py --model fast

Asks once, then the same again (an exact cache hit), then the same as a stream (a hit replayed
as SSE). Within CACHE_TTL_S of an earlier run the first call is a hit too: change --prompt.
"""

import argparse
import os
import time

from openai import OpenAI


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="fast", help="an alias from config/models.yaml")
    parser.add_argument("--prompt", default="Say hello in five words.")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("GATEWAY_API_KEY"),
        help="a gateway key from scripts/create_key.py (default: $GATEWAY_API_KEY)",
    )
    args = parser.parse_args()

    if not args.api_key:
        parser.error("set GATEWAY_API_KEY or pass --api-key")
    client = OpenAI(base_url=args.base_url, api_key=args.api_key, max_retries=0)
    # temperature 0, so the cache may answer.
    request = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "temperature": 0,
        "max_tokens": 50,
    }

    for label in ("first", "again"):
        started = time.perf_counter()
        raw = client.chat.completions.with_raw_response.create(**request)
        resp = raw.parse()
        print(f"[{label}] {_served(raw.headers, started)}")
        print(f"  model:  {resp.model}")
        print(f"  finish: {resp.choices[0].finish_reason}")
        print(f"  reply:  {resp.choices[0].message.content}")
        print(f"  tokens: {resp.usage.prompt_tokens} in, {resp.usage.completion_tokens} out")
        # The answer is stored after the response is sent: give it a moment to land.
        time.sleep(0.2)

    started = time.perf_counter()
    raw = client.chat.completions.with_raw_response.create(
        **request, stream=True, stream_options={"include_usage": True}
    )
    chunks = list(raw.parse())
    text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    usage = chunks[-1].usage
    print(f"[stream] {_served(raw.headers, started)}")
    print(f"  reply:  {text}")
    print(f"  tokens: {usage.prompt_tokens} in, {usage.completion_tokens} out")


def _served(headers, started: float) -> str:
    return (
        f"cache {headers.get('x-gateway-cache')}, "
        f"served by {headers.get('x-gateway-provider')} "
        f"(attempts {headers.get('x-gateway-attempts')}, "
        f"fallback {headers.get('x-gateway-fallback')}), "
        f"{(time.perf_counter() - started) * 1000:.1f} ms"
    )


if __name__ == "__main__":
    main()

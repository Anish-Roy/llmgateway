"""Manual check: the openai SDK against a running gateway. Calls the real provider; not for CI.

uv run uvicorn app.main:app            # with GROQ_API_KEY / GEMINI_API_KEY in .env
uv run python -m scripts.create_key create smoke           # once; then export the key it prints
GATEWAY_API_KEY=gw-... uv run python scripts/smoke_openai_sdk.py --model fast
"""

import argparse
import os

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
    raw = client.chat.completions.with_raw_response.create(
        model=args.model,
        messages=[{"role": "user", "content": args.prompt}],
        temperature=0,
        max_tokens=50,
    )
    resp = raw.parse()
    print(f"model:  {resp.model}")
    print(
        f"served: {raw.headers.get('x-gateway-provider')} "
        f"(attempts {raw.headers.get('x-gateway-attempts')}, "
        f"fallback {raw.headers.get('x-gateway-fallback')})"
    )
    print(f"finish: {resp.choices[0].finish_reason}")
    print(f"reply:  {resp.choices[0].message.content}")
    print(f"tokens: {resp.usage.prompt_tokens} in, {resp.usage.completion_tokens} out")


if __name__ == "__main__":
    main()

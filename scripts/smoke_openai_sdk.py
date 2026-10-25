"""Manual check: the openai SDK against a running gateway. Calls the real provider; not for CI.

uv run uvicorn app.main:app            # with GROQ_API_KEY in .env
uv run python scripts/smoke_openai_sdk.py
"""

import argparse

from openai import OpenAI


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="fast", help="an alias from config/models.yaml")
    parser.add_argument("--prompt", default="Say hello in five words.")
    args = parser.parse_args()

    # The gateway doesn't check keys until Phase 4, but the SDK requires one.
    client = OpenAI(base_url=args.base_url, api_key="unused-until-phase-4", max_retries=0)
    resp = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": args.prompt}],
        temperature=0,
        max_tokens=50,
    )
    print(f"model:  {resp.model}")
    print(f"reply:  {resp.choices[0].message.content}")
    print(f"tokens: {resp.usage.prompt_tokens} in, {resp.usage.completion_tokens} out")


if __name__ == "__main__":
    main()

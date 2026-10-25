"""Manual check: do real models keep the PII placeholders intact? Calls the real provider; not
for CI.

The gateway sends the model "[EMAIL_1]" in place of an email and puts the email back only where
the reply has "[EMAIL_1]" exactly. A model that rewrites it ("EMAIL_1", "\\[EMAIL\\_1\\]",
"[Email 1]") leaves the client a placeholder instead of the value. This asks prompts that want
the values repeated, plain and streamed, with the cache off, and counts for each value whether
it came back, was mangled, or was dropped.

uv run uvicorn app.main:app            # with GROQ_API_KEY / GEMINI_API_KEY in .env
GATEWAY_API_KEY=gw-... uv run python scripts/check_pii_live.py --model fast

With --note, a system message explains the placeholders first: to measure whether telling
the model helps, before the gateway does it.

The x-gateway-provider header says who answered. To check Gemini, run a second gateway with
GROQ_API_KEY blanked, so `fast` falls through to it, and point --base-url there:

GROQ_API_KEY= uv run uvicorn app.main:app --port 8001
GATEWAY_API_KEY=gw-... uv run python scripts/check_pii_live.py --base-url http://localhost:8001/v1
"""

import argparse
import os
import re
from collections import Counter

from openai import OpenAI

# Each prompt asks for its values back, so a reply without them is the model's doing.
PROMPTS = [
    (
        "Write a one-line reply to priya.sharma@example.com confirming Friday's meeting. "
        "Start with the address.",
        ["priya.sharma@example.com"],
    ),
    (
        "My number is 98765 43210. Repeat it back to me exactly, then say thanks.",
        ["98765 43210"],
    ),
    (
        "Draft an SMS to +91 9123456780 saying the parcel is out for delivery, and put "
        "the number at the top.",
        ["+91 9123456780"],
    ),
    (
        "Format this as a contact card: Ravi Kumar, ravi.k@corp.in, 7012345678.",
        ["ravi.k@corp.in", "7012345678"],
    ),
    (
        "Fill in: 'Payment of Rs 500 was charged to card 4111 1111 1111 1111.' Quote the "
        "whole sentence.",
        ["4111 1111 1111 1111"],
    ),
    (
        "Is 2345 6789 0124 a valid-looking Aadhaar number format? Quote it in your answer.",
        ["2345 6789 0124"],
    ),
    (
        "Write a short email from a@b.com to c@d.org asking for the report. Show the From "
        "and To lines.",
        ["a@b.com", "c@d.org"],
    ),
    (
        "Make a markdown table with columns Name, Email, Phone for: Neha, neha@mail.org, "
        "9988776655.",
        ["neha@mail.org", "9988776655"],
    ),
]

# Sent client-side with --note. The gateway masks nothing in it.
NOTE = (
    "Text in square brackets such as [EMAIL_1] or [PHONE_1] stands for a real value the user "
    "gave, hidden for privacy. Treat it as that value, and wherever the value belongs in your "
    "reply, copy the placeholder exactly, brackets included."
)

# What's left of a placeholder the gateway couldn't restore: any spelling of a kind and index.
_LEFTOVER = re.compile(r"(EMAIL|PHONE|AADHAAR|CARD)\W{0,3}_?\W{0,3}\d", re.IGNORECASE)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="fast", help="an alias from config/models.yaml")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("GATEWAY_API_KEY"),
        help="a gateway key from scripts/create_key.py (default: $GATEWAY_API_KEY)",
    )
    parser.add_argument("--note", action="store_true", help="explain placeholders first")
    parser.add_argument("--show", action="store_true", help="print every reply")
    args = parser.parse_args()

    if not args.api_key:
        parser.error("set GATEWAY_API_KEY or pass --api-key")
    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
        max_retries=0,
        default_headers={"x-gateway-cache": "off"},
    )

    counts: Counter[str] = Counter()
    providers: Counter[str] = Counter()
    for prompt, values in PROMPTS:
        request = {
            "model": args.model,
            "messages": [
                *([{"role": "system", "content": NOTE}] if args.note else []),
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 400,
        }
        for mode in ("plain", "stream"):
            if mode == "plain":
                raw = client.chat.completions.with_raw_response.create(**request)
                reply = raw.parse().choices[0].message.content or ""
            else:
                raw = client.chat.completions.with_raw_response.create(**request, stream=True)
                chunks = raw.parse()
                reply = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
            providers[raw.headers.get("x-gateway-provider", "?")] += 1
            leftover = _LEFTOVER.findall(reply)
            for value in values:
                if value in reply:
                    outcome = "restored"
                elif leftover:
                    outcome = "mangled"
                else:
                    outcome = "dropped"
                counts[outcome] += 1
                if outcome != "restored" or args.show:
                    print(f"[{mode}] {outcome}: {value!r}\n  {reply.strip()[:300]!r}\n")

    total = sum(counts.values())
    print(f"served by: {dict(providers)}")
    for outcome in ("restored", "mangled", "dropped"):
        print(f"{outcome:>9}: {counts[outcome]:>3} of {total} values")


if __name__ == "__main__":
    main()

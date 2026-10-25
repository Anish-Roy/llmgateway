from decimal import ROUND_HALF_UP, Decimal

from app.api.schemas import Usage
from app.routing.registry import Price

# request_logs.cost_usd is numeric(14, 10): round here so a row reads back as it was written.
COST_PLACES = Decimal("1e-10")
_PER_MILLION = Decimal(1_000_000)


def cost(price: Price, usage: Usage) -> Decimal:
    """USD for one call: prompt tokens at the input price, completion tokens (reasoning and
    thinking included) at the output price."""
    usd = (
        usage.prompt_tokens * price.input + usage.completion_tokens * price.output
    ) / _PER_MILLION
    return usd.quantize(COST_PLACES, rounding=ROUND_HALF_UP)

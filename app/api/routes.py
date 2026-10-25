from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response

from app.api.auth import require_api_key
from app.api.rate_limit_headers import rate_limit_headers
from app.api.schemas import ChatRequest, ChatResponse, Model, ModelList
from app.api.sse import sse_response
from app.auth.keys import ApiKey
from app.auth.rate_limit import RateLimiter, Reservation
from app.routing.registry import Target
from app.routing.router import ModelRouter, StreamRecord
from app.usage.tokens import estimate_usage

# Every /v1 route needs a key, /v1/models included, as on OpenAI. /health stays open.
router = APIRouter(prefix="/v1", dependencies=[Depends(require_api_key)])


def get_model_router(request: Request) -> ModelRouter:
    """The router built at startup. Tests swap it with `app.dependency_overrides`."""
    return request.app.state.model_router


def get_rate_limiter(request: Request) -> RateLimiter:
    return request.app.state.rate_limiter


@router.post("/chat/completions", response_model=ChatResponse)
async def chat_completions(
    body: ChatRequest,
    response: Response,
    model_router: Annotated[ModelRouter, Depends(get_model_router)],
    key: Annotated[ApiKey, Depends(require_api_key)],
    limiter: Annotated[RateLimiter, Depends(get_rate_limiter)],
):
    # A 429 here, before any provider is called. Admitted requests hold their estimated
    # tokens until they're settled with the real usage below.
    reservation = limiter.reserve(key, body)
    limit_headers = rate_limit_headers(reservation.status)

    if body.stream:
        # The router picks a provider before the response starts, so a failure before the
        # first token is still a normal JSON error with its own status, and the headers below
        # can say which provider is streaming.
        try:
            stream = await model_router.stream(body)
        except BaseException:
            # No provider answered: the tokens go back (the request still counts).
            reservation.settle(0)
            raise
        headers = _gateway_headers(stream.target, stream.attempts, stream.fallback)
        return sse_response(
            stream,
            include_usage=_wants_usage(body),
            headers={**headers, **limit_headers},
            on_finish=lambda record: _settle_stream(reservation, record),
        )
    try:
        routed = await model_router.chat(body)
    except BaseException:
        reservation.settle(0)
        raise
    usage = routed.response.usage
    if not usage.total_tokens:
        # A provider that reported no usage still used tokens.
        reply = routed.response.choices[0].message.content if routed.response.choices else ""
        usage = estimate_usage(body.messages, reply or "")
    reservation.settle(usage.total_tokens)
    response.headers.update(_gateway_headers(routed.target, routed.attempts, routed.fallback))
    response.headers.update(limit_headers)
    return routed.response


def _settle_stream(reservation: Reservation, record: StreamRecord) -> None:
    # The record has the provider's usage, or an estimate if the stream was cut short. If
    # closing the stream itself failed and left no usage, charge what was reserved.
    reservation.settle(record.usage.total_tokens if record.usage else reservation.tokens)


def _gateway_headers(target: Target, attempts: int, fallback: bool) -> dict[str, str]:
    return {
        # The registry name ("groq", "flaky"...), so the fallback demo shows which entry answered.
        "x-gateway-provider": target.provider,
        "x-gateway-attempts": str(attempts),
        "x-gateway-fallback": "true" if fallback else "false",
    }


def _wants_usage(body: ChatRequest) -> bool:
    options = (body.model_extra or {}).get("stream_options")
    return isinstance(options, dict) and options.get("include_usage") is True


@router.get("/models", response_model=ModelList)
async def list_models(model_router: Annotated[ModelRouter, Depends(get_model_router)]):
    # The aliases clients can send as `model`, not the vendor models behind them. Not rate
    # limited: it costs no provider call, and OpenAI doesn't count it either.
    return ModelList(data=[Model(id=alias) for alias in model_router.available_aliases()])

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response

from app.api.auth import require_api_key
from app.api.rate_limit_headers import rate_limit_headers
from app.api.schemas import ChatRequest, ChatResponse, Model, ModelList, Usage
from app.api.sse import sse_response
from app.auth.keys import ApiKey
from app.auth.rate_limit import RateLimiter, Reservation
from app.cache.key import CACHE_HEADER, storable
from app.cache.store import CacheEntry, ResponseCache
from app.routing.registry import Target
from app.routing.router import ModelRouter, Routed, RoutedStream
from app.usage.recorder import Recording, UsageRecorder
from app.usage.tokens import estimate_usage

logger = logging.getLogger(__name__)

# Every /v1 route needs a key, /v1/models included, as on OpenAI. /health stays open.
router = APIRouter(prefix="/v1", dependencies=[Depends(require_api_key)])


def get_model_router(request: Request) -> ModelRouter:
    """The router built at startup. Tests swap it with `app.dependency_overrides`."""
    return request.app.state.model_router


def get_rate_limiter(request: Request) -> RateLimiter:
    return request.app.state.rate_limiter


def get_usage_recorder(request: Request) -> UsageRecorder:
    return request.app.state.usage_recorder


def get_response_cache(request: Request) -> ResponseCache:
    return request.app.state.response_cache


@router.post("/chat/completions", response_model=ChatResponse)
async def chat_completions(
    body: ChatRequest,
    request: Request,
    response: Response,
    model_router: Annotated[ModelRouter, Depends(get_model_router)],
    key: Annotated[ApiKey, Depends(require_api_key)],
    limiter: Annotated[RateLimiter, Depends(get_rate_limiter)],
    recorder: Annotated[UsageRecorder, Depends(get_usage_recorder)],
    cache: Annotated[ResponseCache, Depends(get_response_cache)],
):
    # Every request from here on gets a row in request_logs, written after it ends: a reply,
    # a closed stream, or an error.
    recording = recorder.start(key, body)
    # A 429 here, before any provider is called. Admitted requests hold their estimated
    # tokens until they're settled with the real usage below.
    try:
        reservation = limiter.reserve(key, body)
    except BaseException as e:
        recording.failed(e)
        raise
    limit_headers = rate_limit_headers(reservation.status)

    if body.stream:
        # The router picks a provider before the response starts, so a failure before the
        # first token is still a normal JSON error with its own status, and the headers below
        # can say which provider is streaming.
        try:
            stream = await model_router.stream(body)
        except BaseException as e:
            # No provider answered: the tokens go back (the request still counts).
            reservation.settle(0)
            recording.failed(e)
            raise
        headers = _gateway_headers(stream.target, stream.attempts, stream.fallback)
        return sse_response(
            stream,
            include_usage=_wants_usage(body),
            headers={**headers, **limit_headers},
            on_finish=lambda _: _stream_finished(stream, reservation, recording),
        )
    # None when the request isn't cacheable or the client turned the cache off.
    cache_key = cache.key_for(body, request.headers.get(CACHE_HEADER))
    try:
        entry = await cache.get(cache_key) if cache_key is not None else None
    except BaseException as e:
        # The lookup never raises; this is the client leaving while it ran.
        reservation.settle(0)
        recording.failed(e)
        raise
    if entry is not None:
        # No provider call: the reserved tokens go back (the request still counts).
        reservation.settle(0)
        recording.cached(entry)
        response.headers.update(_cache_hit_headers(entry))
        response.headers.update(limit_headers)
        return _served_again(entry.response)

    try:
        routed = await model_router.chat(body)
    except BaseException as e:
        reservation.settle(0)
        recording.failed(e)
        raise
    usage = routed.response.usage
    estimated = not usage.total_tokens
    if estimated:
        # A provider that reported no usage still used tokens.
        reply = routed.response.choices[0].message.content if routed.response.choices else ""
        usage = estimate_usage(body.messages, reply or "")
    reservation.settle(usage.total_tokens)
    recording.replied(routed, usage, estimated)
    if cache_key is not None:
        _keep(cache, recorder, cache_key, body, routed, usage)
    response.headers.update(_gateway_headers(routed.target, routed.attempts, routed.fallback))
    response.headers.update(limit_headers)
    response.headers[CACHE_HEADER] = "miss" if cache_key is not None else "bypass"
    return routed.response


def _keep(
    cache: ResponseCache,
    recorder: UsageRecorder,
    key: bytes,
    body: ChatRequest,
    routed: Routed,
    usage: Usage,
) -> None:
    """Stores a reply for the next identical request, if it's worth serving again. Never
    fails the reply, which the client is about to get."""
    if not storable(routed.response):
        return
    try:
        cost_usd = recorder.cost(routed.target, usage)
    except Exception:
        logger.warning("could not price a reply for the cache", exc_info=True)
        return
    cache.put(
        key,
        alias=body.model,
        response=routed.response,
        provider=routed.target.provider,
        model=routed.target.model,
        cost_usd=cost_usd,
    )


def _served_again(stored: ChatResponse) -> ChatResponse:
    # A fresh id and timestamp, as for any new completion; the answer and its usage as stored.
    return ChatResponse.model_validate(stored.model_dump(exclude={"id", "created"}))


def _cache_hit_headers(entry: CacheEntry) -> dict[str, str]:
    # Who first answered; no provider was called this time.
    target = Target(provider=entry.provider, model=entry.model)
    return {**_gateway_headers(target, 0, False), CACHE_HEADER: "exact"}


def _stream_finished(stream: RoutedStream, reservation: Reservation, recording: Recording) -> None:
    # The record has the provider's usage, or an estimate if the stream was cut short. If
    # closing the stream itself failed and left no usage, charge what was reserved.
    record = stream.record
    reservation.settle(record.usage.total_tokens if record.usage else reservation.tokens)
    recording.streamed(stream)


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

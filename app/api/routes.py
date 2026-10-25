import logging
from collections.abc import Callable
from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response

from app.api.auth import require_api_key
from app.api.rate_limit_headers import rate_limit_headers
from app.api.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    Choice,
    Model,
    ModelList,
    Usage,
)
from app.api.sse import sse_replay, sse_response
from app.auth.keys import ApiKey
from app.auth.rate_limit import RateLimiter, Reservation
from app.cache.key import CACHE_HEADER, storable
from app.cache.semantic import SEMANTIC
from app.cache.semantic_store import SemanticCache, SemanticLookup
from app.cache.store import CacheEntry, ResponseCache
from app.guardrails.pii import explain_placeholders, mask_request, restore_response
from app.routing.registry import Target
from app.routing.router import ModelRouter, RoutedStream
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


def get_semantic_cache(request: Request) -> SemanticCache:
    return request.app.state.semantic_cache


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
    semantic: Annotated[SemanticCache, Depends(get_semantic_cache)],
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

    # PII is masked before anything else sees the request: from here on the caches (their
    # keys and what they store), the providers and token estimates get placeholders only.
    # Only what's sent to the client gets the real values back, in copies: the objects the
    # caches store keep the placeholders. A masked request also tells the model what the
    # placeholders are, or some models drop them.
    masked = mask_request(body)
    body, values = masked.request, masked.values
    if values:
        body = explain_placeholders(body)
    recording.masked(masked.types)

    # Both kinds of request share entries: the key leaves `stream` out. None when the request
    # isn't cacheable or the client turned the cache off.
    cache_header = request.headers.get(CACHE_HEADER)
    cache_key = cache.key_for(body, cache_header)
    # Only with `x-gateway-cache: semantic`, on a cacheable single-turn request, matched among
    # this API key's own entries. None otherwise.
    lookup = semantic.lookup_for(body, cache_header, key.id)
    hit_status = "exact"
    try:
        # An alias that's gone from the registry (404), or has no provider here any more
        # (503), fails as it would without a cache: entries stored for it aren't served.
        model_router.targets(body.model)
        entry = await cache.get(cache_key) if cache_key is not None else None
        # Exact first: it's cheaper, and it can't be wrong.
        if entry is None and lookup is not None:
            entry = await semantic.get(lookup)
            hit_status = SEMANTIC
    except BaseException as e:
        # The lookups never raise: this is the alias check, or the client leaving.
        reservation.settle(0)
        recording.failed(e)
        raise
    if entry is not None:
        # No provider call: the reserved tokens go back (the request still counts).
        reservation.settle(0)
        recording.cached(entry, hit_status)
        headers = {**_cache_hit_headers(entry, hit_status), **limit_headers}
        if body.stream:
            return sse_replay(
                entry.response, include_usage=_wants_usage(body), headers=headers, values=values
            )
        response.headers.update(headers)
        return restore_response(_served_again(entry.response), values)
    cache_status = "miss" if cache_key is not None else "bypass"

    def keep(answer: ChatResponse, target: Target, usage: Usage) -> None:
        if cache_key is not None or lookup is not None:
            _keep(cache, semantic, recorder, cache_key, lookup, body.model, answer, target, usage)

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
            headers={**headers, **limit_headers, CACHE_HEADER: cache_status},
            values=values,
            on_finish=lambda _: _stream_finished(stream, reservation, recording, keep),
        )

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
    keep(routed.response, routed.target, usage)
    response.headers.update(_gateway_headers(routed.target, routed.attempts, routed.fallback))
    response.headers.update(limit_headers)
    response.headers[CACHE_HEADER] = cache_status
    return restore_response(routed.response, values)


def _keep(
    cache: ResponseCache,
    semantic: SemanticCache,
    recorder: UsageRecorder,
    key: bytes | None,
    lookup: SemanticLookup | None,
    alias: str,
    answer: ChatResponse,
    target: Target,
    usage: Usage,
) -> None:
    """Stores an answer for the next identical request, and for the next similar one if this
    request opted in to the semantic cache, if it's worth serving again. Never fails the
    request: the client has the answer, or is about to."""
    if not storable(answer):
        return
    try:
        cost_usd = recorder.cost(target, usage)
    except Exception:
        logger.warning("could not price an answer for the cache", exc_info=True)
        return
    stored = {
        "alias": alias,
        "response": answer,
        "provider": target.provider,
        "model": target.model,
        "cost_usd": cost_usd,
    }
    if key is not None:
        cache.put(key, **stored)
    if lookup is not None:
        semantic.put(lookup, **stored)


def _served_again(stored: ChatResponse) -> ChatResponse:
    # A fresh id and timestamp, as for any new completion; the answer and its usage as stored.
    return ChatResponse.model_validate(stored.model_dump(exclude={"id", "created"}))


def _cache_hit_headers(entry: CacheEntry, status: str) -> dict[str, str]:
    # Who first answered; no provider was called this time.
    target = Target(provider=entry.provider, model=entry.model)
    return {**_gateway_headers(target, 0, False), CACHE_HEADER: status}


def _stream_finished(
    stream: RoutedStream,
    reservation: Reservation,
    recording: Recording,
    keep: Callable[[ChatResponse, Target, Usage], None],
) -> None:
    # The record has the provider's usage, or an estimate if the stream was cut short. If
    # closing the stream itself failed and left no usage, charge what was reserved.
    record = stream.record
    reservation.settle(record.usage.total_tokens if record.usage else reservation.tokens)
    recording.streamed(stream)
    # Only an answer that streamed to the end is kept: a cut one would be served cut forever.
    if record.outcome == "completed" and record.usage is not None:
        keep(_streamed_answer(stream), stream.target, record.usage)


def _streamed_answer(stream: RoutedStream) -> ChatResponse:
    """A finished stream as the reply a non-streamed request would have got, so either kind
    of request can be served from it. Text only: a tool call's deltas aren't kept, and its
    finish_reason (tool_calls) keeps it out of the cache."""
    record = stream.record
    return ChatResponse(
        model=stream.target.model,
        choices=[
            Choice(
                message=ChatMessage(role="assistant", content=record.text),
                finish_reason=record.finish_reason,
            )
        ],
        usage=record.usage,
    )


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

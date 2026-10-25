from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response

from app.api.schemas import ChatRequest, ChatResponse, Model, ModelList
from app.api.sse import sse_response
from app.routing.registry import Target
from app.routing.router import ModelRouter

router = APIRouter(prefix="/v1")


def get_model_router(request: Request) -> ModelRouter:
    """The router built at startup. Tests swap it with `app.dependency_overrides`."""
    return request.app.state.model_router


@router.post("/chat/completions", response_model=ChatResponse)
async def chat_completions(
    body: ChatRequest,
    response: Response,
    model_router: Annotated[ModelRouter, Depends(get_model_router)],
):
    if body.stream:
        # The router picks a provider before the response starts, so a failure before the
        # first token is still a normal JSON error with its own status, and the headers below
        # can say which provider is streaming.
        stream = await model_router.stream(body)
        headers = _gateway_headers(stream.target, stream.attempts, stream.fallback)
        return sse_response(stream, include_usage=_wants_usage(body), headers=headers)
    routed = await model_router.chat(body)
    response.headers.update(_gateway_headers(routed.target, routed.attempts, routed.fallback))
    return routed.response


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
    # The aliases clients can send as `model`, not the vendor models behind them.
    return ModelList(data=[Model(id=alias) for alias in model_router.available_aliases()])

from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response

from app.api.errors import error_response
from app.api.schemas import ChatRequest, ChatResponse, Model, ModelList
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
        # Streaming lands in Phase 3. Fail loudly rather than return a non-stream body.
        return error_response(400, "Streaming is not supported yet", param="stream")
    routed = await model_router.chat(body)
    # The registry name ("groq", "flaky"...), so the fallback demo shows which entry answered.
    response.headers["x-gateway-provider"] = routed.target.provider
    response.headers["x-gateway-attempts"] = str(routed.attempts)
    response.headers["x-gateway-fallback"] = "true" if routed.fallback else "false"
    return routed.response


@router.get("/models", response_model=ModelList)
async def list_models(model_router: Annotated[ModelRouter, Depends(get_model_router)]):
    # The aliases clients can send as `model`, not the vendor models behind them.
    return ModelList(data=[Model(id=alias) for alias in model_router.available_aliases()])

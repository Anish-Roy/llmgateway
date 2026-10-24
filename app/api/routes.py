from typing import Annotated

from fastapi import APIRouter, Depends, Request

from app.api.errors import error_response
from app.api.schemas import ChatRequest, ChatResponse
from app.providers.base import Provider

router = APIRouter(prefix="/v1")


def get_provider(request: Request) -> Provider:
    """The provider built at startup. Tests swap it with `app.dependency_overrides`."""
    return request.app.state.provider


@router.post("/chat/completions", response_model=ChatResponse)
async def chat_completions(body: ChatRequest, provider: Annotated[Provider, Depends(get_provider)]):
    if body.stream:
        # Streaming lands in Phase 3. Fail loudly rather than return a non-stream body.
        return error_response(400, "Streaming is not supported yet", param="stream")
    return await provider.chat(body)

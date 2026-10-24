import httpx

from app.api.schemas import ChatRequest, ChatResponse
from app.providers.base import ProviderError

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class GroqProvider:
    """Groq speaks the OpenAI format already, so this adapter mostly forwards and maps errors."""

    name = "groq"

    def __init__(self, client: httpx.AsyncClient, api_key: str | None) -> None:
        # The client is shared and owned by the app (see main.lifespan), not created per request.
        self._client = client
        self._api_key = api_key

    async def chat(self, request: ChatRequest) -> ChatResponse:
        if not self._api_key:
            raise ProviderError(
                500, "GROQ_API_KEY is not set", type="server_error", provider=self.name
            )

        body = request.model_dump(exclude_none=True)
        try:
            resp = await self._client.post(
                f"{GROQ_BASE_URL}/chat/completions",
                json=body,
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
        except httpx.TimeoutException as e:
            raise ProviderError(504, "Groq timed out", type="timeout", provider=self.name) from e
        except httpx.TransportError as e:
            raise ProviderError(
                502, f"Could not reach Groq: {e}", type="api_connection_error", provider=self.name
            ) from e

        if resp.status_code >= 400:
            raise _error_from_response(resp)
        return ChatResponse.model_validate(resp.json())


def _error_from_response(resp: httpx.Response) -> ProviderError:
    # Groq's error bodies are OpenAI-shaped: {"error": {"message", "type", "code", "param"}}.
    try:
        err = resp.json().get("error") or {}
    except ValueError:
        err = {}
    if resp.status_code in (401, 403):
        # The gateway's own Groq key was rejected. Passing a 401 on would tell the client
        # *their* key is wrong, so report it as an upstream failure instead.
        return ProviderError(
            502,
            "Upstream provider rejected the gateway's credentials",
            type="api_error",
            provider="groq",
        )
    return ProviderError(
        resp.status_code,
        err.get("message") or resp.text or f"Groq returned {resp.status_code}",
        type=err.get("type") or "api_error",
        code=err.get("code"),
        param=err.get("param"),
        provider="groq",
    )

from typing import Annotated

from fastapi import Depends, Header, Request

from app.auth.authenticator import Authenticator
from app.auth.keys import ApiKey


def get_authenticator(request: Request) -> Authenticator:
    """The authenticator built at startup. Tests swap it with `app.dependency_overrides`."""
    return request.app.state.authenticator


async def require_api_key(
    request: Request,
    authenticator: Annotated[Authenticator, Depends(get_authenticator)],
    authorization: Annotated[str | None, Header()] = None,
) -> ApiKey:
    """The caller's key, or a 401. Kept on `request.state` for the rate limiter and logging."""
    key = await authenticator.authenticate(authorization)
    request.state.api_key = key
    return key

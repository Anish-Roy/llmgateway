"""Turn every error into OpenAI's error JSON, so SDK clients raise the right exception type."""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from app.api.rate_limit_headers import duration, rate_limit_headers, retry_after_headers
from app.api.schemas import ErrorDetail, ErrorResponse
from app.auth.authenticator import AuthError, KeyStoreUnavailable
from app.auth.rate_limit import RateLimitExceeded
from app.providers.base import ProviderError
from app.routing.router import ModelNotFoundError

logger = logging.getLogger(__name__)

_TYPE_BY_STATUS = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    429: "rate_limit_exceeded",
}


def error_body(
    status: int,
    message: str,
    type: str | None = None,
    param: str | None = None,
    code: str | None = None,
) -> ErrorResponse:
    detail = ErrorDetail(
        message=message,
        type=type or _TYPE_BY_STATUS.get(status, "api_error"),
        param=param,
        code=code,
    )
    return ErrorResponse(error=detail)


def error_response(
    status: int,
    message: str,
    type: str | None = None,
    param: str | None = None,
    code: str | None = None,
) -> JSONResponse:
    body = error_body(status, message, type, param, code)
    return JSONResponse(body.model_dump(), status_code=status)


async def _provider_error(request: Request, exc: ProviderError) -> JSONResponse:
    if exc.provider:
        logger.warning("provider %s failed: %s %s", exc.provider, exc.status, exc.message)
    else:
        logger.warning("request failed: %s %s", exc.status, exc.message)
    return error_response(exc.status, exc.message, exc.type, exc.param, exc.code)


async def _model_not_found(request: Request, exc: ModelNotFoundError) -> JSONResponse:
    # OpenAI's answer to an unknown model: a 404 the SDK raises as NotFoundError.
    return error_response(
        404, str(exc), "invalid_request_error", param="model", code="model_not_found"
    )


async def _auth_error(request: Request, exc: AuthError) -> JSONResponse:
    # OpenAI's shape for a bad key: the SDK raises AuthenticationError on the 401 status.
    response = error_response(401, exc.message, "invalid_request_error", code="invalid_api_key")
    response.headers["WWW-Authenticate"] = "Bearer"
    return response


async def _key_store_unavailable(request: Request, exc: KeyStoreUnavailable) -> JSONResponse:
    return error_response(503, str(exc), "api_error", code="service_unavailable")


async def _rate_limited(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    # OpenAI's shape for its own limits: `type` says which limit ("requests" or "tokens"),
    # which also tells this 429 apart from a provider's (type "rate_limit_exceeded").
    key = request.state.api_key
    limits = {"requests": "requests per minute (RPM)", "tokens": "tokens per minute (TPM)"}
    bucket = {"requests": exc.status.requests, "tokens": exc.status.tokens}
    reached = "; ".join(
        f"{limits[name]}: limit {bucket[name].limit}, remaining {bucket[name].remaining}"
        for name in exc.limited
    )
    message = (
        f"Rate limit reached on API key {key.key_prefix}... for {reached}. "
        f"Please try again in {duration(exc.retry_after_s)}."
    )
    response = error_response(429, message, exc.limited[0], code="rate_limit_exceeded")
    response.headers.update(rate_limit_headers(exc.status))
    response.headers.update(retry_after_headers(exc.retry_after_s))
    return response


async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    # FastAPI's default is a 422 with {"detail": [...]}. OpenAI answers bad input with a 400.
    first = exc.errors()[0]
    loc = [str(p) for p in first.get("loc", ()) if p != "body"]
    param = ".".join(loc) or None
    message = f"{param}: {first['msg']}" if param else first["msg"]
    return error_response(400, message, param=param)


async def _http_error(request: Request, exc: HTTPException) -> JSONResponse:
    return error_response(exc.status_code, str(exc.detail))


async def _unhandled_error(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled error")
    return error_response(500, "Internal gateway error", "server_error")


def install_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(ProviderError, _provider_error)
    app.add_exception_handler(ModelNotFoundError, _model_not_found)
    app.add_exception_handler(AuthError, _auth_error)
    app.add_exception_handler(KeyStoreUnavailable, _key_store_unavailable)
    app.add_exception_handler(RateLimitExceeded, _rate_limited)
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(HTTPException, _http_error)
    app.add_exception_handler(Exception, _unhandled_error)

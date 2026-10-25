"""Turn every error into OpenAI's error JSON, so SDK clients raise the right exception type."""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from app.api.schemas import ErrorDetail, ErrorResponse
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
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(HTTPException, _http_error)
    app.add_exception_handler(Exception, _unhandled_error)

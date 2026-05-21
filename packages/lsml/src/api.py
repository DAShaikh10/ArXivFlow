"""
FastAPI backend providing Label Studio compatible endpoints.

This module exposes a FastAPI `app` with endpoints that mirror the label-studio-ml backend API:
`/predict`, `/setup`, `/health`.

`@author`: DAShaikh10
"""

import asyncio
import os
import timeit
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import Any, Optional

import anyio
from fastapi import Body, FastAPI
from fastapi.requests import Request
from fastapi.responses import JSONResponse

import wandb

from src.lib.models import LabelStudioMLBase

from .utils import logger, set_basic_auth, verify_basic_auth


def init_app(
    model_class: LabelStudioMLBase, basic_auth_user: Optional[str] = None, basic_auth_pass: Optional[str] = None
) -> FastAPI:
    """
    Initialize FastAPI app with the provided model class and optional basic auth.
    Compatibility shim like original `label-studio-ml`.

    Args:
        model_class (LabelStudioMLBase): The ML backend model class to use for predictions.
        basic_auth_user (str, optional): Optional username for basic authentication.
        basic_auth_pass (str, optional): Optional password for basic authentication.

    Returns:
        FastAPI: The initialized FastAPI app instance.

    Raises:
        ValueError: If the provided `model_class` is not a subclass of `LabelStudioMLBase`.
    """

    if not issubclass(model_class, LabelStudioMLBase):
        raise ValueError("Inference class should be the subclass of " + LabelStudioMLBase.__class__.__name__)

    app.state.model_class = model_class
    if basic_auth_user and basic_auth_pass:
        app.state.basic_auth = (basic_auth_user, basic_auth_pass)
        set_basic_auth(app.state.basic_auth)

    return app


@asynccontextmanager
async def lifespan(_: FastAPI):
    """
    FastAPI lifespan context manager to handle startup and shutdown events, including WandB initialization and cleanup.
    """

    # Initialize off the main thread to prevent startup blocking.
    await asyncio.to_thread(
        wandb.init,
        project=os.getenv("WANDB_PROJECT_NAME", "arxivflow"),
        name="lsml_inference_server",
        job_type="inference",
    )
    yield

    await asyncio.to_thread(wandb.finish)


app = FastAPI(lifespan=lifespan)


app = FastAPI(
    title="Label Studio ML Backend",
    description="A FastAPI backend for Label Studio ML integration.",
    lifespan=lifespan,
)


@app.middleware("http")
async def check_auth(request: Request, call_next) -> JSONResponse:
    """
    FastAPI middleware to check for optional basic authentication on incoming requests.

    Args:
        request (Request): The incoming HTTP request.
        call_next: The next middleware or route handler to call if authentication is successful.

    Returns:
        JSONResponse: A response indicating unauthorized access if authentication fails,
                      or the response from the next handler if successful.
    """

    if getattr(app.state, "basic_auth", None) is not None:
        valid = await verify_basic_auth(request)
        if not valid:
            return JSONResponse(
                content="Unauthorized",
                status_code=HTTPStatus.UNAUTHORIZED,
                headers={"WWW-Authenticate": 'Basic realm="Login required"'},
            )

    req_data = await request.body()

    logger.debug("Request headers: %s", request.headers)
    logger.debug("Request body: %s", req_data)
    wandb.log({"request_headers": dict(request.headers)})
    wandb.log({"request_body": req_data.decode("utf-8") if isinstance(req_data, bytes) else str(req_data)})

    # If BASIC_AUTH is None, or no header was provided at all, pass through.
    start_time = timeit.default_timer()
    response = await call_next(request)
    elapsed = timeit.default_timer() - start_time

    logger.debug("Request processing time: %.4f seconds", elapsed)
    logger.debug("Response status: %s", response.status_code)
    logger.debug("Response headers: %s", response.headers)

    wandb.log({"request_processing_time_seconds": elapsed})
    wandb.log({"response_status": response.status_code})
    wandb.log({"response_headers": dict(response.headers)})

    return response


@app.get("/")
@app.get("/health")
async def health() -> JSONResponse:
    """
    Return a minimal health check and the configured model class name.
    """

    return JSONResponse({"status": "UP", "model_class": app.state.model_class.__name__})


@app.post("/setup")
async def setup(request: Request) -> JSONResponse:
    """
    Setup model metadata for a project, instantiate the model for inference and return `model_version`.

    Args:
        request (Request): The incoming HTTP request.

    Returns:
        JSONResponse: A response containing the model version information.

    Raises:
        HTTPException: If the authentication fails.
    """

    # Read request body parameters.
    body = await request.json()
    project_id = str(body.get("project")).split(".", 1)[0]
    label_config = body.get("schema")
    extra_params = body.get("extra_params") or {}

    # Instantiate and load the model into memory.
    if getattr(app.state, "model", None) is None:
        app.state.model = await anyio.to_thread.run_sync(
            app.state.model_class, project_id, label_config, **extra_params
        )
        await anyio.to_thread.run_sync(app.state.model.load)

    return JSONResponse({"model_version": getattr(app.state.model, "hf_model_name")})


@app.post("/predict")
async def predict(body: dict = Body(...)) -> JSONResponse:
    """
    Run prediction for a batch of tasks and return Label Studio formatted results.

    Args:
        body (dict): The request body containing tasks, label config, project info and optional parameters
        _auth: Dependency for optional basic authentication.

    Returns:
        JSONResponse: A response containing the prediction results formatted for Label Studio.
    """

    tasks: str | None = body.get("tasks")
    params: dict[str, Any] = body.get("params", {}) or {}
    context: dict[str, Any] = params.pop("context", {}) if isinstance(params, dict) else {}

    result = await anyio.to_thread.run_sync(app.state.model.predict, tasks, context)

    # Normalize response similar to original label-studio-ml behavior.
    if hasattr(result, "model_dump"):
        response_obj = result.model_dump()
    else:
        response_obj = result

    result = response_obj or []

    if isinstance(result, dict):
        result = result.get("predictions", result.get("results", []))

    wandb.log({"prediction_results": result})

    return JSONResponse({"results": result})

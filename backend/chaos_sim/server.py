"""Chaos simulator API. Run with uv run python -m backend.chaos_sim.server."""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from functools import partial
from logging import getLogger
from os import getenv

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import ValidationError

from backend.chaos_sim.chaos import ChaosWrapper
from backend.chaos_sim.cpu_env import CpuPickEnv
from backend.chaos_sim.env import Env
from backend.chaos_sim.libero_env import LiberoEnvAdapter, LiberoUnavailable, libero_available
from backend.chaos_sim.media import Recording
from backend.chaos_sim.session import Session
from backend.shared.schemas import PolicyHealthResponse, SimHealthResponse

DEFAULT_POLICY_URL = "http://127.0.0.1:8001"
logger = getLogger(__name__)


def configured_env() -> tuple[Callable[[], Env], str]:
    kind = getenv("CHAOS_SIM_ENV", "cpu")
    if kind == "libero":
        if libero_available():
            return partial(
                LiberoEnvAdapter.from_task,
                suite=getenv("LIBERO_SUITE", "libero_spatial"),
                task_id=int(getenv("LIBERO_TASK_ID", "0")),
                initial_state_id=int(getenv("LIBERO_INITIAL_STATE_ID", "0")),
            ), LiberoEnvAdapter.__name__
        logger.warning("Optional LIBERO dependencies are unavailable. Using CpuPickEnv.")
    elif kind != "cpu":
        raise ValueError(f"Unknown CHAOS_SIM_ENV {kind!r}")
    return CpuPickEnv, CpuPickEnv.__name__


def close_env(env: Env) -> None:
    close = getattr(env, "close", None)
    if close is not None:
        close()


def create_app(
    policy_url: str | None = None,
    *,
    env_factory: Callable[[], Env] | None = None,
    env_type: str | None = None,
    policy_transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    policy_url = policy_url or getenv("POLICY_SERVER_URL", DEFAULT_POLICY_URL)
    if env_factory is None:
        env_factory, env_type = configured_env()
    elif env_type is None and isinstance(env_factory, type):
        env_type = env_factory.__name__

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with httpx.AsyncClient(
            base_url=policy_url,
            timeout=httpx.Timeout(5.0, connect=1.0),
            transport=policy_transport,
            trust_env=False,
        ) as client:
            app.state.policy_client = client
            app.state.latest_recording = None
            app.state.env_type = env_type
            yield

    app = FastAPI(title="Break-My-VLA Chaos Simulator", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
        expose_headers=["Content-Disposition"],
    )

    @app.get("/health")
    async def health() -> SimHealthResponse:
        try:
            response = await app.state.policy_client.get("/health")
            response.raise_for_status()
            policy = PolicyHealthResponse.model_validate_json(response.content)
        except (httpx.HTTPError, ValidationError) as error:
            raise HTTPException(status_code=503, detail="Policy health unavailable") from error
        if app.state.env_type is None:
            env = env_factory()
            try:
                app.state.env_type = type(env).__name__
            finally:
                close_env(env)
        return SimHealthResponse(**policy.model_dump(), env=app.state.env_type)

    @app.websocket("/ws/session")
    async def websocket_session(websocket: WebSocket) -> None:
        await websocket.accept()
        env = None
        try:
            env = env_factory()
            app.state.env_type = type(env).__name__
            session = Session(websocket, ChaosWrapper(env), app.state.policy_client)
            await session.run()
        except WebSocketDisconnect:
            pass
        except LiberoUnavailable as error:
            logger.exception("LIBERO cannot start")
            await websocket.close(code=1011, reason=str(error)[:80])
        finally:
            if env is not None:
                close_env(env)

    @app.post(
        "/api/export_gif",
        response_class=Response,
        responses={
            200: {"content": {"image/gif": {}}},
            404: {"description": "No recorded frames"},
        },
    )
    async def export_gif() -> Response:
        recording: Recording | None = app.state.latest_recording
        data = await recording.export() if recording is not None else None
        if data is None:
            raise HTTPException(status_code=404, detail="No recorded frames")
        return Response(
            data,
            media_type="image/gif",
            headers={"Content-Disposition": 'attachment; filename="break-my-vla.gif"'},
        )

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, ws="websockets-sansio")

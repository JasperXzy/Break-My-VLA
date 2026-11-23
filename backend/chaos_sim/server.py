"""Chaos simulator API. Run with uv run python -m backend.chaos_sim.server."""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from os import getenv

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from backend.chaos_sim.chaos import ChaosWrapper
from backend.chaos_sim.cpu_env import CpuPickEnv
from backend.chaos_sim.env import Env
from backend.chaos_sim.media import Recording
from backend.chaos_sim.session import Session

DEFAULT_POLICY_URL = "http://127.0.0.1:8001"


def create_app(
    policy_url: str | None = None,
    *,
    env_factory: Callable[[], Env] = CpuPickEnv,
    policy_transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    policy_url = policy_url or getenv("POLICY_SERVER_URL", DEFAULT_POLICY_URL)

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
            yield

    app = FastAPI(title="Break-My-VLA Chaos Simulator", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
        expose_headers=["Content-Disposition"],
    )

    @app.websocket("/ws/session")
    async def websocket_session(websocket: WebSocket) -> None:
        await websocket.accept()
        session = Session(websocket, ChaosWrapper(env_factory()), app.state.policy_client)
        try:
            await session.run()
        except WebSocketDisconnect:
            pass

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

"""CPU-only mock policy. Run with uv run python -m backend.policy_servers.mock_server."""

import argparse
from asyncio import sleep
from dataclasses import dataclass
from hashlib import blake2b
from time import perf_counter

import numpy as np
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from numpy.typing import NDArray

from backend.shared.schemas import (
    PolicyActRequest,
    PolicyActResponse,
    PolicyHealthResponse,
    Precision,
    policy_backend_header,
)

CHUNK_SIZE = 8
BASE_LATENCY_MS = 8.0


@dataclass(frozen=True, slots=True)
class PrecisionProfile:
    ms_per_step: float
    noise: float


PROFILES: dict[Precision, PrecisionProfile] = {
    "bf16": PrecisionProfile(ms_per_step=4.3, noise=0.015),
    "fp8": PrecisionProfile(ms_per_step=3.1, noise=0.045),
    "int8": PrecisionProfile(ms_per_step=2.6, noise=0.075),
    "nvfp4": PrecisionProfile(ms_per_step=1.7, noise=0.130),
}

app = FastAPI(title="Break-My-VLA Mock Policy")
app.state.realtime = True


@app.exception_handler(RequestValidationError)
async def validation_error(_request: Request, error: RequestValidationError) -> JSONResponse:
    # The raw input can contain nonfinite numbers, which JSON cannot echo in an error.
    detail = [{key: issue[key] for key in ("loc", "msg", "type")} for issue in error.errors()]
    return JSONResponse(status_code=422, content={"detail": detail})


def pick_action(proprio: NDArray[np.float64]) -> NDArray[np.float64]:
    action = np.zeros(7)
    if proprio.size != 7:
        return action

    # The CPU demo gives the mock hand xyz, block xyz, and a grasp flag.
    hand, block = proprio[:3], proprio[3:6]
    if proprio[6] > 0.5:
        target = np.array([hand[0], hand[1], 0.60])
        action[6] = 0.9
    else:
        target = block.copy()
        if np.linalg.norm((block - hand)[:2]) > 0.065:
            target[2] = max(0.28, block[2] + 0.14)
        action[6] = 0.9 if np.linalg.norm(block - hand) < 0.09 else -0.9

    action[:3] = np.clip((target - hand) * 5.5, -0.9, 0.9)
    return action


def noisy_chunk(request: PolicyActRequest, noise: float) -> list[list[float]]:
    proprio = np.asarray(request.proprio, dtype="<f8")
    digest = blake2b(proprio.tobytes(), digest_size=8).digest()
    rng = np.random.default_rng(int.from_bytes(digest, "little"))
    chunk = np.tile(pick_action(np.clip(proprio, -1.0, 1.0)), (CHUNK_SIZE, 1))
    chunk[:, :6] *= np.linspace(1.0, 0.55, CHUNK_SIZE)[:, None]
    jitter = noise + (10 - request.denoise_steps) * 0.024
    chunk += rng.normal(0.0, jitter, chunk.shape)
    return np.clip(chunk, -1.0, 1.0).tolist()


@app.get("/health")
async def health(response: Response) -> PolicyHealthResponse:
    for mode in PROFILES:
        response.headers[policy_backend_header(mode)] = "tensorrt" if mode == "nvfp4" else "pytorch"
    return PolicyHealthResponse(model="mock", precisions=tuple(PROFILES), device="cpu")


@app.post("/act")
async def act(request: PolicyActRequest, http_request: Request) -> PolicyActResponse:
    profile = PROFILES[request.precision]
    delay_ms = BASE_LATENCY_MS + profile.ms_per_step * request.denoise_steps
    started_at = perf_counter()
    realtime = http_request.app.state.realtime
    if realtime:
        await sleep(delay_ms / 1000)
    action_chunk = noisy_chunk(request, profile.noise)
    return PolicyActResponse(
        action_chunk=action_chunk,
        infer_ms=(perf_counter() - started_at) * 1000 if realtime else delay_ms,
        vram_mb=0.0,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument(
        "--instant", action="store_true", help="Report simulated latency without sleep"
    )
    args = parser.parse_args()
    app.state.realtime = not args.instant
    uvicorn.run(app, host="127.0.0.1", port=args.port, access_log=False)

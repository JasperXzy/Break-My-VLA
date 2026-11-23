"""Common LIBERO preprocessing, precision switching, and HTTP policy endpoints."""

from base64 import b64decode
from binascii import Error
from io import BytesIO
from threading import Lock
from time import perf_counter
from typing import Protocol

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from PIL import Image

from backend.policy_servers.quantization import LinearWeights, available_precisions
from backend.shared.schemas import (
    PolicyActRequest,
    PolicyActResponse,
    PolicyHealthResponse,
    Precision,
)

STATE_SIZE = 8


class PolicyEngine(Protocol):
    name: str
    device: object
    precisions: tuple[Precision, ...]

    def act(
        self, request: PolicyActRequest, main: np.ndarray, wrist: np.ndarray
    ) -> PolicyActResponse: ...


class LeRobotEngine:
    denoise_field: str
    quantized_projections: tuple[str, ...]

    def __init__(self, *, checkpoint: str, device: str, seed: int) -> None:
        import torch

        self.name = checkpoint
        self.torch = torch
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("LeRobot VLA inference requires a CUDA device")
        self.capability = torch.cuda.get_device_capability(self.device)
        self.precisions = available_precisions(
            self.capability, has_fp8_dtype=hasattr(torch, "float8_e4m3fn")
        )
        self._seed = seed
        self._lock = Lock()

    def _set_policy(self, model, processors, camera_keys: tuple[str, str]) -> None:
        self.model = model
        self.preprocessor, self.postprocessor = processors
        self._main_key, self._wrist_key = camera_keys
        self._weights = LinearWeights(
            model, self.torch, self.device, projections=self.quantized_projections
        )
        config = model.config
        generator = self.torch.Generator(device=self.device).manual_seed(self._seed)
        self._noise = self.torch.randn(
            (1, config.chunk_size, config.max_action_dim), device=self.device, generator=generator
        )

    def act(
        self, request: PolicyActRequest, main: np.ndarray, wrist: np.ndarray
    ) -> PolicyActResponse:
        torch = self.torch
        with self._lock, torch.inference_mode():
            self._weights.select(request.precision)
            setattr(self.model.config, self.denoise_field, request.denoise_steps)
            self.model.reset()
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            started = perf_counter()
            observation = self.preprocessor(
                {
                    self._main_key: torch.tensor(main).permute(2, 0, 1) / 255.0,
                    self._wrist_key: torch.tensor(wrist).permute(2, 0, 1) / 255.0,
                    "observation.state": torch.tensor(request.proprio, dtype=torch.float32),
                    "task": request.prompt,
                }
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                chunk = self.model.predict_action_chunk(observation, noise=self._noise)
            actions = self.postprocessor(chunk.float()).squeeze(0).tolist()
            torch.cuda.synchronize(self.device)
            return PolicyActResponse(
                action_chunk=actions,
                infer_ms=(perf_counter() - started) * 1000,
                vram_mb=torch.cuda.max_memory_allocated(self.device) / 1024**2,
            )


def decode_image(encoded: str) -> np.ndarray:
    with Image.open(BytesIO(b64decode(encoded, validate=True))) as image:
        return np.asarray(image.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR))


def policy_app(engine: PolicyEngine, *, title: str) -> FastAPI:
    app = FastAPI(title=title)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, error: RequestValidationError) -> JSONResponse:
        detail = [{key: issue[key] for key in ("loc", "msg", "type")} for issue in error.errors()]
        return JSONResponse(status_code=422, content={"detail": detail})

    @app.get("/health")
    async def health() -> PolicyHealthResponse:
        return PolicyHealthResponse(
            model=engine.name, precisions=engine.precisions, device=str(engine.device)
        )

    @app.post("/act")
    def act(request: PolicyActRequest) -> PolicyActResponse:
        if request.precision not in engine.precisions:
            raise HTTPException(
                status_code=422, detail=f"Supported precisions: {', '.join(engine.precisions)}"
            )
        if len(request.proprio) != STATE_SIZE:
            raise HTTPException(status_code=422, detail="Expected the 8-element LIBERO state")
        try:
            main, wrist = decode_image(request.rgb_main_b64), decode_image(request.rgb_wrist_b64)
        except (Error, OSError, ValueError, Image.DecompressionBombError) as error:
            raise HTTPException(
                status_code=422, detail="Expected Base64-encoded RGB images"
            ) from error
        return engine.act(request, main, wrist)

    return app

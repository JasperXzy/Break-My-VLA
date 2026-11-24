"""Common LIBERO preprocessing, precision switching, and HTTP policy endpoints."""

from base64 import b64decode
from binascii import Error
from contextlib import nullcontext
from io import BytesIO
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Protocol

import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from PIL import Image

from backend.policy_servers.quantization import LinearWeights, available_precisions
from backend.policy_servers.trt_engine import (
    ActionComponent,
    EngineManifest,
    load_tensorrt_backends,
)
from backend.shared.schemas import (
    PolicyActRequest,
    PolicyActResponse,
    PolicyBackend,
    PolicyHealthResponse,
    Precision,
    policy_backend_header,
)

STATE_SIZE = 8


class PolicyEngine(Protocol):
    name: str
    device: object
    precisions: tuple[Precision, ...]

    def act(
        self, request: PolicyActRequest, main: np.ndarray, wrist: np.ndarray
    ) -> PolicyActResponse: ...


class GPUUnavailable(RuntimeError):
    """The optional GPU policy runtime is unavailable."""


class CudaPolicyEngine:
    model_id: str
    trt_component: ActionComponent
    _tensorrt = None
    _nvfp4 = None

    def __init__(self, *, checkpoint: str, device: str, seed: int) -> None:
        try:
            import torch
        except (ImportError, OSError) as error:
            raise GPUUnavailable("Install CUDA-enabled PyTorch in the policy runtime") from error

        self.name = checkpoint
        self.torch = torch
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise GPUUnavailable("VLA inference requires a CUDA device")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.capability = torch.cuda.get_device_capability(self.device)
        self.precisions = available_precisions(
            self.capability, has_fp8_dtype=hasattr(torch, "float8_e4m3fn")
        )
        self._seed = seed
        self._lock = Lock()

    def load_tensorrt(self, directory: Path, *, revision: str | None) -> None:
        expected = EngineManifest(
            self.model_id, self.name, revision, "bf16", self.capability, self.trt_component
        )
        self._tensorrt = load_tensorrt_backends(directory, expected, device=self.device)
        if "nvfp4" in self._tensorrt and "nvfp4" not in self.precisions:
            self.precisions += ("nvfp4",)

    def precision_backends(self) -> dict[Precision, PolicyBackend]:
        tensorrt = self._tensorrt or {}
        return {
            mode: "tensorrt" if mode in tensorrt else "pytorch"
            for mode in self.precisions
            if mode != "nvfp4" or mode in tensorrt  # A legacy factory owns its implementation.
        }

    def act(
        self, request: PolicyActRequest, main: np.ndarray, wrist: np.ndarray
    ) -> PolicyActResponse:
        with self._lock:
            backend = (self._tensorrt or {}).get(request.precision)
            if backend is None and request.precision == "nvfp4" and self._nvfp4 is not None:
                return self._nvfp4(request, main, wrist)
            return self._act(request, main, wrist, backend)

    def _act(self, request, main, wrist, backend) -> PolicyActResponse:
        torch = self.torch
        with torch.inference_mode():
            self._weights.select("bf16" if backend is not None else request.precision)
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            started = perf_counter()
            replacement = backend.installed(self.model) if backend is not None else nullcontext()
            try:
                with replacement:
                    actions = self._predict_actions(request, main, wrist)
            finally:
                torch.cuda.synchronize(self.device)
            return PolicyActResponse(
                action_chunk=actions,
                infer_ms=(perf_counter() - started) * 1000,
                vram_mb=torch.cuda.max_memory_allocated(self.device) / 1024**2,
            )


class LeRobotEngine(CudaPolicyEngine):
    denoise_field: str
    quantized_projections: tuple[str, ...]
    trt_component = ActionComponent("model.action_out_proj", ("input",))

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

    def _predict_actions(self, request: PolicyActRequest, main: np.ndarray, wrist: np.ndarray):
        torch = self.torch
        setattr(self.model.config, self.denoise_field, request.denoise_steps)
        self.model.reset()
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
        return self.postprocessor(chunk.float()).squeeze(0).tolist()


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
    async def health(response: Response) -> PolicyHealthResponse:
        if backends := getattr(engine, "precision_backends", None):
            for mode, backend in backends().items():
                response.headers[policy_backend_header(mode)] = backend
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

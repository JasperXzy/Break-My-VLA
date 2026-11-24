"""CUDA TensorRT action components and backend-independent action metrics."""

import json
import logging
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from typing import Literal, TypedDict

import numpy as np
from numpy.typing import ArrayLike

TRTPrecision = Literal["bf16", "fp8", "nvfp4"]
TRT_PRECISIONS: tuple[TRTPrecision, ...] = ("bf16", "fp8", "nvfp4")
MANIFEST = "manifest.json"
ONNX_FILE = "action.onnx"
PLAN_FILE = "action.engine"
OUTPUT_NAME = "output"
LOGGER = logging.getLogger(__name__)


class ActionAlignment(TypedDict):
    action_l1_vs_bf16: float
    action_mse_vs_bf16: float
    action_max_abs_diff_vs_bf16: float
    action_cos_vs_bf16: float


def compute_action_alignment(reference: ArrayLike, candidate: ArrayLike) -> ActionAlignment:
    """Compare two nonempty [horizon, 7] chunks in LIBERO action units."""
    ref, act = (np.asarray(chunk, dtype=np.float64) for chunk in (reference, candidate))
    if ref.shape != act.shape or ref.ndim != 2 or ref.shape[1] != 7 or not ref.shape[0]:
        raise ValueError("Expected matching nonempty action chunks of shape (horizon, 7)")
    if not np.isfinite(ref).all() or not np.isfinite(act).all():
        raise ValueError("Expected finite action chunk values")

    scale_ref = np.max(np.abs(ref))
    scale_act = np.max(np.abs(act))
    if scale_ref and scale_act:
        r, a = ref.ravel() / scale_ref, act.ravel() / scale_act
        cosine = np.dot(r / np.linalg.norm(r), a / np.linalg.norm(a))
    else:
        cosine = float(scale_ref == scale_act)
    with np.errstate(over="ignore"):
        error = np.abs(act - ref)
        mse = np.mean(np.square(error))
    if not np.isfinite(mse):
        raise ValueError("Action chunk error exceeds the representable range")
    return {
        "action_l1_vs_bf16": float(np.mean(error)),
        "action_mse_vs_bf16": float(mse),
        "action_max_abs_diff_vs_bf16": float(np.max(error)),
        "action_cos_vs_bf16": float(np.clip(cosine, -1.0, 1.0)),
    }


def precision_supported(precision: TRTPrecision, capability: tuple[int, int]) -> bool:
    if precision == "nvfp4":
        return capability == (12, 0)
    return capability >= ((8, 9) if precision == "fp8" else (8, 0))


@dataclass(frozen=True)
class ActionComponent:
    path: str
    inputs: tuple[str, ...]
    sequence_inputs: tuple[str, ...] = ()

    def bind(self, args: tuple, kwargs: dict) -> tuple:
        names = self.inputs[len(args) :]
        return (*args, *(kwargs[name] for name in names))


@dataclass(frozen=True)
class EngineManifest:
    model: str
    checkpoint: str
    revision: str | None
    precision: TRTPrecision
    capability: tuple[int, int]
    component: ActionComponent
    version: int = 1

    def as_json(self) -> dict:
        return json.loads(json.dumps(asdict(self)))


class TensorRTActionBackend:
    """Run one serialized action component with PyTorch CUDA buffers."""

    def __init__(self, path: Path, *, device, component: ActionComponent) -> None:
        import tensorrt as trt
        import torch

        self.torch = torch
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("TensorRT action components require CUDA")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.component = component
        self._lock = Lock()
        self._stream = None
        with torch.cuda.device(self.device):
            self._logger = trt.Logger(trt.Logger.WARNING)
            trt.init_libnvinfer_plugins(self._logger, "")
            self._runtime = trt.Runtime(self._logger)
            self._engine = self._runtime.deserialize_cuda_engine(path.read_bytes())
            if self._engine is None:
                raise RuntimeError(f"TensorRT could not deserialize {path}")
            self._context = self._engine.create_execution_context()
            if self._context is None:
                raise RuntimeError(f"TensorRT could not create a context for {path}")

        engine = self._engine
        names = {engine.get_tensor_name(i) for i in range(engine.num_io_tensors)}
        inputs = {name for name in names if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT}
        if inputs != set(component.inputs) or names - inputs != {OUTPUT_NAME}:
            raise ValueError(f"Unexpected TensorRT bindings in {path}: {sorted(names)}")
        self._dtypes = {}
        types = {
            trt.DataType.FLOAT: torch.float32,
            trt.DataType.HALF: torch.float16,
            trt.DataType.BF16: torch.bfloat16,
            trt.DataType.INT32: torch.int32,
            trt.DataType.INT64: torch.int64,
            trt.DataType.BOOL: torch.bool,
        }
        for name in names:
            dtype = types.get(engine.get_tensor_dtype(name))
            if dtype is None or engine.get_tensor_location(name) != trt.TensorLocation.DEVICE:
                raise ValueError(f"Unsupported TensorRT buffer for {name!r} in {path}")
            if engine.get_tensor_format(name) != trt.TensorFormat.LINEAR:
                raise ValueError(f"Expected linear TensorRT I/O for {name!r} in {path}")
            self._dtypes[name] = dtype

    def run(self, inputs: Mapping):
        """Enqueue on the PyTorch stream and return a tensor on the same device."""
        if inputs.keys() != self._dtypes.keys() - {OUTPUT_NAME}:
            raise ValueError(f"Expected TensorRT inputs {self.component.inputs}")
        torch = self.torch
        with self._lock, torch.cuda.device(self.device):
            stream = torch.cuda.current_stream(self.device)
            if self._stream is not None and stream != self._stream:
                self._stream.synchronize()
            self._stream = stream
            context = self._context
            buffers = {}
            for name, tensor in inputs.items():
                buffer = tensor.to(device=self.device, dtype=self._dtypes[name]).contiguous()
                buffers[name] = buffer
                shape = tuple(buffer.shape)
                if not context.set_input_shape(name, shape):
                    raise ValueError(f"TensorRT input {name!r} rejected shape {shape}")
                buffer.record_stream(stream)
            shape = tuple(context.get_tensor_shape(OUTPUT_NAME))
            if not shape or min(shape) < 1:
                raise RuntimeError(f"TensorRT returned an unresolved action output shape {shape}")
            output = torch.empty(shape, device=self.device, dtype=self._dtypes[OUTPUT_NAME])
            buffers[OUTPUT_NAME] = output
            for name, buffer in buffers.items():
                if not context.set_tensor_address(name, buffer.data_ptr()):
                    raise RuntimeError(f"TensorRT could not bind {name!r}")
            if not context.execute_async_v3(stream_handle=stream.cuda_stream):
                raise RuntimeError("TensorRT action component execution failed")
            return output

    @contextmanager
    def installed(self, model):
        """Replace the component for one serialized policy call and restore it on exit."""
        parent_path, _, name = self.component.path.rpartition(".")
        parent = model.get_submodule(parent_path)
        original = getattr(parent, name)
        backend = self

        class ComponentModule(self.torch.nn.Module):
            def forward(self, *args, **kwargs):
                values = backend.component.bind(args, kwargs)
                inputs = dict(zip(backend.component.inputs, values, strict=True))
                return backend.run(inputs).to(dtype=backend.torch.bfloat16)

        setattr(parent, name, ComponentModule().eval())
        try:
            yield
        finally:
            setattr(parent, name, original)


def load_tensorrt_backends(directory: Path, expected: EngineManifest, *, device) -> dict:
    """Load matching plans while leaving available PyTorch tiers usable."""
    backends = {}
    for mode in TRT_PRECISIONS:
        path = directory / mode
        if not path.exists():
            continue
        try:
            if not precision_supported(mode, expected.capability):
                raise ValueError(f"{mode} is unsupported on CUDA capability {expected.capability}")
            manifest = json.loads((path / MANIFEST).read_text(encoding="utf-8"))
            if manifest != expected.as_json() | {"precision": mode}:
                raise ValueError("engine manifest does not match the model or CUDA capability")
            backends[mode] = TensorRTActionBackend(
                path / PLAN_FILE, device=device, component=expected.component
            )
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            LOGGER.warning("TensorRT %s unavailable for %s (%s)", mode, expected.model, error)
    return backends

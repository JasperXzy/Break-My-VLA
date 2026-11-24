import json
import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from backend.policy_servers import trt_engine
from backend.policy_servers.trt_engine import (
    MANIFEST,
    PLAN_FILE,
    ActionComponent,
    EngineManifest,
    TensorRTActionBackend,
    compute_action_alignment,
    load_tensorrt_backends,
    precision_supported,
)

COMPONENT = ActionComponent("model.action_out_proj", ("input",))


def test_action_alignment_uses_every_action_and_dimension():
    bf16 = np.ones((2, 7))
    candidate = bf16.copy()
    candidate[0, 0] = 3
    candidate[1, 6] = 0
    assert compute_action_alignment(bf16, candidate) == pytest.approx(
        {
            "action_l1_vs_bf16": 3 / 14,
            "action_mse_vs_bf16": 5 / 14,
            "action_max_abs_diff_vs_bf16": 2,
            "action_cos_vs_bf16": 15 / (14 * 21) ** 0.5,
        }
    )
    assert compute_action_alignment(bf16, -bf16)["action_cos_vs_bf16"] == pytest.approx(-1)
    zeros = np.zeros_like(bf16)
    assert compute_action_alignment(zeros, zeros)["action_cos_vs_bf16"] == 1
    assert compute_action_alignment(bf16, zeros)["action_cos_vs_bf16"] == 0


@pytest.mark.parametrize(
    "reference,candidate",
    [
        ([], []),
        ([[0] * 6], [[0] * 6]),
        ([[0] * 7], [[0] * 7] * 2),
        ([[float("nan")] * 7], [[0] * 7]),
        ([[1e300] * 7], [[0] * 7]),
    ],
)
def test_invalid_action_alignment_samples_fail(reference, candidate):
    with pytest.raises(ValueError, match="action chunk|Action chunk"):
        compute_action_alignment(reference, candidate)


def test_precision_support_follows_gpu_capability():
    assert not precision_supported("bf16", (7, 5))
    assert precision_supported("bf16", (8, 0))
    assert not precision_supported("fp8", (8, 0))
    assert precision_supported("fp8", (8, 9))
    assert not precision_supported("nvfp4", (8, 9))
    assert not precision_supported("nvfp4", (10, 0))
    assert precision_supported("nvfp4", (12, 0))


def test_backend_load_constrains_manifest_device_model_and_optional_deps(tmp_path, monkeypatch):
    expected = EngineManifest("smolvla", "checkpoint", "revision", "bf16", (8, 9), COMPONENT)
    for mode in ("bf16", "fp8", "nvfp4"):
        tier = tmp_path / mode
        tier.mkdir()
        manifest = expected.as_json() | {"precision": mode}
        if mode == "fp8":
            manifest["checkpoint"] = "other-model"
        (tier / MANIFEST).write_text(json.dumps(manifest))
    backend = Mock()
    monkeypatch.setattr(trt_engine, "TensorRTActionBackend", backend)
    assert load_tensorrt_backends(tmp_path, expected, device="cuda:0") == {
        "bf16": backend.return_value,
    }
    backend.assert_called_once_with(
        tmp_path / "bf16" / PLAN_FILE, component=COMPONENT, device="cuda:0"
    )
    backend.side_effect = ModuleNotFoundError("tensorrt")
    assert load_tensorrt_backends(tmp_path, expected, device="cuda:0") == {}


def test_tensor_rt_binds_shapes_buffers_and_current_stream(monkeypatch, tmp_path):
    context = Mock()
    context.get_tensor_shape.return_value = (1, 8, 7)
    engine = Mock(num_io_tensors=2)
    engine.get_tensor_name.side_effect = ["input", "output"]
    engine.get_tensor_mode.side_effect = lambda name: name
    engine.get_tensor_dtype.return_value = "bf16"
    engine.get_tensor_location.return_value = "device"
    engine.get_tensor_format.return_value = "linear"
    engine.create_execution_context.return_value = context
    trt = SimpleNamespace(
        Logger=Mock(WARNING=1),
        Runtime=lambda _: SimpleNamespace(
            deserialize_cuda_engine=lambda _: engine,
        ),
        init_libnvinfer_plugins=Mock(),
        TensorIOMode=SimpleNamespace(INPUT="input"),
        TensorLocation=SimpleNamespace(DEVICE="device"),
        TensorFormat=SimpleNamespace(LINEAR="linear"),
        DataType=SimpleNamespace(
            FLOAT="fp32", HALF="fp16", BF16="bf16", INT32="int32", INT64="int64", BOOL="bool"
        ),
    )
    device = SimpleNamespace(type="cuda", index=0)
    stream = Mock(cuda_stream=170)
    cuda = SimpleNamespace(
        is_available=lambda: True,
        device=lambda _: nullcontext(),
        current_stream=Mock(return_value=stream),
    )
    output = Mock()
    output.data_ptr.return_value = 42
    torch = SimpleNamespace(
        device=lambda _: device,
        cuda=cuda,
        empty=Mock(return_value=output),
        float32="fp32",
        float16="fp16",
        bfloat16="bf16",
        int32="int32",
        int64="int64",
        bool="bool",
    )
    monkeypatch.setitem(sys.modules, "tensorrt", trt)
    monkeypatch.setitem(sys.modules, "torch", torch)
    path = tmp_path / PLAN_FILE
    path.write_bytes(b"mock TensorRT plan")
    backend = TensorRTActionBackend(path, device="cuda:0", component=COMPONENT)

    input_tensor, buffer = Mock(), Mock(shape=(1, 8, 32))
    input_tensor.to.return_value.contiguous.return_value = buffer
    buffer.data_ptr.return_value = 24
    assert backend.run({"input": input_tensor}) is output
    input_tensor.to.assert_called_once_with(device=device, dtype="bf16")
    buffer.record_stream.assert_called_once_with(stream)
    context.set_input_shape.assert_called_once_with("input", (1, 8, 32))
    context.set_tensor_address.assert_any_call("input", 24)
    context.set_tensor_address.assert_any_call("output", 42)
    context.execute_async_v3.assert_called_once_with(stream_handle=170)
    context.get_tensor_shape.return_value = (-1, 8, 7)
    cuda.current_stream.return_value = Mock(cuda_stream=171)
    with pytest.raises(RuntimeError, match="unresolved"):
        backend.run({"input": input_tensor})
    stream.synchronize.assert_called_once()


def test_tensor_rt_replacement_restores_model_after_failure():
    backend = TensorRTActionBackend.__new__(TensorRTActionBackend)
    backend.component = COMPONENT

    class Module:
        def eval(self):
            return self

    backend.torch = SimpleNamespace(nn=SimpleNamespace(Module=Module), bfloat16="bf16")
    backend.run = Mock(side_effect=RuntimeError("inference failed"))
    original = object()
    parent = SimpleNamespace(action_out_proj=original)
    model = SimpleNamespace(get_submodule=Mock(return_value=parent))
    with pytest.raises(RuntimeError, match="inference failed"), backend.installed(model):
        parent.action_out_proj.forward("observation")
    assert parent.action_out_proj is original
    backend.run.assert_called_once_with({"input": "observation"})

import json
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from backend.policy_servers.trt_engine import MANIFEST, ONNX_FILE, PLAN_FILE, ActionComponent
from scripts import export_tensorrt

ROOT = Path(__file__).resolve().parents[1]


def test_export_cli_mock_runs_without_gpu_packages(tmp_path):
    output = tmp_path / "exports"
    inputs = export_tensorrt.observations(None, seed=8)
    observation_path = tmp_path / "observations.jsonl"
    observation_path.write_text("\n".join(sample.model_dump_json() for sample in inputs))

    # A separate interpreter exercises the CLI and fails on any optional GPU import.
    command = """
import builtins
import runpy
from pathlib import Path

original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'tensorrt', 'modelopt', 'gr00t', 'onnx', 'lerobot'}:
        raise AssertionError(f'Unexpected optional dependency {name}')
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
runpy.run_path(str(Path('scripts/export_tensorrt.py')), run_name='__main__')
"""
    run = subprocess.run(
        [
            sys.executable,
            "-c",
            command,
            "--mock",
            "--model",
            "groot",
            "--iterations",
            "4",
            "--output",
            str(output),
            "--calibration",
            str(observation_path),
            "--observations",
            str(observation_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    directory = output / "groot"
    report = json.loads((directory / "profile.json").read_text())
    assert "TensorRT profile" in run.stdout
    assert report["mode"] == "cpu_mock"
    assert report["reference"] == {"backend": "pytorch", "precision": "bf16"}
    assert report["alignment_samples"] == 2
    assert set(report["tiers"]) == {"bf16", "fp8", "nvfp4"}
    for mode, tier in report["tiers"].items():
        assert tier["status"] == "mock"
        assert tier["backend"] == "tensorrt"
        assert tier["latency"]["source"] == "simulated"
        assert tier["latency"]["samples"] == 4
        assert tier["latency"]["p99_ms"] > 0
        assert tier["alignment"]["action_mse_vs_bf16"] >= 0
        assert tier["alignment"]["action_max_abs_diff_vs_bf16"] >= 0
        assert tier["calibration_samples"] == (0 if mode == "bf16" else 2)
        assert json.loads((directory / tier["onnx"]).read_text())["stage"] == "export"
        assert json.loads((directory / tier["engine"]).read_text())["stage"] == "build"
        assert not (directory / mode / MANIFEST).exists()
    assert report["tiers"]["bf16"]["alignment"]["action_mse_vs_bf16"] == 0
    assert report["tiers"]["nvfp4"]["alignment"]["action_mse_vs_bf16"] > 0


@pytest.mark.parametrize("value", ["int8", "bf16,", "unknown"])
def test_tensor_rt_cli_rejects_invalid_tiers(value):
    with pytest.raises(SystemExit) as failure:
        export_tensorrt.options(["--precisions", value])
    assert failure.value.code == 2


def test_real_export_pipeline_builds_profiles_then_publishes_only_supported_tiers(
    tmp_path,
    monkeypatch,
):
    component = ActionComponent("action_head.model", ("hidden_states",))
    sample = export_tensorrt.observations(None, seed=5)[:1]
    torch = SimpleNamespace(
        cuda=SimpleNamespace(device=lambda _: nullcontext(), empty_cache=Mock())
    )
    policy = SimpleNamespace(
        trt_component=component,
        name="groot/test",
        capability=(8, 9),
        device="cuda:0",
        model=object(),
        torch=torch,
    )
    monkeypatch.setattr(export_tensorrt, "create_policy", lambda _: (policy, "revision"))
    reference = SimpleNamespace(action_chunk=[[0] * 7], infer_ms=24)
    candidate = SimpleNamespace(action_chunk=[[1] * 7], infer_ms=15)
    capture = Mock(return_value=([reference], [(object(),)]))
    monkeypatch.setattr(export_tensorrt, "capture_component_inputs", capture)

    def export(_policy, mode, _batches, path):
        path.write_text(f"ONNX:{mode}")

    def build(onnx_path, plan_path, _component, _batches, **_kwargs):
        assert onnx_path.name == ONNX_FILE
        plan_path.write_text(f"Plan for {onnx_path.read_text()}")

    backend = Mock()
    backend.installed.return_value = nullcontext()
    monkeypatch.setattr(export_tensorrt, "export_onnx", export)
    monkeypatch.setattr(export_tensorrt, "build_engine", build)
    monkeypatch.setattr(export_tensorrt, "TensorRTActionBackend", lambda *a, **kw: backend)
    monkeypatch.setattr(export_tensorrt, "run_policy", lambda *_: [candidate])
    profile = Mock(return_value={"source": "cuda_event", "p99_ms": 1.2})
    monkeypatch.setattr(export_tensorrt, "profile_component", profile)

    args = export_tensorrt.options(["--model", "groot"])
    report = export_tensorrt.run_gpu(args, sample, sample, tmp_path)
    assert capture.call_count == 2 and profile.call_count == 2
    assert report["tiers"]["nvfp4"]["status"] == "skipped"
    for mode in ("bf16", "fp8"):
        tier = tmp_path / mode
        assert (tier / PLAN_FILE).read_text() == f"Plan for ONNX:{mode}"
        identity = json.loads((tier / MANIFEST).read_text())
        assert identity["model"] == "groot" and identity["component"]["path"] == component.path
        assert identity["precision"] == mode and identity["capability"] == [8, 9]
        assert report["tiers"][mode]["alignment"]["action_l1_vs_bf16"] == 1
    assert not (tmp_path / "nvfp4").is_dir()


def test_cuda_event_profile_warms_up_then_times_on_the_execution_stream():
    start, stop = Mock(), Mock()
    start.elapsed_time.side_effect = [1, 2, 3]
    stream = Mock()
    cuda = SimpleNamespace(
        device=lambda _: nullcontext(),
        current_stream=lambda _: stream,
        Event=Mock(side_effect=[start, stop]),
    )
    torch = SimpleNamespace(cuda=cuda, inference_mode=nullcontext)
    backend = SimpleNamespace(
        torch=torch, device="cuda:0", run=Mock(), component=ActionComponent("action", ("input",))
    )
    inputs = [(Mock(),), (Mock(),)]
    result = export_tensorrt.profile_component(backend, inputs, warmup=2, iterations=3)
    assert backend.run.call_count == 5
    assert start.record.call_count == stop.record.call_count == stop.synchronize.call_count == 3
    start.record.assert_called_with(stream)
    stop.record.assert_called_with(stream)
    stream.synchronize.assert_called_once()
    assert result["source"] == "cuda_event" and result["samples"] == 3
    assert result["p99_ms"] == pytest.approx(2.98)

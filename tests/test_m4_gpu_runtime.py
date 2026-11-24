import ast
import json
from pathlib import Path
from unittest.mock import Mock

from backend.chaos_sim.cpu_env import CpuPickEnv
from backend.chaos_sim.libero_env import LiberoEnvAdapter
from benchmarks import auto_chaos
from scripts import m3_gpu_runtime, m4_gpu_runtime

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/break_my_vla_m4_groot_trt_nvfp4.ipynb"


def test_m4_runtime_builds_each_model_in_the_matching_python(monkeypatch, tmp_path):
    install = Mock(side_effect=lambda uv, path, packages, **kwargs: path / "bin/python")
    run, source, renderer = Mock(), Mock(), Mock(return_value="osmesa")
    monkeypatch.setattr(m4_gpu_runtime, "install_runtime", install)
    monkeypatch.setattr(m4_gpu_runtime, "install_groot_source", source)
    monkeypatch.setattr(m4_gpu_runtime, "rendering_backend", renderer)
    monkeypatch.setattr(m4_gpu_runtime, "uv_command", lambda: ["uv"])
    monkeypatch.setattr(m4_gpu_runtime.subprocess, "run", run)
    runtime, output = tmp_path / "runtime", tmp_path / "results"
    m4_gpu_runtime.main(
        [
            "--runtime-dir",
            str(runtime),
            "--output",
            str(output),
            "--tasks",
            "libero_goal:1",
            "--pi05-tokenizer",
            "test/tokenizer",
            "--install-osmesa",
        ]
    )

    base, pi05, groot = (
        runtime / f"{model}-venv-py311/bin/python" for model in ("sim", "pi05", "groot")
    )
    source.assert_called_once_with(groot, runtime / "isaac-gr00t")
    renderer.assert_called_once_with(base, install_osmesa=True)
    assert install.call_count == 3
    for call in install.call_args_list:
        assert call.kwargs["extra_indexes"] == (m4_gpu_runtime.NVIDIA_INDEX,)
        assert set(m4_gpu_runtime.TRT_PACKAGES) <= set(call.args[2])
        invocation = m3_gpu_runtime.pip_command(*call.args, **call.kwargs)
        assert invocation[invocation.index("--index-strategy") + 1] == "unsafe-best-match"

    gpu, collect, *exports, benchmark = [call.args[0] for call in run.call_args_list]
    assert gpu[0] == "nvidia-smi"
    assert collect[:4] == [base, "-m", "scripts.m4_gpu_runtime", "capture"]
    assert len(exports) == 3
    for command, expected in zip(exports, (base, pi05, groot), strict=True):
        assert command[:3] == [expected, "-m", "scripts.export_tensorrt"]
        assert command[command.index("--precisions") + 1] == "bf16,fp8,nvfp4"
        assert command[command.index("--groot-suite") + 1] == "libero_goal"
        assert command[command.index("--calibration") + 1] == str(output / "calibration.jsonl")
        assert command[command.index("--observations") + 1] == str(output / "heldout.jsonl")
    assert benchmark[:3] == [base, "-m", "benchmarks.auto_chaos"]
    assert benchmark[benchmark.index("--levels") + 1] == "0-5"
    assert benchmark[benchmark.index("--precisions") + 1] == "bf16,fp8,int8,nvfp4"
    assert str(output / "tensorrt") in benchmark
    assert f"groot={groot}" in benchmark and f"pi05={pi05}" in benchmark
    for call in run.call_args_list[1:]:
        assert call.kwargs["env"]["MPLBACKEND"] == "Agg"
        assert call.kwargs["env"]["MUJOCO_GL"] == "osmesa"


def test_m4_runtime_mock_dispatches_export_and_benchmark_without_gpu(monkeypatch, tmp_path):
    install, run = Mock(), Mock()
    monkeypatch.setattr(m4_gpu_runtime, "install_runtime", install)
    monkeypatch.setattr(m4_gpu_runtime, "uv_command", lambda: ["uv"])
    monkeypatch.setattr(m4_gpu_runtime.subprocess, "run", run)
    m4_gpu_runtime.main(["--mock", "--runtime-dir", str(tmp_path)])
    install.assert_not_called()
    commands = [call.args[0] for call in run.call_args_list]
    assert len(commands) == 4
    for command in commands:
        assert command[:6] == ["uv", "run", "--python", "3.11", "python", "-m"]
        assert "--mock" in command
    assert commands[-1][6] == "benchmarks.auto_chaos"
    assert run.call_args.kwargs["env"]["UV_PROJECT_ENVIRONMENT"] == str(
        tmp_path / "mock-venv-py311"
    )


def test_capture_uses_disjoint_libero_initial_states_and_closes_each_env(monkeypatch, tmp_path):
    closed = []

    def open_sim(suite, task_id, *, initial_state_id, seed):
        env = CpuPickEnv()
        env.close = lambda: closed.append(initial_state_id)
        env.prompt = f"{suite}:{task_id}:{initial_state_id}:{seed}"
        return env

    monkeypatch.setattr(LiberoEnvAdapter, "from_task", open_sim)
    monkeypatch.setattr(auto_chaos, "prepare_libero", Mock())
    calibration, holdout = tmp_path / "calibration.jsonl", tmp_path / "heldout.jsonl"
    m4_gpu_runtime.capture_observations(
        [
            "--calibration-count",
            "2",
            "--observation-count",
            "1",
            "--calibration",
            str(calibration),
            "--observations",
            str(holdout),
            "--",
            "--mock",
            "--tasks",
            "libero_goal:1,libero_spatial:2",
            "--seed",
            "3",
        ]
    )
    assert closed == [1, 2, 3]
    prompts = [json.loads(row)["prompt"] for row in calibration.read_text().splitlines()]
    assert prompts == ["libero_goal:1:1:3", "libero_spatial:2:2:3"]
    assert json.loads(holdout.read_text())["prompt"] == "libero_goal:1:3:3"


def test_m4_notebook_is_valid_python_311_and_invokes_mock_launcher(monkeypatch, tmp_path):
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert len({cell["id"] for cell in notebook["cells"]}) == len(notebook["cells"])
    code = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            source = "".join(cell["source"])
            ast.parse(source, feature_version=(3, 11))
            code[cell["id"]] = compile(source, str(NOTEBOOK), "exec")
    assert tuple(code) == ("parameters", "workspace", "run", "results")
    for variable in (
        "BMVLA_MODELS",
        "BMVLA_PRECISIONS",
        "BMVLA_TASKS",
        "BMVLA_LEVELS",
        "BMVLA_MAX_STEPS",
    ):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("BMVLA_MOCK", "1")
    monkeypatch.setenv("BMVLA_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("BMVLA_REPO_DIR", str(ROOT))
    monkeypatch.setattr(m4_gpu_runtime.subprocess, "run", run := Mock())
    namespace = {}
    for cell in ("parameters", "workspace", "run"):
        exec(code[cell], namespace)
    command = run.call_args.args[0]
    assert command[1:3] == ["-m", "scripts.m4_gpu_runtime"]
    assert command[command.index("--models") + 1] == "smolvla,pi05,groot"
    assert command[command.index("--precisions") + 1] == "bf16,fp8,int8,nvfp4"
    assert command[command.index("--levels") + 1] == "0-5"
    assert "--mock" in command and str(tmp_path) in command
    assert run.call_args.kwargs == {"cwd": ROOT, "check": True}

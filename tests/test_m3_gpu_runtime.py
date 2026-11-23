import sys
from unittest.mock import Mock

from scripts import m3_gpu_runtime


def test_gpu_runtime_uses_separate_pi05_python_and_headless_environment(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "m3_gpu_runtime",
            "--models",
            "smolvla, pi05",
            "--install-osmesa",
            "--runtime-dir",
            str(tmp_path),
            "--",
            "--levels",
            "0,3",
        ],
    )
    run = Mock()
    install = Mock(side_effect=lambda uv, path, packages: path / "bin/python")
    render = Mock(return_value="egl")
    monkeypatch.setattr(m3_gpu_runtime.subprocess, "run", run)
    monkeypatch.setattr(m3_gpu_runtime, "uv_command", lambda: ["uv"])
    monkeypatch.setattr(m3_gpu_runtime, "install_runtime", install)
    monkeypatch.setattr(m3_gpu_runtime, "rendering_backend", render)

    m3_gpu_runtime.main()

    sim = tmp_path / "venv-py311/bin/python"
    pi05 = tmp_path / "pi05-venv-py311/bin/python"
    render.assert_called_once_with(sim, install_osmesa=True)
    assert [call.args[2] for call in install.call_args_list] == [
        m3_gpu_runtime.SIM_PACKAGES,
        m3_gpu_runtime.PI05_PACKAGES,
    ]
    assert run.call_args_list[0].args[0][0] == "nvidia-smi"
    command = run.call_args.args[0]
    assert command[:3] == [sim, "-m", "benchmarks.auto_chaos"]
    assert command[-2:] == ["--policy-python", f"pi05={pi05}"]
    assert "--levels" in command and "0,3" in command
    assert run.call_args.kwargs["env"]["MPLBACKEND"] == "Agg"
    assert run.call_args.kwargs["env"]["MUJOCO_GL"] == "egl"

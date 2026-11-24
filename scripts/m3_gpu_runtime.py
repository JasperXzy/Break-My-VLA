"""Set up the notebook's Python 3.11 GPU runtimes and run Auto-Chaos."""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CUDA_INDEX = "https://download.pytorch.org/whl/cu128"
CUDA_PACKAGES = ("torch==2.10.0+cu128", "torchvision==0.25.0+cu128")
SIM_PACKAGES = (
    "lerobot[smolvla,libero]==0.4.4",
    "hf-libero==0.1.5",
    "mujoco==3.3.7",
    "transformers==4.57.6",
)
# LeRobot v0.4.4 Pi policies use the OpenPI Gemma and SigLIP changes from this fork.
PI05_TRANSFORMERS = (
    "transformers @ git+https://github.com/huggingface/transformers.git"
    "@dcddb970176382c0fcf4521b0c0e6fc15894dfe0"
)
PI05_PACKAGES = ("lerobot==0.4.4", PI05_TRANSFORMERS, "scipy>=1.10.1,<1.15", "sentencepiece")
RENDER_PROBE = "import cv2, mujoco; ctx = mujoco.GLContext(32, 32); ctx.make_current(); ctx.free()"


def uv_command() -> list[str]:
    if executable := shutil.which("uv"):
        return [executable]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
    return [sys.executable, "-m", "uv"]


def pip_command(
    uv: list[str], python: Path, packages: tuple[str, ...], *, extra_indexes: tuple[str, ...] = ()
) -> list[str]:
    return [
        *uv,
        "pip",
        "install",
        "--python",
        str(python),
        "--index-strategy",
        "unsafe-best-match",
        "--extra-index-url",
        CUDA_INDEX,
        *(part for index in extra_indexes for part in ("--extra-index-url", index)),
        "--requirements",
        str(ROOT / "pyproject.toml"),
        *CUDA_PACKAGES,
        *packages,
    ]


def install_runtime(
    uv: list[str], path: Path, packages: tuple[str, ...], *, extra_indexes: tuple[str, ...] = ()
) -> Path:
    python = path / "bin" / "python"
    if not python.exists():
        subprocess.run([*uv, "venv", "--python", "3.11", path], check=True)
    subprocess.run(pip_command(uv, python, packages, extra_indexes=extra_indexes), check=True)
    return python


def rendering_backend(python: Path, *, install_osmesa: bool) -> str:
    def can_render(backend: str) -> bool:
        result = subprocess.run(
            [python, "-c", RENDER_PROBE],
            env=os.environ | {"MUJOCO_GL": backend, "MPLBACKEND": "Agg"},
            capture_output=True,
            text=True,
        )
        return result.returncode == 0

    for backend in (os.environ.get("MUJOCO_GL", "egl"), "osmesa"):
        if can_render(backend):
            return backend
    if install_osmesa:
        subprocess.run(["apt-get", "-qq", "update"], check=True)
        subprocess.run(["apt-get", "-qq", "install", "-y", "libgl1", "libosmesa6"], check=True)
        if can_render("osmesa"):
            return "osmesa"
    raise RuntimeError(
        "MuJoCo rendering failed. Configure EGL or install the OSMesa system library."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="smolvla,pi05")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--install-osmesa", action="store_true", help="Allow apt fallback in Colab")
    parser.add_argument("--runtime-dir", type=Path, default=Path.home() / ".cache/break-my-vla/m3")
    parser.add_argument("benchmark_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    benchmark_args = args.benchmark_args
    if benchmark_args[:1] == ["--"]:
        benchmark_args = benchmark_args[1:]
    benchmark = ["-m", "benchmarks.auto_chaos", "--models", args.models, *benchmark_args]
    env = os.environ | {
        "MPLBACKEND": "Agg",
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    if args.mock:
        subprocess.run(
            [*uv_command(), "run", "python", *benchmark, "--mock"], cwd=ROOT, env=env, check=True
        )
        return

    subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"], check=True
    )
    uv = uv_command()
    runtime = args.runtime_dir.resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    python = install_runtime(uv, runtime / "venv-py311", SIM_PACKAGES)
    if "pi05" in {model.strip() for model in args.models.split(",")}:
        pi05_python = install_runtime(uv, runtime / "pi05-venv-py311", PI05_PACKAGES)
        benchmark.extend(("--policy-python", f"pi05={pi05_python}"))
    renderer = rendering_backend(python, install_osmesa=args.install_osmesa)
    env.update({"MUJOCO_GL": renderer, "LIBERO_CONFIG_PATH": str(runtime / "libero-config")})
    subprocess.run([python, *benchmark], cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()

"""Run the real M2 benchmark on an ephemeral Colab GPU and retrieve its artifacts."""

import argparse
import hashlib
import importlib.metadata
import os
import re
import subprocess
import sys
from base64 import b64decode, b64encode
from io import BytesIO
from pathlib import Path
from time import monotonic, sleep
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = Path("/content/break-my-vla-m2")
ARTIFACT_MARKER = "BMVLA_M2_ARTIFACT "
CHUNK_SIZE = 48_000


def install_runtime() -> Path:
    venv = WORKSPACE / "venv"
    subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", venv], check=True)
    python = venv / "bin" / "python"
    pip = [python, "-m", "pip", "install", "--disable-pip-version-check", "-q"]

    try:
        torch_pair = tuple(
            importlib.metadata.version(name).split("+")[0] for name in ("torch", "torchvision")
        )
    except importlib.metadata.PackageNotFoundError:
        torch_pair = ()
    if torch_pair not in {("2.8.0", "0.23.0"), ("2.9.0", "0.24.0"), ("2.10.0", "0.25.0")}:
        print("Installing CUDA PyTorch for Blackwell", flush=True)
        subprocess.run(
            [
                *pip,
                "--index-url",
                "https://download.pytorch.org/whl/cu128",
                "torch==2.10.0",
                "torchvision==0.25.0",
            ],
            check=True,
        )
    print("Installing the LIBERO and LeRobot runtime", flush=True)
    subprocess.run(
        [
            *pip,
            "lerobot[smolvla,libero]==0.4.4",
            "hf-libero==0.1.5",
            "mujoco==3.3.7",
            "transformers==4.57.6",
            "fastapi>=0.115,<1",
            "uvicorn>=0.35,<1",
            "httpx>=0.27,<1",
        ],
        check=True,
    )
    return python


def source_archive() -> bytes:
    buffer = BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
        paths = sorted((ROOT / "backend").rglob("*.py")) + [ROOT / "benchmarks/m2_smoke.py"]
        for path in paths:
            archive.write(path, path.relative_to(ROOT))
    return buffer.getvalue()


def rendering_backend(python: Path) -> str:
    probe = "import cv2, mujoco; ctx = mujoco.GLContext(32, 32); ctx.make_current(); ctx.free()"
    if subprocess.run([python, "-c", probe], env=os.environ | {"MUJOCO_GL": "egl"}).returncode == 0:
        return "egl"
    print("EGL unavailable. Installing an offscreen OSMesa renderer.", flush=True)
    subprocess.run(["apt-get", "-qq", "update"], check=True)
    subprocess.run(["apt-get", "-qq", "install", "-y", "libgl1", "libosmesa6"], check=True)
    subprocess.run([python, "-c", probe], env=os.environ | {"MUJOCO_GL": "osmesa"}, check=True)
    return "osmesa"


def run_remote(args) -> None:
    subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"], check=True
    )
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    python = install_runtime()
    if args.source_b64:
        source = b64decode(args.source_b64, validate=True)
    else:
        source_path = Path(args.remote_source)
        deadline = monotonic() + 600
        print(f"Waiting for source archive at {source_path}", flush=True)
        while not source_path.is_file():
            if monotonic() >= deadline:
                raise TimeoutError(f"Source archive did not arrive at {source_path}")
            sleep(2)
        source = source_path.read_bytes()
    with ZipFile(BytesIO(source)) as archive:
        archive.extractall(WORKSPACE)

    output = WORKSPACE / "results"
    subprocess.run(
        [python, "-m", "benchmarks.m2_smoke", "--output", output, *args.benchmark_args],
        cwd=WORKSPACE,
        env=os.environ
        | {
            "MUJOCO_GL": rendering_backend(python),
            "LIBERO_CONFIG_PATH": str(WORKSPACE / "libero-config"),
            "BMVLA_SOURCE_SHA256": hashlib.sha256(source).hexdigest(),
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
        },
        check=True,
    )
    buffer = BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
        for path in sorted(output.iterdir()):
            if path.suffix in (".gif", ".json"):
                archive.write(path, path.name)
    data = buffer.getvalue()
    print(f"{ARTIFACT_MARKER}sha256 {hashlib.sha256(data).hexdigest()}", flush=True)
    encoded = b64encode(data).decode()
    for start in range(0, len(encoded), CHUNK_SIZE):
        print(
            f"{ARTIFACT_MARKER}{start // CHUNK_SIZE} {encoded[start : start + CHUNK_SIZE]}",
            flush=True,
        )
        sleep(0.08)


def receive_artifacts(lines, output: Path) -> None:
    chunks: dict[int, str] = {}
    digest = None
    for line in lines:
        match = re.search(rf"{ARTIFACT_MARKER}(sha256|[0-9]+) ([A-Za-z0-9+/=]+)", line)
        if match is None:
            print(line, end="", flush=True)
            continue
        key, value = match.groups()
        if key == "sha256":
            digest = value
        else:
            chunks[int(key)] = value
    if not chunks or not digest or sorted(chunks) != list(range(len(chunks))):
        raise ValueError("Incomplete Colab artifact stream")
    data = b64decode("".join(chunks[i] for i in range(len(chunks))), validate=True)
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError("Colab artifact checksum does not match")
    with ZipFile(BytesIO(data)) as archive:
        expected = {
            "baseline.gif",
            "object_nudge.gif",
            "lighting_shift.gif",
            "int8_low_steps.gif",
            "metrics.json",
        }
        if set(archive.namelist()) != expected:
            raise ValueError(f"Unexpected Colab artifacts: {archive.namelist()}")
        output.mkdir(parents=True, exist_ok=True)
        archive.extractall(output)
    print(f"Retrieved {len(expected)} verified files in {output}", flush=True)


def launch(args) -> None:
    bundle = b64encode(source_archive()).decode()
    with subprocess.Popen(
        [
            "colab",
            "run",
            "--gpu",
            args.gpu,
            "--timeout",
            "600",
            "-s",
            args.session,
            str(Path(__file__).resolve()),
            "remote",
            "--source-b64",
            bundle,
            *args.benchmark_args,
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        text=True,
    ) as job:
        assert job.stdout is not None
        try:
            receive_artifacts(job.stdout, args.output)
        finally:
            if job.wait() != 0:
                raise subprocess.CalledProcessError(job.returncode, "colab run")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_subparsers(dest="mode", required=True)
    local = mode.add_parser("launch")
    local.add_argument("--session", default="break-my-vla-m2")
    local.add_argument("--gpu", choices=("G4", "L4"), default="G4")
    local.add_argument("--output", type=Path, default=ROOT / "benchmarks/results/m2_smoke")
    remote = mode.add_parser("remote")
    source = remote.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-b64")
    source.add_argument("--remote-source")
    retrieve = mode.add_parser("retrieve")
    retrieve.add_argument("log", type=Path)
    retrieve.add_argument("--output", type=Path, default=ROOT / "benchmarks/results/m2_smoke")
    args, args.benchmark_args = parser.parse_known_args()
    if args.mode == "launch":
        launch(args)
    elif args.mode == "remote":
        run_remote(args)
    else:
        with args.log.open() as log:
            receive_artifacts(log, args.output)


if __name__ == "__main__":
    main()

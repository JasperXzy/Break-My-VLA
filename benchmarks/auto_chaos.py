"""Run closed-loop Auto-Chaos rollouts and export metrics, Gallery JSON, and GIFs."""

import argparse
import json
import re
import shutil
import socket
import subprocess
import sys
from collections import deque
from contextlib import closing, contextmanager, nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from os import environ
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import monotonic, perf_counter, sleep
from typing import get_args

import httpx
from pydantic import ValidationError

from backend.chaos_sim.chaos import ChaosWrapper
from backend.chaos_sim.cpu_env import CpuPickEnv
from backend.chaos_sim.libero_env import ASSETS_REPO, ASSETS_REVISION, LiberoEnvAdapter
from backend.chaos_sim.media import TARGET_FPS, jpeg_b64, replay_frame
from backend.chaos_sim.policies import MODEL_ID, parse_policy_servers
from backend.shared.schemas import (
    PolicyActRequest,
    PolicyActResponse,
    PolicyHealthResponse,
    Precision,
)
from benchmarks.chaos_gallery import SCORED, build_gallery
from benchmarks.chaos_matrix import Case, stress_matrix

MODULES = {
    "smolvla": "backend.policy_servers.smolvla_server",
    "pi05": "backend.policy_servers.pi05_server",
}
REPLAY_SKILLS = {"object_nudge", "lighting_shift", "latency_inject"}
REPLAY_STRIDE = 4


@dataclass(frozen=True, slots=True)
class Task:
    suite: str
    task_id: int

    @property
    def label(self) -> str:
        return f"{self.suite}:{self.task_id}"

    @classmethod
    def parse(cls, text: str, default_suite: str) -> "Task":
        suite, separator, task = text.partition(":")
        if not separator:
            suite, task = default_suite, suite
        if not re.fullmatch(r"libero_[a-z0-9_]+", suite) or not re.fullmatch(r"[0-9]+", task):
            raise ValueError(f"Expected TASK_ID or LIBERO_SUITE:TASK_ID, got {text!r}")
        return cls(suite, int(task))


def levels_from(values: tuple[str, ...]) -> tuple[int, ...]:
    levels = set()
    for value in values:
        if not re.fullmatch(r"[0-5](?:-[0-5])?", value):
            raise ValueError(f"Expected stress levels from 0 to 5, got {value!r}")
        lo, _, hi = value.partition("-")
        start, end = int(lo), int(hi or lo)
        if end < start:
            raise ValueError(f"Expected an ascending level range, got {value!r}")
        levels.update(range(start, end + 1))
    return tuple(sorted(levels))


def options(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", help="Policy aliases, separated by spaces or commas")
    parser.add_argument("--precisions", nargs="+", default=["bf16,fp8,int8"])
    parser.add_argument("--tasks", nargs="+", default=["0"], help="IDs or LIBERO_SUITE:TASK_ID")
    parser.add_argument("--levels", nargs="+", default=["0-5"], help="Levels 0-5 or a subset")
    parser.add_argument(
        "--mock", action="store_true", help="Use CPU simulation and HTTP mock workers"
    )
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--initial-state-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--inject-step", type=int, default=0)
    parser.add_argument("--chunk-actions", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--policy-python", action="append", default=[], metavar="MODEL=PYTHON")
    parser.add_argument("--pi05-tokenizer")
    parser.add_argument("--pi05-nvfp4-backend")
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/m3_auto_chaos"))
    args = parser.parse_args(argv)

    def csv(values: list[str]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(part.strip() for value in values for part in value.split(",")))

    try:
        configured = environ.get("POLICY_SERVERS")
        args.endpoints = parse_policy_servers(configured) if configured and not args.mock else {}
        args.models = csv(args.models) if args.models else tuple(args.endpoints or MODULES)
        for name in args.models:
            MODEL_ID.validate_python(name)
            if not args.mock and name not in MODULES and name not in args.endpoints:
                raise ValueError(f"Unknown model {name!r}. Configure it in POLICY_SERVERS")
        args.precisions = csv(args.precisions)
        if any(mode not in get_args(Precision) for mode in args.precisions):
            raise ValueError("Expected precisions from bf16, fp8, int8, nvfp4")
        args.levels = levels_from(csv(args.levels))
        args.tasks = tuple(dict.fromkeys(Task.parse(task, args.suite) for task in csv(args.tasks)))
        python_paths = {}
        for entry in args.policy_python:
            model, _, python = entry.partition("=")
            if model not in args.models or not python or model in python_paths:
                raise ValueError(f"Expected a unique requested MODEL=PYTHON, got {entry!r}")
            python_paths[model] = python
        args.policy_python = python_paths
    except ValueError as error:
        parser.error(str(error))

    if args.max_steps is None:
        args.max_steps = 40 if args.mock else 280
    if args.chunk_actions is None:
        args.chunk_actions = 1 if args.mock else 10
    if not 0 <= args.inject_step < args.max_steps or args.chunk_actions < 1:
        parser.error("Expected 0 <= inject-step < max-steps and chunk-actions >= 1")
    if args.initial_state_id < 0 or args.seed < 0:
        parser.error("Expected nonnegative initial-state-id and seed")
    return args


def read_health(client: httpx.Client) -> PolicyHealthResponse:
    response = client.get("/health", timeout=2)
    response.raise_for_status()
    return PolicyHealthResponse.model_validate_json(response.content)


@contextmanager
def policy_worker(args: argparse.Namespace, model: str):
    timeout = httpx.Timeout(60.0, connect=1.0)
    if model in args.endpoints:
        with httpx.Client(
            base_url=args.endpoints[model], timeout=timeout, trust_env=False
        ) as client:
            yield client, read_health(client)
        return

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    python = args.policy_python.get(model, sys.executable)
    module = "backend.policy_servers.mock_server" if args.mock else MODULES[model]
    command = [python, "-m", module, "--port", str(port)]
    if args.mock:
        command.append("--instant")
    else:
        command.extend(("--seed", str(args.seed), "--device", args.device))
        if model == "pi05" and args.pi05_tokenizer:
            command.extend(("--tokenizer", args.pi05_tokenizer))
        if model == "pi05" and args.pi05_nvfp4_backend:
            command.extend(("--nvfp4-backend", args.pi05_nvfp4_backend))

    with (
        httpx.Client(
            base_url=f"http://127.0.0.1:{port}", timeout=timeout, trust_env=False
        ) as client,
        subprocess.Popen(command) as process,
    ):
        try:
            deadline = monotonic() + (15 if args.mock else 1200)
            while monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"{model} policy server exited with status {process.returncode}"
                    )
                try:
                    health = read_health(client)
                except httpx.HTTPError:
                    sleep(0.1 if args.mock else 1)
                else:
                    yield client, health
                    break
            else:
                raise TimeoutError(f"Timed out waiting for {model} policy health")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def rollout(
    client: httpx.Client,
    wrapper: ChaosWrapper,
    args: argparse.Namespace,
    *,
    case: Case,
    task: Task,
    model: str,
    record: bool,
) -> dict:
    wrapper.reset()
    scene = wrapper.observe()
    prompt = getattr(wrapper.env, "prompt", "Pick up the orange block.")
    run_id = (
        f"{model}-{task.suite}-{task.task_id}-{case.nominal_precision}-l{case.level}-{case.tag}"
    )
    caption = f"{case.precision.upper()} {case.tag} L{case.level}"
    frames = [replay_frame(scene, case=caption, step=0)] if record else []
    pending = deque()
    calls = []
    injected_at, status, error = None, None, None
    started = perf_counter()

    if (
        case.tag == "object_nudge"
        and isinstance(wrapper.env, LiberoEnvAdapter)
        and wrapper.env.object_name is None
    ):
        status, error = "not_exercised", "The task has no movable object for object_nudge"

    while status is None and wrapper.step_count < args.max_steps:
        if scene.observation.task_progress >= 1:
            break
        if case.command is not None and wrapper.step_count == args.inject_step:
            wrapper.inject(case.command)
            injected_at = wrapper.step_count
            pending.clear()
            scene = wrapper.observe()
            if record:
                frames.append(replay_frame(scene, case=caption, step=wrapper.step_count))
            if scene.observation.task_progress >= 1:
                break

        if not pending:
            obs = wrapper.observe().observation
            request = PolicyActRequest(
                rgb_main_b64=jpeg_b64(obs.rgb_main),
                rgb_wrist_b64=jpeg_b64(obs.rgb_wrist),
                proprio=obs.proprio,
                prompt=prompt,
                precision=case.precision,
                denoise_steps=case.denoise_steps,
            )
            injected_ms = wrapper.latency_ms(0.0)
            call_started = perf_counter()
            if injected_ms and not args.mock:
                sleep(injected_ms / 1000)
            try:
                response = client.post("/act", json=request.model_dump(mode="json"))
                response.raise_for_status()
                result = PolicyActResponse.model_validate_json(response.content)
            except (httpx.HTTPError, ValidationError) as exc:
                status, error = "policy_error", str(exc)
                break
            calls.append(
                {
                    "step": wrapper.step_count,
                    "infer_ms": result.infer_ms,
                    "injected_latency_ms": injected_ms,
                    "effective_infer_ms": result.infer_ms + injected_ms,
                    "wall_ms": (perf_counter() - call_started) * 1000,
                    "vram_mb": result.vram_mb,
                }
            )
            pending.extend(result.action_chunk[: args.chunk_actions])

        scene = wrapper.step(pending.popleft())
        if record and (
            wrapper.step_count % REPLAY_STRIDE == 0 or scene.observation.task_progress >= 1
        ):
            frames.append(replay_frame(scene, case=caption, step=wrapper.step_count))

    elapsed = perf_counter() - started
    success = scene.observation.task_progress >= 1
    if status is None:
        if case.command is not None and injected_at is None:
            status, error = "not_exercised", "The episode ended before the scheduled injection"
        else:
            status = "success" if success else "step_limit"

    gif = None
    if record and status in SCORED:
        if wrapper.step_count % REPLAY_STRIDE and not success:
            frames.append(replay_frame(scene, case=caption, step=wrapper.step_count))
        gif = f"./replays/{run_id}.gif"
        frames[0].save(
            args.output / gif,
            save_all=True,
            append_images=frames[1:],
            duration=REPLAY_STRIDE * 1000 // TARGET_FPS,
            loop=0,
        )
    return {
        "id": run_id,
        "model": model,
        "task": task.label,
        "prompt": prompt,
        "axis": case.axis,
        "tag": case.tag,
        "level": case.level,
        "nominal_precision": case.nominal_precision,
        "precision": case.precision,
        "denoise_steps": case.denoise_steps,
        "command": case.command.model_dump(mode="json") if case.command is not None else None,
        "injected_at_step": injected_at,
        "status": status,
        "error": error,
        "success": success,
        "steps": wrapper.step_count,
        "task_progress": scene.observation.task_progress,
        "wall_s": elapsed,
        "calls": calls,
        "gif": gif,
    }


def run_model(args: argparse.Namespace, model: str, metrics: dict) -> None:
    with policy_worker(args, model) as (client, health):
        supported = tuple(mode for mode in args.precisions if mode in health.precisions)
        skipped = [mode for mode in args.precisions if mode not in health.precisions]
        reference = "bf16" if "bf16" in supported else next(iter(supported), None)
        metrics["models"][model] = {
            "health": health.model_dump(mode="json"),
            "precisions": supported,
            "skipped_precisions": skipped,
            "reference_precision": reference,
        }
        if skipped:
            print(f"{model}: unavailable precisions: {', '.join(skipped)}", flush=True)
        if not supported:
            return
        cases = stress_matrix(supported, args.levels)
        for index, task in enumerate(args.tasks):
            resource = (
                nullcontext(CpuPickEnv())
                if args.mock
                else closing(
                    LiberoEnvAdapter.from_task(
                        task.suite,
                        task.task_id,
                        initial_state_id=args.initial_state_id,
                        seed=args.seed,
                    )
                )
            )
            with resource as env:
                wrapper = ChaosWrapper(env)
                for case in cases:
                    record = index == 0 and (
                        case.axis == "clean"
                        or (
                            case.level == max(args.levels)
                            and (
                                case.axis == "quantization"
                                or (
                                    case.nominal_precision == reference
                                    and case.tag in REPLAY_SKILLS
                                )
                            )
                        )
                    )
                    run = rollout(
                        client, wrapper, args, case=case, task=task, model=model, record=record
                    )
                    metrics["runs"].append(run)
                    print(f"{run['id']}: {run['status']} at step {run['steps']}", flush=True)
                    if run["error"]:
                        print(f"  {run['error']}", flush=True)


def prepare_libero() -> str:
    from huggingface_hub import snapshot_download

    environ.setdefault("MPLBACKEND", "Agg")
    snapshot_download(
        ASSETS_REPO,
        revision=ASSETS_REVISION,
        repo_type="dataset",
        local_dir=Path.home() / ".cache/libero/assets",
    )
    if smi := shutil.which("nvidia-smi"):
        gpu = subprocess.run(
            [smi, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        )
        if gpu.returncode == 0 and gpu.stdout.strip():
            return " | ".join(gpu.stdout.strip().splitlines())
    return "LIBERO · policy devices in metrics.json"


def write_json(path: Path, data: dict) -> None:
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as temp:
        staging = Path(temp.name)
        try:
            json.dump(data, temp, ensure_ascii=False, indent=2, allow_nan=False)
            temp.write("\n")
            temp.close()
            staging.replace(path)
        finally:
            staging.unlink(missing_ok=True)


def main(argv=None) -> None:
    args = options(argv)
    (args.output / "replays").mkdir(parents=True, exist_ok=True)
    metrics = {
        "schema_version": "1.0",
        "mode": "cpu_mock" if args.mock else "libero",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "hardware": "CPU Mock" if args.mock else prepare_libero(),
        "requested_models": args.models,
        "requested_precisions": args.precisions,
        "tasks": [task.label for task in args.tasks],
        "levels": args.levels,
        "max_steps": args.max_steps,
        "inject_step": args.inject_step,
        "chunk_actions": args.chunk_actions,
        "seed": args.seed,
        "initial_state_id": args.initial_state_id,
        "renderer": None if args.mock else environ.get("MUJOCO_GL", "egl"),
        "libero_assets_revision": None if args.mock else ASSETS_REVISION,
        "models": {},
        "runs": [],
    }
    for model in args.models:
        run_model(args, model, metrics)
        write_json(args.output / "metrics.json", metrics)
        write_json(args.output / "gallery.json", build_gallery(metrics))
    print(f"Auto-Chaos artifacts: {args.output.resolve()}", flush=True)
    if not metrics["runs"] or any(run["status"] == "policy_error" for run in metrics["runs"]):
        raise SystemExit("Auto-Chaos had unavailable policy runs. See metrics.json for details.")


if __name__ == "__main__":
    main()

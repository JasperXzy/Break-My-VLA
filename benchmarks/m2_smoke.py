"""Compare four closed-loop LIBERO episodes against the isolated SmolVLA server."""

import argparse
import json
import subprocess
import sys
from collections import deque
from contextlib import contextmanager
from importlib.metadata import version
from os import environ

environ["MPLBACKEND"] = "Agg"
from pathlib import Path
from time import monotonic, perf_counter, sleep

import httpx
import numpy as np
from PIL import ImageStat

from backend.chaos_sim.chaos import ChaosWrapper, Scene
from backend.chaos_sim.libero_env import ASSETS_REPO, ASSETS_REVISION, LiberoEnvAdapter
from backend.chaos_sim.media import jpeg_b64, replay_frame
from backend.policy_servers.smolvla_server import CHECKPOINT, REVISION, VLM_REVISION
from backend.shared.schemas import (
    LightingShift,
    LightingShiftParams,
    ObjectNudge,
    ObjectNudgeParams,
    PolicyActRequest,
    PolicyActResponse,
    PolicyHealthResponse,
)

REPLAY_STRIDE = 4


@contextmanager
def policy_server(client: httpx.Client, port: int, seed: int):
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "backend.policy_servers.smolvla_server",
            "--port",
            str(port),
            "--seed",
            str(seed),
        ],
    )
    try:
        deadline = monotonic() + 600
        while monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"SmolVLA server exited with status {process.returncode}")
            try:
                response = client.get("/health")
                response.raise_for_status()
                health = PolicyHealthResponse.model_validate_json(response.content)
                break
            except httpx.HTTPError:
                sleep(1)
        else:
            raise TimeoutError("Timed out loading SmolVLA")
        yield health
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def scene_evidence(scene: Scene, env: LiberoEnvAdapter) -> dict:
    return {
        "object_xyz_m": env.object_position,
        "main_luma": ImageStat.Stat(scene.observation.rgb_main.convert("L")).mean[0],
    }


def rollout(client, wrapper, env, args, *, name, precision, denoise_steps, effect=None):
    print(f"Starting {name} with {precision}/{denoise_steps} denoise steps", flush=True)
    wrapper.reset()
    scene = wrapper.observe()
    frames = [replay_frame(scene, case=name, step=0)]
    pending = deque()
    infer_ms, vram_mb = [], []
    injection = None
    start = perf_counter()

    while wrapper.step_count < args.max_steps and not env.success:
        if effect and wrapper.step_count == args.inject_step:
            before = scene_evidence(wrapper.observe(), env)
            wrapper.inject(effect)
            after_scene = wrapper.observe()
            injection = {
                "step": wrapper.step_count,
                "command": effect.model_dump(mode="json"),
                "before": before,
                "after": scene_evidence(after_scene, env),
            }
            frames.append(replay_frame(after_scene, case=name, step=wrapper.step_count))
            pending.clear()
        if not pending:
            observation = wrapper.observe().observation
            request = PolicyActRequest(
                rgb_main_b64=jpeg_b64(observation.rgb_main),
                rgb_wrist_b64=jpeg_b64(observation.rgb_wrist),
                proprio=observation.proprio,
                prompt=env.prompt,
                precision=precision,
                denoise_steps=denoise_steps,
            )
            latency = wrapper.latency_ms(0.0)
            if latency:
                sleep(latency / 1000)
            response = client.post("/act", json=request.model_dump(mode="json"))
            if response.is_error:
                print(f"Policy /act returned {response.status_code}: {response.text}", flush=True)
            response.raise_for_status()
            result = PolicyActResponse.model_validate_json(response.content)
            infer_ms.append(result.infer_ms + latency)
            vram_mb.append(result.vram_mb)
            pending.extend(result.action_chunk[: args.chunk_actions])

        scene = wrapper.step(pending.popleft())
        if wrapper.step_count % REPLAY_STRIDE == 0 or env.success:
            frames.append(replay_frame(scene, case=name, step=wrapper.step_count))
    elapsed = perf_counter() - start

    if effect and injection is None:
        raise RuntimeError(f"{name} finished before injection step {args.inject_step}")
    frames[0].save(
        args.output / f"{name}.gif",
        save_all=True,
        append_images=frames[1:],
        duration=REPLAY_STRIDE * 50,
        loop=0,
    )
    result = {
        "case": name,
        "precision": precision,
        "denoise_steps": denoise_steps,
        "success": env.success,
        "steps": wrapper.step_count,
        "injection": injection,
        "policy_calls": len(infer_ms),
        "infer_ms_mean": float(np.mean(infer_ms)),
        "infer_ms_p50": float(np.percentile(infer_ms, 50)),
        "infer_ms_p95": float(np.percentile(infer_ms, 95)),
        "peak_policy_process_vram_mb": max(vram_mb),
        "rollout_wall_s": elapsed,
        "gif": f"{name}.gif",
    }
    print(
        f"{name}: success={env.success} steps={wrapper.step_count} wall={elapsed:.1f}s", flush=True
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--initial-state-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=280)
    parser.add_argument("--inject-step", type=int, default=20)
    parser.add_argument("--chunk-actions", type=int, default=10)
    parser.add_argument("--policy-port", type=int, default=8001)
    args = parser.parse_args()
    if not 0 <= args.inject_step < args.max_steps or args.chunk_actions < 1:
        parser.error("Expected 0 <= inject-step < max-steps and chunk-actions >= 1")
    args.output.mkdir(parents=True, exist_ok=True)
    cases = [
        ("baseline", "bf16", 10, None),
        (
            "object_nudge",
            "bf16",
            10,
            ObjectNudge(skill="object_nudge", params=ObjectNudgeParams(dx=0.12, dy=-0.08)),
        ),
        (
            "lighting_shift",
            "bf16",
            10,
            LightingShift(
                skill="lighting_shift",
                params=LightingShiftParams(brightness=0.18, saturation=0.7, duration_steps=600),
            ),
        ),
        ("int8_low_steps", "int8", 2, None),
    ]
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        text=True,
    ).strip()
    from huggingface_hub import snapshot_download

    print("Fetching pinned LIBERO assets", flush=True)
    snapshot_download(
        ASSETS_REPO,
        revision=ASSETS_REVISION,
        repo_type="dataset",
        local_dir=Path.home() / ".cache/libero/assets",
    )
    with (
        httpx.Client(
            base_url=f"http://127.0.0.1:{args.policy_port}", timeout=60, trust_env=False
        ) as client,
        policy_server(client, args.policy_port, args.seed) as health,
    ):
        env = LiberoEnvAdapter.from_task(
            args.suite, args.task_id, initial_state_id=args.initial_state_id, seed=args.seed
        )
        try:
            wrapper = ChaosWrapper(env)
            if env.object_name is None:
                raise RuntimeError("Select a task with a movable LIBERO object for the nudge case")
            runs = [
                rollout(
                    client,
                    wrapper,
                    env,
                    args,
                    name=name,
                    precision=precision,
                    denoise_steps=steps,
                    effect=effect,
                )
                for name, precision, steps, effect in cases
            ]
        finally:
            env.close()
    metadata = {
        "policy": health.model_dump(mode="json"),
        "checkpoint": CHECKPOINT,
        "checkpoint_revision": REVISION,
        "vlm_config_revision": VLM_REVISION,
        "libero_assets_revision": ASSETS_REVISION,
        "gpu": gpu,
        "renderer": environ.get("MUJOCO_GL", "egl"),
        "source_archive_sha256": environ.get("BMVLA_SOURCE_SHA256"),
        "versions": {name: version(name) for name in ("torch", "lerobot", "hf-libero", "mujoco")},
        "suite": args.suite,
        "task_id": args.task_id,
        "task": env.prompt,
        "initial_state_id": args.initial_state_id,
        "object": env.object_name,
        "seed": args.seed,
        "policy_noise": "fixed seeded Gaussian tensor reused on every replan",
        "max_steps_per_case": args.max_steps,
        "chunk_actions": args.chunk_actions,
        "int8_precision": "per-output-channel INT8 MLP and projection weights, BF16 arithmetic",
        "runs": runs,
    }
    (args.output / "metrics.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()

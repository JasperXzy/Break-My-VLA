"""Prepare isolated M4 runtimes, export plans and run the complete Auto-Chaos matrix."""

import argparse
import os
import subprocess
import sys
from contextlib import closing
from itertools import cycle, islice
from pathlib import Path

from backend.policy_servers import GROOT_DEFAULT_SUITE, GROOT_SUITES, POLICY_MODULES, TRT_PRECISIONS
from scripts.m3_gpu_runtime import (
    PI05_PACKAGES,
    ROOT,
    SIM_PACKAGES,
    install_runtime,
    rendering_backend,
    uv_command,
)

NVIDIA_INDEX = "https://pypi.nvidia.com"
TRT_PACKAGES = ("tensorrt-cu12==10.16.1.11", "nvidia-modelopt[onnx]==0.43.0")
GROOT_SOURCE = "https://github.com/NVIDIA/Isaac-GR00T.git"
GROOT_REVISION = "708617f5716c63fc0a327391492ea2b91977ae90"
GROOT_PACKAGES = (
    "albumentations==1.4.18",
    "opencv-python-headless==4.11.0.86",
    "diffusers==0.35.1",
    "dm-tree>=0.1.8,<1",
    "numpy==1.26.4",
    "pandas==2.2.3",
    "scipy==1.15.3",
    "transformers==4.57.3",
    "tyro==0.9.17",
    "accelerate>=1.2,<2",
)
GROOT_IMPORT = (
    "from pathlib import Path; import site, sys; "
    "Path(site.getsitepackages()[0], 'bmvla_isaac_gr00t.pth').write_text(sys.argv[1] + '\\n'); "
    "sys.path.insert(0, sys.argv[1]); "
    "from gr00t.policy.gr00t_policy import Gr00tPolicy; import gr00t.model"
)


def install_groot_source(python: Path, source: Path) -> None:
    # The upstream package metadata covers training on Python 3.12. Use its
    # pinned inference source with an independent Python 3.11 environment.
    if not (source / ".git").exists():
        subprocess.run(["git", "init", str(source)], check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source, capture_output=True, text=True
    )
    if commit.stdout.strip() != GROOT_REVISION:
        subprocess.run(
            ["git", "fetch", "--depth", "1", GROOT_SOURCE, GROOT_REVISION], cwd=source, check=True
        )
        subprocess.run(["git", "checkout", "--detach", GROOT_REVISION], cwd=source, check=True)
    subprocess.run([python, "-c", GROOT_IMPORT, str(source.resolve())], check=True)


def capture_observations(argv) -> None:
    from backend.chaos_sim.chaos import ChaosWrapper
    from backend.chaos_sim.libero_env import LiberoEnvAdapter
    from benchmarks.auto_chaos import observation_request, options, prepare_libero

    parser = argparse.ArgumentParser(description="Capture clean LIBERO policy observations")
    parser.add_argument("--calibration-count", type=int, required=True)
    parser.add_argument("--observation-count", type=int, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("benchmark_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if min(args.calibration_count, args.observation_count) < 1:
        parser.error("Calibration and observation counts must be positive")
    benchmark_args = args.benchmark_args
    benchmark = options(benchmark_args[1:] if benchmark_args[:1] == ["--"] else benchmark_args)
    prepare_libero()
    start = benchmark.initial_state_id + 1
    tasks = cycle(benchmark.tasks)
    for path, count in (
        (args.calibration, args.calibration_count),
        (args.observations, args.observation_count),
    ):
        with path.open("w", encoding="utf-8") as target:
            for state_id, task in enumerate(islice(tasks, count), start=start):
                resource = LiberoEnvAdapter.from_task(
                    task.suite, task.task_id, initial_state_id=state_id, seed=benchmark.seed
                )
                with closing(resource) as env:
                    wrapper = ChaosWrapper(env)
                    wrapper.reset()
                    target.write(observation_request(wrapper).model_dump_json() + "\n")
        start += count


def options(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=",".join(POLICY_MODULES))
    parser.add_argument("--precisions", default="bf16,fp8,int8,nvfp4")
    parser.add_argument("--tasks", default="libero_object:0")
    parser.add_argument("--levels", default="0-5")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--pi05-tokenizer")
    parser.add_argument("--groot-suite", choices=GROOT_SUITES)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/m4_auto_chaos"))
    parser.add_argument("--runtime-dir", type=Path, default=Path.home() / ".cache/break-my-vla/m4")
    parser.add_argument("--calibration-count", type=int, default=4)
    parser.add_argument("--observation-count", type=int, default=2)
    parser.add_argument(
        "--skip-export", action="store_true", help="Use plans already in the output"
    )
    parser.add_argument("--install-osmesa", action="store_true", help="Allow apt fallback in Colab")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args(argv)
    args.models = tuple(dict.fromkeys(model.strip() for model in args.models.split(",")))
    args.precisions = tuple(dict.fromkeys(mode.strip() for mode in args.precisions.split(",")))
    if set(args.models) - POLICY_MODULES.keys() or set(args.precisions) - {*TRT_PRECISIONS, "int8"}:
        parser.error("Expected M4 models and precisions")
    if min(args.calibration_count, args.observation_count) < 1 or args.seed < 0:
        parser.error("Sample counts must be positive and seed must be nonnegative")
    suites = {
        task.partition(":")[0].strip() if ":" in task else "libero_object"
        for task in args.tasks.split(",")
    }
    if args.groot_suite is None:
        args.groot_suite = suites.pop() if len(suites) == 1 else GROOT_DEFAULT_SUITE
        if args.groot_suite not in GROOT_SUITES:
            args.groot_suite = GROOT_DEFAULT_SUITE
    return args


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["capture"]:
        capture_observations(argv[1:])
        return

    args = options(argv)
    output, runtime = args.output.resolve(), args.runtime_dir.resolve()
    trt_root = output / "tensorrt"
    export_modes = [mode for mode in args.precisions if mode in TRT_PRECISIONS]
    benchmark = [
        "--models",
        ",".join(args.models),
        "--precisions",
        ",".join(args.precisions),
        "--tasks",
        args.tasks,
        "--levels",
        args.levels,
        "--seed",
        str(args.seed),
        "--groot-suite",
        args.groot_suite,
        "--output",
        str(output),
    ]
    if args.max_steps is not None:
        benchmark.extend(("--max-steps", str(args.max_steps)))
    if args.pi05_tokenizer is not None:
        benchmark.extend(("--pi05-tokenizer", args.pi05_tokenizer))
    env = os.environ | {
        "MPLBACKEND": "Agg",
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    uv = uv_command()
    samples, policy_pythons = [], {}
    if args.mock:
        sim = [*uv, "run", "--python", "3.11", "python"]
        env["UV_PROJECT_ENVIRONMENT"] = str(runtime / "mock-venv-py311")
        benchmark.append("--mock")
    else:
        subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"],
            check=True,
        )
        runtime.mkdir(parents=True, exist_ok=True)
        packages = TRT_PACKAGES if export_modes else ()
        indexes = (NVIDIA_INDEX,) if packages else ()
        python = install_runtime(
            uv, runtime / "sim-venv-py311", SIM_PACKAGES + packages, extra_indexes=indexes
        )
        sim = [python]
        for model, requirements in (("pi05", PI05_PACKAGES), ("groot", GROOT_PACKAGES)):
            if model in args.models:
                policy_pythons[model] = install_runtime(
                    uv,
                    runtime / f"{model}-venv-py311",
                    requirements + packages,
                    extra_indexes=indexes,
                )
        renderer = rendering_backend(python, install_osmesa=args.install_osmesa)
        env.update({"MUJOCO_GL": renderer, "LIBERO_CONFIG_PATH": str(runtime / "libero-config")})
        if "groot" in policy_pythons:
            install_groot_source(policy_pythons["groot"], runtime / "isaac-gr00t")

        if export_modes and not args.skip_export:
            output.mkdir(parents=True, exist_ok=True)
            samples = [
                "--calibration",
                str(output / "calibration.jsonl"),
                "--observations",
                str(output / "heldout.jsonl"),
            ]
            subprocess.run(
                [
                    *sim,
                    "-m",
                    "scripts.m4_gpu_runtime",
                    "capture",
                    *samples,
                    "--calibration-count",
                    str(args.calibration_count),
                    "--observation-count",
                    str(args.observation_count),
                    "--",
                    *benchmark,
                ],
                cwd=ROOT,
                env=env,
                check=True,
            )
        benchmark.extend(("--trt-root", str(trt_root)))
        for model, python in policy_pythons.items():
            benchmark.extend(("--policy-python", f"{model}={python}"))

    if export_modes and not args.skip_export:
        for model in args.models:
            command = [policy_pythons[model]] if model in policy_pythons else sim
            export_args = [
                "--model",
                model,
                "--output",
                str(trt_root),
                "--seed",
                str(args.seed),
                "--precisions",
                ",".join(export_modes),
                "--groot-suite",
                args.groot_suite,
            ]
            if args.pi05_tokenizer is not None:
                export_args.extend(("--pi05-tokenizer", args.pi05_tokenizer))
            if args.mock:
                export_args.append("--mock")
            subprocess.run(
                [*command, "-m", "scripts.export_tensorrt", *export_args, *samples],
                cwd=ROOT,
                env=env,
                check=True,
            )
    subprocess.run([*sim, "-m", "benchmarks.auto_chaos", *benchmark], cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()

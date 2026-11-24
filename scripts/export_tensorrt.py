"""Export and profile a VLA's TensorRT action component against PyTorch BF16."""

import argparse
import sys
from contextlib import nullcontext
from copy import deepcopy
from itertools import cycle, islice
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.chaos_sim.media import jpeg_b64
from backend.policy_servers import groot_server, pi05_server, smolvla_server
from backend.policy_servers.lerobot_runtime import decode_image
from backend.policy_servers.mock_server import BASE_LATENCY_MS, PROFILES, noisy_chunk
from backend.policy_servers.trt_engine import (
    MANIFEST,
    ONNX_FILE,
    OUTPUT_NAME,
    PLAN_FILE,
    TRT_PRECISIONS,
    EngineManifest,
    TensorRTActionBackend,
    compute_action_alignment,
    precision_supported,
)
from backend.shared.schemas import PolicyActRequest
from benchmarks.auto_chaos import write_json

MODELS = ("smolvla", "pi05", "groot")
PROFILE_FILE = "profile.json"


def options(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, default="groot")
    parser.add_argument("--precisions", default=",".join(TRT_PRECISIONS))
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/m4_tensorrt"))
    parser.add_argument("--mock", action="store_true", help="Simulate each stage on CPU")
    parser.add_argument(
        "--calibration", type=Path, help="JSONL of policy requests for ModelOpt PTQ"
    )
    parser.add_argument("--observations", type=Path, help="Held-out policy requests in JSONL")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint")
    parser.add_argument("--revision")
    parser.add_argument(
        "--groot-suite", choices=groot_server.SUITES, default=groot_server.DEFAULT_SUITE
    )
    parser.add_argument("--pi05-tokenizer")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--workspace-mib", type=int, default=2048)
    parser.add_argument("--max-sequence-length", type=int, default=1024)
    args = parser.parse_args(argv)
    args.precisions = tuple(dict.fromkeys(mode.strip() for mode in args.precisions.split(",")))
    if set(args.precisions) - set(TRT_PRECISIONS):
        parser.error("--precisions must use bf16, fp8, or nvfp4")
    if min(args.iterations, args.workspace_mib, args.max_sequence_length) < 1:
        parser.error("--iterations, --workspace-mib and --max-sequence-length must be positive")
    if min(args.warmup, args.seed) < 0:
        parser.error("--warmup and --seed must be nonnegative")
    return args


def observations(path: Path | None, *, seed: int) -> list[PolicyActRequest]:
    if path is not None:
        samples = [
            PolicyActRequest.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not samples or any(len(sample.proprio) != 8 for sample in samples):
            raise ValueError(f"{path} must contain policy requests with 8-element LIBERO states")
        return samples

    rng = np.random.default_rng(seed)
    return [
        PolicyActRequest(
            rgb_main_b64=jpeg_b64(Image.fromarray(rng.integers(0, 256, (256, 256, 3), np.uint8))),
            rgb_wrist_b64=jpeg_b64(Image.fromarray(rng.integers(0, 256, (256, 256, 3), np.uint8))),
            proprio=tuple(rng.uniform(-0.1, 0.1, size=8).tolist()),
            prompt=prompt,
            precision="bf16",
            denoise_steps=10,
        )
        for prompt in ("Pick up the bowl.", "Place the bowl on the plate.")
    ]


def latency_stats(samples: list[float], *, source: str) -> dict:
    summary = {f"p{p}_ms": float(np.percentile(samples, p)) for p in (50, 95, 99)}
    return {
        "source": source,
        "samples": len(samples),
        "mean_ms": float(np.mean(samples)),
        **summary,
    }


def alignment_stats(reference, candidate) -> dict:
    values = [
        compute_action_alignment(ref, act) for ref, act in zip(reference, candidate, strict=True)
    ]
    return {
        key: max(row[key] for row in values)
        if key == "action_max_abs_diff_vs_bf16"
        else float(np.mean([row[key] for row in values]))
        for key in values[0]
    }


def run_mock(args, calibration, samples, output: Path) -> dict:
    reference = [noisy_chunk(sample, PROFILES["bf16"].noise) for sample in samples]
    report = {"checkpoint": "mock", "device": "cpu", "tiers": {}}
    for mode in args.precisions:
        tier = output / mode
        tier.mkdir(exist_ok=True)
        profile = PROFILES[mode]
        write_json(tier / "onnx.mock.json", {"mock": True, "precision": mode, "stage": "export"})
        write_json(
            tier / "engine.mock.json",
            {
                "mock": True,
                "precision": mode,
                "stage": "build",
                "source": "onnx.mock.json",
            },
        )
        candidate = [noisy_chunk(sample, profile.noise) for sample in samples]
        times = [
            BASE_LATENCY_MS + profile.ms_per_step * sample.denoise_steps
            for sample in islice(cycle(samples), args.iterations)
        ]
        report["tiers"][mode] = {
            "status": "mock",
            "backend": "tensorrt",
            "calibration_samples": len(calibration) if mode != "bf16" else 0,
            "onnx": f"{mode}/onnx.mock.json",
            "engine": f"{mode}/engine.mock.json",
            "alignment": alignment_stats(reference, candidate),
            "latency": latency_stats(times, source="simulated"),
        }
    return report


def create_policy(args):
    settings = {"device": args.device, "seed": args.seed}
    if args.model == "groot":
        checkpoint = args.checkpoint or groot_server.CHECKPOINT
        revision = args.revision or (
            groot_server.REVISION if checkpoint == groot_server.CHECKPOINT else None
        )
        policy = groot_server.GrootEngine(
            checkpoint=checkpoint, revision=revision, suite=args.groot_suite, **settings
        )
    elif args.model == "pi05":
        checkpoint = args.checkpoint or pi05_server.CHECKPOINT
        if checkpoint == pi05_server.LEGACY_CHECKPOINT:
            checkpoint = pi05_server.CHECKPOINT
        revision = args.revision or (
            pi05_server.REVISION if checkpoint == pi05_server.CHECKPOINT else None
        )
        policy = pi05_server.Pi05Engine(
            checkpoint=checkpoint, revision=revision, tokenizer=args.pi05_tokenizer, **settings
        )
    else:
        checkpoint = args.checkpoint or smolvla_server.CHECKPOINT
        revision = args.revision or (
            smolvla_server.REVISION if checkpoint == smolvla_server.CHECKPOINT else None
        )
        policy = smolvla_server.SmolVLAEngine(checkpoint=checkpoint, revision=revision, **settings)
    return policy, revision


def run_policy(policy, samples):
    return [
        policy.act(
            sample.model_copy(update={"precision": "bf16"}),
            decode_image(sample.rgb_main_b64),
            decode_image(sample.rgb_wrist_b64),
        )
        for sample in samples
    ]


def capture_component_inputs(policy, samples):
    component = policy.trt_component
    module = policy.model.get_submodule(component.path)
    inputs = []

    def capture(_module, args, kwargs):
        inputs.append(
            tuple(tensor.detach().to("cpu", copy=True) for tensor in component.bind(args, kwargs))
        )

    handle = module.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        results = run_policy(policy, samples)
    finally:
        handle.remove()
    if not inputs:
        raise RuntimeError(f"Policy did not execute {component.path}")
    return results, inputs


def additive_attention_mask(_module, args, kwargs):
    mask = kwargs.get("attention_mask")
    if mask is None:
        return None

    import torch

    if mask.dtype != torch.bool:
        return None
    hidden = args[0] if args else kwargs["hidden_states"]
    return args, kwargs | {
        "attention_mask": torch.where(
            mask, hidden.new_zeros(()), hidden.new_full((), torch.finfo(hidden.dtype).min)
        )
    }


def modelopt_config(mtq, mode: str, *, dit: bool) -> dict:
    if mode == "fp8":
        return deepcopy(mtq.FP8_DEFAULT_CFG)
    config = deepcopy(mtq.NVFP4_DEFAULT_CFG)
    if dit:
        # The GR00T DiT uses FP8 attention and NVFP4 feed-forward projections.
        config = deepcopy(mtq.FP8_DEFAULT_CFG)
        for layer in ("ff.net.0.proj", "ff.net.2"):
            for quantizer in ("input_quantizer", "weight_quantizer"):
                config["quant_cfg"][f"*transformer_blocks.*.{layer}.{quantizer}"] = deepcopy(
                    mtq.NVFP4_DEFAULT_CFG["quant_cfg"][f"*{quantizer}"]
                )
    return config


def export_onnx(policy, mode: str, batches: list[tuple], path: Path) -> None:
    import onnx
    import torch

    component = policy.trt_component
    dtype = torch.bfloat16 if mode == "bf16" else torch.float16
    dit = bool(component.sequence_inputs)

    class ExportModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = deepcopy(policy.model.get_submodule(component.path)).to(dtype=dtype)

        def forward(self, *inputs):
            if dit:
                return self.inner(**dict(zip(component.inputs, inputs, strict=True)))
            return self.inner(*inputs)

    def input_tensors(batch):
        return tuple(
            tensor.to(
                device=policy.device, dtype=dtype if tensor.is_floating_point() else tensor.dtype
            )
            for tensor in batch
        )

    wrapper = ExportModule().eval()
    handles = []
    if dit:
        handles = [
            block.attn1.register_forward_pre_hook(additive_attention_mask, with_kwargs=True)
            for block in wrapper.inner.transformer_blocks
        ]

    def export_context():
        if mode == "bf16":
            return nullcontext()

        import modelopt.torch.quantization as mtq
        from modelopt.torch.quantization.export_onnx import configure_linear_module_onnx_quantizers

        def calibrate(model):
            with torch.inference_mode():
                for batch in batches:
                    model(*input_tensors(batch))

        mtq.quantize(wrapper, modelopt_config(mtq, mode, dit=dit), forward_loop=calibrate)
        if mode == "nvfp4":
            for module in wrapper.modules():
                quantizer = getattr(module, "input_quantizer", None)
                if quantizer is not None and quantizer.block_sizes:
                    quantizer._trt_high_precision_dtype = "Half"
        return configure_linear_module_onnx_quantizers(wrapper)

    try:
        with export_context(), torch.inference_mode():
            torch.onnx.export(
                wrapper,
                input_tensors(batches[0]),
                str(path),
                input_names=list(component.inputs),
                output_names=[OUTPUT_NAME],
                dynamic_axes={name: {1: "vl_seq_len"} for name in component.sequence_inputs},
                opset_version=19,
                dynamo=False,
            )
    finally:
        for handle in handles:
            handle.remove()

    graph = onnx.load(str(path), load_external_data=False)
    operators = {node.op_type for node in graph.graph.node}
    if mode == "fp8" and not operators.intersection({"TRT_FP8QuantizeLinear", "QuantizeLinear"}):
        raise RuntimeError("ModelOpt did not export FP8 quantizers")
    if mode == "nvfp4":
        from modelopt.onnx.export import NVFP4QuantExporter

        if not {"TRT_FP4QDQ", "TRT_FP4DynamicQuantize"} <= operators:
            raise RuntimeError("ModelOpt did not export NVFP4 weight and activation quantizers")
        onnx.external_data_helper.load_external_data_for_model(graph, str(path.parent))
        graph = NVFP4QuantExporter.process_model(graph)
        data_file = "nvfp4.weights"
        onnx.save_model(
            graph,
            str(path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=data_file,
            size_threshold=1024,
        )


def build_engine(onnx_path: Path, plan_path: Path, component, batches, *, max_seq, workspace_mib):
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT rejected {onnx_path}\n{errors}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mib * 1024**2)
    if component.sequence_inputs:
        profile = builder.create_optimization_profile()
        for name in component.sequence_inputs:
            index = component.inputs.index(name)
            shape = tuple(batches[0][index].shape)
            if any(batch[index].shape[1] > max_seq for batch in batches):
                raise ValueError(f"--max-sequence-length {max_seq} is too small for {name}")
            low, high = (shape[0], 1, *shape[2:]), (shape[0], max_seq, *shape[2:])
            if not profile.set_shape(name, low, shape, high):
                raise ValueError(f"TensorRT rejected the optimization profile for {name}")
        if config.add_optimization_profile(profile) < 0:
            raise RuntimeError("TensorRT rejected the action optimization profile")
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError(f"TensorRT could not build {onnx_path}")
    plan_path.write_bytes(bytes(plan))


def profile_component(backend, batches, *, warmup: int, iterations: int) -> dict:
    torch = backend.torch
    with torch.cuda.device(backend.device), torch.inference_mode():
        inputs = [
            dict(zip(backend.component.inputs, (t.to(backend.device) for t in batch), strict=True))
            for batch in batches
        ]
        stream = torch.cuda.current_stream(backend.device)
        for sample in islice(cycle(inputs), warmup):
            backend.run(sample)
        stream.synchronize()
        start, stop = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        times = []
        for sample in islice(cycle(inputs), iterations):
            start.record(stream)
            backend.run(sample)
            stop.record(stream)
            stop.synchronize()
            times.append(start.elapsed_time(stop))
    return latency_stats(times, source="cuda_event")


def publish_tier(staging: Path, tier: Path, manifest: EngineManifest) -> None:
    tier.mkdir(exist_ok=True)
    (tier / MANIFEST).unlink(missing_ok=True)
    for path in staging.iterdir():
        path.replace(tier / path.name)
    write_json(tier / MANIFEST, manifest.as_json())


def run_gpu(args, calibration, samples, output: Path) -> dict:
    policy, revision = create_policy(args)
    component = policy.trt_component
    report = {
        "checkpoint": policy.name,
        "revision": revision,
        "device": str(policy.device),
        "capability": policy.capability,
        "component": component.path,
        "tiers": {},
    }
    modes = [mode for mode in args.precisions if precision_supported(mode, policy.capability)]
    for mode in args.precisions:
        if mode not in modes:
            report["tiers"][mode] = {
                "status": "skipped",
                "reason": f"Unsupported CUDA capability {policy.capability}",
            }
    if not modes:
        return report

    reference, profile_inputs = capture_component_inputs(policy, samples)
    if set(modes) - {"bf16"}:
        _, calibration_inputs = capture_component_inputs(policy, calibration)

    for mode in modes:
        batches = profile_inputs[:1] if mode == "bf16" else calibration_inputs
        with TemporaryDirectory(prefix=f".{mode}-", dir=output) as temp:
            staging = Path(temp)
            onnx_path, plan_path = staging / ONNX_FILE, staging / PLAN_FILE
            print(
                f"{args.model} {mode}: exporting {component.path} and building TensorRT", flush=True
            )
            with policy.torch.cuda.device(policy.device):
                export_onnx(policy, mode, batches, onnx_path)
                policy.torch.cuda.empty_cache()
                build_engine(
                    onnx_path,
                    plan_path,
                    component,
                    batches + profile_inputs,
                    max_seq=args.max_sequence_length,
                    workspace_mib=args.workspace_mib,
                )
            backend = TensorRTActionBackend(plan_path, device=policy.device, component=component)
            latency = profile_component(
                backend, profile_inputs, warmup=args.warmup, iterations=args.iterations
            )
            with backend.installed(policy.model):
                candidate = run_policy(policy, samples)
            del backend
            manifest = EngineManifest(
                args.model, policy.name, revision, mode, policy.capability, component
            )
            alignment = alignment_stats(
                [result.action_chunk for result in reference],
                [result.action_chunk for result in candidate],
            )
            publish_tier(staging, output / mode, manifest)
        report["tiers"][mode] = {
            "status": "built",
            "backend": "tensorrt",
            "calibration_samples": 0 if mode == "bf16" else len(calibration),
            "onnx": f"{mode}/{ONNX_FILE}",
            "engine": f"{mode}/{PLAN_FILE}",
            "alignment": alignment,
            "latency": latency,
            "policy_latency": latency_stats(
                [result.infer_ms for result in candidate], source="wall_clock"
            ),
        }
    return report


def main(argv=None) -> None:
    args = options(argv)
    output = args.output / args.model
    output.mkdir(parents=True, exist_ok=True)
    worker = run_mock if args.mock else run_gpu
    try:
        calibration = observations(args.calibration, seed=args.seed)
        samples = observations(args.observations, seed=args.seed + 1)
        report = worker(args, calibration, samples, output)
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"TensorRT export failed for {args.model}: {error}") from error
    write_json(
        output / PROFILE_FILE,
        {
            "schema_version": "1.0",
            "mode": "cpu_mock" if args.mock else "cuda",
            "model": args.model,
            "reference": {"backend": "pytorch", "precision": "bf16"},
            "alignment_samples": len(samples),
            "calibration_source": str(args.calibration) if args.calibration else "synthetic",
            "observation_source": str(args.observations) if args.observations else "synthetic",
            **report,
        },
    )
    print(f"TensorRT profile: {(output / PROFILE_FILE).resolve()}", flush=True)


if __name__ == "__main__":
    main()

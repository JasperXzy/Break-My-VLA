"""Serve a LeRobot Pi0.5 LIBERO checkpoint over the shared HTTP policy API."""

import argparse
from importlib import import_module
from pathlib import Path

import uvicorn
from fastapi import FastAPI

from backend.policy_servers.lerobot_runtime import LeRobotEngine, policy_app
from backend.policy_servers.trt_engine import precision_supported

CHECKPOINT = "lerobot/pi05_libero_finetuned_v044"
REVISION = "7152f13e34c6c115d808aea0e5d08aca74eac14b"
LEGACY_CHECKPOINT = "lerobot/pi05_libero_finetuned"


def load_nvfp4_backend(spec: str, capability: tuple[int, int], **options):
    if not precision_supported("nvfp4", capability):
        raise ValueError("The NVFP4/TensorRT backend requires an sm_120 Blackwell GPU")
    module, separator, factory = spec.partition(":")
    if not separator or not module or not factory:
        raise ValueError("Expected --nvfp4-backend module:factory")
    backend = getattr(import_module(module), factory)(**options)
    if not callable(backend):
        raise TypeError("The NVFP4 backend factory must return a callable policy")
    return backend


class Pi05Engine(LeRobotEngine):
    model_id = "pi05"
    denoise_field = "num_inference_steps"
    quantized_projections = (
        "model.action_in_proj",
        "model.action_out_proj",
        "model.time_mlp_in",
        "model.time_mlp_out",
    )

    def __init__(
        self,
        *,
        checkpoint: str,
        revision: str | None,
        device: str,
        seed: int,
        tokenizer: str | None = None,
        nvfp4_backend: str | None = None,
        trt_dir: Path | None = None,
    ) -> None:
        super().__init__(checkpoint=checkpoint, device=device, seed=seed)
        self._nvfp4 = None
        if nvfp4_backend is not None:
            self._nvfp4 = load_nvfp4_backend(
                nvfp4_backend,
                self.capability,
                checkpoint=checkpoint,
                revision=revision,
                device=device,
                seed=seed,
            )
            self.precisions += ("nvfp4",)

        from huggingface_hub import snapshot_download
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi05.configuration_pi05 import PI05Config
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors  # noqa: F401
        from safetensors.torch import load_model

        local_path = checkpoint
        if not Path(checkpoint).is_dir():
            local_path = snapshot_download(
                checkpoint, revision=revision, allow_patterns=["*.json", "*.safetensors"]
            )
        config = PreTrainedConfig.from_pretrained(local_path, cli_overrides=[f"--device={device}"])
        if not isinstance(config, PI05Config):
            raise ValueError(f"{checkpoint!r} must contain a LeRobot Pi0.5 config")
        main_key, wrist_key = libero_cameras(config.image_features)
        config.dtype = "bfloat16"
        config.compile_model = False  # The benchmark changes denoise steps and weight modules.
        config.gradient_checkpointing = False
        with self.device:
            policy = PI05Policy(config).eval()
        # The Pi05 override can return a model after a failed load. Use the LeRobot
        # checkpoint's safetensors names and enforce matching weights, including tied aliases.
        load_model(policy, str(Path(local_path) / "model.safetensors"), strict=True)
        overrides = {"device_processor": {"device": device}}
        if tokenizer is not None:
            overrides["tokenizer_processor"] = {"tokenizer_name": tokenizer}
        processors = make_pre_post_processors(config, local_path, preprocessor_overrides=overrides)
        self._set_policy(policy, processors, (main_key, wrist_key))
        if trt_dir is not None:
            self.load_tensorrt(trt_dir, revision=revision)


def libero_cameras(image_features) -> tuple[str, str]:
    cameras = [name for name in image_features if ".empty_camera_" not in name]
    if len(cameras) != 2:
        raise ValueError("Expected two real LIBERO camera features in the Pi0.5 checkpoint")
    return cameras[0], cameras[1]


def create_app(engine: Pi05Engine) -> FastAPI:
    return policy_app(engine, title="Break-My-VLA Pi0.5 Policy")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--revision")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tokenizer", help="Override the tokenizer named in the checkpoint")
    parser.add_argument("--nvfp4-backend", help="Optional sm_120 TensorRT module:factory")
    parser.add_argument("--trt-dir", type=Path, help="Directory of exported TensorRT tiers")
    args = parser.parse_args()
    checkpoint = CHECKPOINT if args.checkpoint == LEGACY_CHECKPOINT else args.checkpoint
    engine = Pi05Engine(
        checkpoint=checkpoint,
        revision=args.revision or (REVISION if checkpoint == CHECKPOINT else None),
        device=args.device,
        seed=args.seed,
        tokenizer=args.tokenizer,
        nvfp4_backend=args.nvfp4_backend,
        trt_dir=args.trt_dir,
    )
    uvicorn.run(create_app(engine), host="127.0.0.1", port=args.port, access_log=False)


if __name__ == "__main__":
    main()

"""Serve LIBERO-finetuned SmolVLA over the shared HTTP policy API."""

import argparse

import uvicorn
from fastapi import FastAPI

from backend.policy_servers.lerobot_runtime import LeRobotEngine, policy_app

CHECKPOINT = "lerobot/smolvla_libero"
REVISION = "a24f1f380461b665158d52acd140152c79e8b36d"
VLM_REPO = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
VLM_REVISION = "4cde9d208e19519c36917aa170fffc7e381aa0ad"


class SmolVLAEngine(LeRobotEngine):
    denoise_field = "num_steps"
    quantized_projections = (
        "model.state_proj",
        "model.action_in_proj",
        "model.action_out_proj",
        "model.action_time_mlp_in",
        "model.action_time_mlp_out",
    )

    def __init__(self, *, checkpoint: str, revision: str, device: str, seed: int) -> None:
        super().__init__(checkpoint=checkpoint, device=device, seed=seed)
        from huggingface_hub import snapshot_download
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: F401
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        local_path = snapshot_download(
            checkpoint, revision=revision, allow_patterns=["*.json", "*.safetensors"]
        )
        config = PreTrainedConfig.from_pretrained(local_path, cli_overrides=[f"--device={device}"])
        config.load_vlm_weights = False  # The complete finetuned weights are in the checkpoint.
        overrides = {"device_processor": {"device": device}}
        if config.vlm_model_name == VLM_REPO:
            config.vlm_model_name = snapshot_download(
                VLM_REPO, revision=VLM_REVISION, allow_patterns=["*.json", "*.txt"]
            )
            overrides["tokenizer_processor"] = {"tokenizer_name": config.vlm_model_name}
        base = SmolVLAPolicy.from_pretrained(local_path, config=config, strict=True).eval()
        processors = make_pre_post_processors(
            base.config, local_path, preprocessor_overrides=overrides
        )
        cameras = ("observation.images.camera1", "observation.images.camera2")
        self._set_policy(base, processors, cameras)


def create_app(engine: SmolVLAEngine) -> FastAPI:
    return policy_app(engine, title="Break-My-VLA SmolVLA Policy")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    engine = SmolVLAEngine(
        checkpoint=args.checkpoint, revision=args.revision, device=args.device, seed=args.seed
    )
    uvicorn.run(create_app(engine), host="127.0.0.1", port=args.port, access_log=False)


if __name__ == "__main__":
    main()

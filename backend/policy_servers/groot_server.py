"""Serve NVIDIA Isaac GR00T N1.7 LIBERO weights through the policy HTTP API."""

import argparse
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI

from backend.policy_servers.lerobot_runtime import CudaPolicyEngine, GPUUnavailable, policy_app
from backend.policy_servers.quantization import LinearWeights
from backend.policy_servers.trt_engine import ActionComponent
from backend.shared.schemas import PolicyActRequest

CHECKPOINT = "nvidia/GR00T-N1.7-LIBERO"
REVISION = "658540327860f93dd2cb9c1ed3e3b10fb792a0eb"
SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
DEFAULT_SUITE = "libero_object"
STATE_AXES = ("x", "y", "z", "roll", "pitch", "yaw")
ACTION_KEYS = (*STATE_AXES, "gripper")
LANGUAGE_KEY = "annotation.human.action.task_description"


def libero_observation(request: PolicyActRequest, main: np.ndarray, wrist: np.ndarray) -> dict:
    state = np.asarray(request.proprio, dtype=np.float32)[None, None, :]
    return {
        "video": {"image": main[None, None], "wrist_image": wrist[None, None]},
        "state": {key: state[..., i : i + 1] for i, key in enumerate(STATE_AXES)}
        | {"gripper": state[..., 6:]},
        "language": {LANGUAGE_KEY: [[request.prompt]]},
    }


def libero_actions(actions: dict) -> list[list[float]]:
    columns = [np.asarray(actions[key], dtype=np.float32) for key in ACTION_KEYS]
    if any(column.ndim != 3 or column.shape[0] != 1 or column.shape[-1] != 1 for column in columns):
        raise ValueError("Expected LIBERO action modalities of shape [1, horizon, 1]")
    chunk = np.concatenate(columns, axis=-1)[0]
    # The GR00T dataset uses 0 for closed and 1 for open. Robosuite uses +1 for closed.
    chunk[:, 6] = -np.sign(2 * chunk[:, 6] - 1)
    return chunk.tolist()


def checkpoint_path(checkpoint: str, revision: str | None, suite: str) -> Path:
    path = Path(checkpoint).expanduser()
    if path.is_dir():
        return path / suite if (path / suite).is_dir() else path

    from huggingface_hub import snapshot_download

    files = (
        "config.json",
        "embodiment_id.json",
        "model*.safetensors",
        "model.safetensors.index.json",
        "processor_config.json",
        "statistics.json",
    )
    return (
        Path(
            snapshot_download(
                checkpoint, revision=revision, allow_patterns=[f"{suite}/{file}" for file in files]
            )
        )
        / suite
    )


class GrootEngine(CudaPolicyEngine):
    model_id = "groot"

    def __init__(
        self,
        *,
        checkpoint: str = CHECKPOINT,
        revision: str | None = None,
        suite: str = DEFAULT_SUITE,
        device: str = "cuda:0",
        seed: int = 0,
        trt_dir: Path | None = None,
    ) -> None:
        super().__init__(checkpoint=checkpoint, device=device, seed=seed)
        revision = revision or (REVISION if checkpoint == CHECKPOINT else None)
        try:
            from gr00t.policy.gr00t_policy import Gr00tPolicy

            self.policy = Gr00tPolicy(
                embodiment_tag="LIBERO_PANDA",
                model_path=str(checkpoint_path(checkpoint, revision, suite)),
                device=str(self.device),
            )
        except (ImportError, OSError) as error:
            raise GPUUnavailable(
                f"Install the Isaac GR00T N1.7 inference dependencies ({error})"
            ) from error

        self.model = self.policy.model
        if not Path(checkpoint).expanduser().is_dir():
            self.name = f"{checkpoint}/{suite}"
        config = self.model.action_head.config
        masks = (
            ("image_mask", "backbone_attention_mask")
            if config.use_alternate_vl_dit
            else ("encoder_attention_mask",)
            if config.strict_vl_padding_mask
            else ()
        )
        action_mask = ("hidden_attention_mask",) if config.strict_action_padding_mask else ()
        self.trt_component = ActionComponent(
            "action_head.model",
            ("hidden_states", "encoder_hidden_states", "timestep", *masks, *action_mask),
            ("encoder_hidden_states", *masks),
        )
        self._weights = LinearWeights(
            self.model, self.torch, self.device, projections=(), prefixes=("action_head.model.",)
        )
        if trt_dir is not None:
            self.load_tensorrt(trt_dir, revision=revision)

    def _predict_actions(self, request: PolicyActRequest, main: np.ndarray, wrist: np.ndarray):
        torch = self.torch
        self.model.action_head.num_inference_timesteps = request.denoise_steps
        with torch.cuda.device(self.device), torch.random.fork_rng(devices=[self.device]):
            torch.cuda.manual_seed(self._seed)
            self.policy.reset()
            actions, _ = self.policy.get_action(libero_observation(request, main, wrist))
        return libero_actions(actions)


def create_app(engine: GrootEngine) -> FastAPI:
    return policy_app(engine, title="Break-My-VLA Isaac GR00T Policy")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--revision")
    parser.add_argument("--checkpoint-suite", choices=SUITES, default=DEFAULT_SUITE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trt-dir", type=Path, help="Directory of exported TensorRT tiers")
    args = parser.parse_args()
    try:
        engine = GrootEngine(
            checkpoint=args.checkpoint,
            revision=args.revision,
            suite=args.checkpoint_suite,
            device=args.device,
            seed=args.seed,
            trt_dir=args.trt_dir,
        )
    except GPUUnavailable as error:
        parser.error(str(error))
    uvicorn.run(create_app(engine), host="127.0.0.1", port=args.port, access_log=False)


if __name__ == "__main__":
    main()

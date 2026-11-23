"""Serve LIBERO-finetuned SmolVLA over the shared HTTP policy API."""

import argparse
from base64 import b64decode
from binascii import Error
from copy import deepcopy
from io import BytesIO
from threading import Lock
from time import perf_counter

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image

from backend.shared.schemas import PolicyActRequest, PolicyActResponse, PolicyHealthResponse

CHECKPOINT = "lerobot/smolvla_libero"
REVISION = "a24f1f380461b665158d52acd140152c79e8b36d"
VLM_REPO = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
VLM_REVISION = "4cde9d208e19519c36917aa170fffc7e381aa0ad"
STATE_SIZE = 8
PRECISIONS = ("bf16", "int8")


def decode_image(encoded: str) -> np.ndarray:
    with Image.open(BytesIO(b64decode(encoded, validate=True))) as image:
        return np.asarray(image.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR))


def quantize_linear_weights(policy, torch):
    """Use signed per-output-channel INT8 weights and BF16 linear arithmetic."""

    class WeightOnlyLinear(torch.nn.Module):
        def __init__(self, linear):
            super().__init__()
            weight = linear.weight.detach().float()
            scale = weight.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127
            self.register_buffer("weight_int8", (weight / scale).round().to(torch.int8))
            self.register_buffer("scale", scale.to(torch.bfloat16))
            bias = linear.bias.detach().to(torch.bfloat16) if linear.bias is not None else None
            self.register_buffer("bias", bias)

        def forward(self, activations):
            weight = self.weight_int8.to(self.scale.dtype) * self.scale
            return torch.nn.functional.linear(activations.to(weight.dtype), weight, self.bias)

    copied = deepcopy(policy)
    targets = [
        (name, module)
        for name, module in copied.named_modules()
        if isinstance(module, torch.nn.Linear)
        and (
            ".mlp." in name
            or name.startswith(
                (
                    "model.state_proj",
                    "model.action_in_proj",
                    "model.action_out_proj",
                    "model.action_time_mlp",
                )
            )
        )
    ]
    if not targets:
        raise RuntimeError("The checkpoint has no matching INT8 Linear layers")
    for name, module in targets:
        parent, _, child = name.rpartition(".")
        setattr(copied.get_submodule(parent), child, WeightOnlyLinear(module))
    print(f"INT8-quantized {len(targets)} linear layers", flush=True)
    return copied


class SmolVLAEngine:
    def __init__(self, *, checkpoint: str, revision: str, device: str, seed: int) -> None:
        import torch
        from huggingface_hub import snapshot_download
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: F401
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        self.torch = torch
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("SmolVLA inference requires a CUDA device")
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
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            base.config,
            local_path,
            preprocessor_overrides=overrides,
        )
        self.models = {"bf16": base, "int8": quantize_linear_weights(base, torch)}
        self.name = checkpoint
        self._lock = Lock()
        generator = torch.Generator(device=self.device).manual_seed(seed)
        self._noise = torch.randn(
            (1, config.chunk_size, config.max_action_dim), device=self.device, generator=generator
        )

    def act(
        self, request: PolicyActRequest, main: np.ndarray, wrist: np.ndarray
    ) -> PolicyActResponse:
        torch = self.torch
        with self._lock, torch.inference_mode():
            model = self.models[request.precision]
            model.config.num_steps = request.denoise_steps
            model.reset()
            torch.cuda.reset_peak_memory_stats(self.device)
            started = perf_counter()
            observation = self.preprocessor(
                {
                    "observation.images.camera1": torch.tensor(main).permute(2, 0, 1) / 255.0,
                    "observation.images.camera2": torch.tensor(wrist).permute(2, 0, 1) / 255.0,
                    "observation.state": torch.tensor(request.proprio, dtype=torch.float32),
                    "task": request.prompt,
                }
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                chunk = model.predict_action_chunk(observation, noise=self._noise)
            actions = self.postprocessor(chunk.float()).squeeze(0).tolist()
            torch.cuda.synchronize(self.device)
            return PolicyActResponse(
                action_chunk=actions,
                infer_ms=(perf_counter() - started) * 1000,
                vram_mb=torch.cuda.max_memory_allocated(self.device) / 1024**2,
            )


def create_app(engine: SmolVLAEngine) -> FastAPI:
    app = FastAPI(title="Break-My-VLA SmolVLA Policy")

    @app.get("/health")
    async def health() -> PolicyHealthResponse:
        return PolicyHealthResponse(
            model=engine.name, precisions=PRECISIONS, device=str(engine.device)
        )

    @app.post("/act")
    def act(request: PolicyActRequest) -> PolicyActResponse:
        if request.precision not in PRECISIONS:
            raise HTTPException(status_code=422, detail="SmolVLA supports bf16 and int8")
        if len(request.proprio) != STATE_SIZE:
            raise HTTPException(status_code=422, detail="Expected the 8-element LIBERO state")
        try:
            main, wrist = decode_image(request.rgb_main_b64), decode_image(request.rgb_wrist_b64)
        except (Error, OSError, ValueError) as error:
            raise HTTPException(
                status_code=422, detail="Expected Base64-encoded RGB images"
            ) from error
        return engine.act(request, main, wrist)

    return app


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

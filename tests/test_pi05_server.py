import sys
from contextlib import nullcontext
from threading import Lock
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import numpy as np
import pytest
from PIL import Image

from backend.chaos_sim.media import jpeg_b64
from backend.policy_servers import pi05_server
from backend.shared.schemas import PolicyActRequest, PolicyActResponse

RESULT = PolicyActResponse(action_chunk=[[0] * 7, [1] * 7], infer_ms=20, vram_mb=2048)
CAMERAS = ("observation.images.image", "observation.images.image2")


@pytest.fixture
def request_json():
    return {
        "rgb_main_b64": jpeg_b64(Image.new("RGB", (32, 48), (60, 0, 0))),
        "rgb_wrist_b64": jpeg_b64(Image.new("RGB", (32, 48), (0, 60, 0))),
        "proprio": [0.0] * 8,
        "prompt": "Put the bowl on the plate.",
        "precision": "fp8",
        "denoise_steps": 3,
    }


def test_pi05_http_contract(serve_app, request_json):
    engine = SimpleNamespace(
        name=pi05_server.CHECKPOINT,
        precisions=("bf16", "fp8", "int8"),
        device="cuda:0",
        act=Mock(return_value=RESULT),
    )
    with serve_app(pi05_server.create_app(engine)) as url, httpx.Client(base_url=url) as http:
        assert http.get("/health").json() == {
            "model": pi05_server.CHECKPOINT,
            "precisions": ["bf16", "fp8", "int8"],
            "device": "cuda:0",
        }
        response = http.post("/act", json=request_json)
        assert PolicyActResponse.model_validate_json(response.content) == RESULT
        request, main, wrist = engine.act.call_args.args
        assert request.denoise_steps == 3
        assert request.precision == "fp8"
        assert main.shape == wrist.shape == (256, 256, 3)
        assert main[0, 0, 0] > 55 and wrist[0, 0, 1] > 55

        engine.act.reset_mock()
        for bad_input in ({"precision": "nvfp4"}, {"proprio": [0]}, {"rgb_main_b64": "??"}):
            assert http.post("/act", json=request_json | bad_input).status_code == 422
        engine.act.assert_not_called()


def test_pi05_camera_keys_follow_checkpoint_and_skip_empty_views():
    assert pi05_server.libero_cameras((*CAMERAS, "observation.images.empty_camera_0")) == CAMERAS
    with pytest.raises(ValueError, match="two real LIBERO camera"):
        pi05_server.libero_cameras(CAMERAS[:1])


def test_pi05_runtime_uses_seeded_noise_and_configures_denoising(monkeypatch, request_json):
    torch = pytest.importorskip("torch", reason="Install optional PyTorch for CPU tensor tests")
    engine = pi05_server.Pi05Engine.__new__(pi05_server.Pi05Engine)
    engine.torch, engine.device, engine._seed = torch, torch.device("cpu"), 42
    engine._lock, engine._nvfp4 = Lock(), None

    policy = torch.nn.Module()
    policy.model = torch.nn.Module()
    policy.model.action_in_proj = torch.nn.Linear(8, 7)
    policy.config = SimpleNamespace(chunk_size=2, max_action_dim=32, num_inference_steps=10)
    policy.reset = Mock()
    policy.predict_action_chunk = Mock(return_value=torch.zeros(1, 2, 7))
    engine._set_policy(policy, (lambda obs: obs, lambda actions: actions), CAMERAS)
    monkeypatch.setattr(torch, "autocast", Mock(return_value=nullcontext()))
    monkeypatch.setattr(torch.cuda, "synchronize", Mock())
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", Mock())
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", Mock(return_value=1024**2))

    request = PolicyActRequest.model_validate(request_json)
    main = np.full((256, 256, 3), 255, dtype=np.uint8)
    wrist = np.zeros_like(main)
    result = engine.act(request, main, wrist)
    assert result.action_chunk == ((0.0,) * 7,) * 2
    assert result.vram_mb == 1.0
    policy.reset.assert_called_once()
    assert policy.config.num_inference_steps == 3
    assert policy.model.action_in_proj.weight.dtype == torch.float8_e4m3fn
    observation = policy.predict_action_chunk.call_args.args[0]
    assert observation["task"] == request.prompt
    assert observation["observation.state"].shape == (8,)
    assert observation[CAMERAS[0]].mean() == 1
    assert observation[CAMERAS[1]].sum() == 0
    noise = policy.predict_action_chunk.call_args.kwargs["noise"]
    assert noise.shape == (1, 2, 32)
    engine.act(request, main, wrist)
    assert policy.predict_action_chunk.call_args.kwargs["noise"] is noise


def test_nvfp4_hook_is_blackwell_only_and_handles_act(monkeypatch, request_json):
    act = Mock(return_value=RESULT)
    factory = Mock(return_value=act)
    monkeypatch.setitem(sys.modules, "fake_tensorrt", SimpleNamespace(make=factory))
    spec = "fake_tensorrt:make"
    with pytest.raises(ValueError, match="sm_120"):
        pi05_server.load_nvfp4_backend(spec, (8, 9))
    factory.assert_not_called()

    engine = pi05_server.Pi05Engine.__new__(pi05_server.Pi05Engine)
    engine._nvfp4 = pi05_server.load_nvfp4_backend(spec, (12, 0), device="cuda:0")
    engine._lock = Lock()
    factory.assert_called_once_with(device="cuda:0")
    request = PolicyActRequest.model_validate(request_json | {"precision": "nvfp4"})
    view = np.zeros((256, 256, 3), dtype=np.uint8)
    assert engine.act(request, view, view) == RESULT
    act.assert_called_once_with(request, view, view)

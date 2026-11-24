import builtins
import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import numpy as np
import pytest
from PIL import Image

from backend.chaos_sim.media import jpeg_b64
from backend.policy_servers import groot_server
from backend.policy_servers.lerobot_runtime import GPUUnavailable
from backend.shared.schemas import PolicyActRequest, PolicyActResponse, PolicyHealthResponse


@pytest.fixture
def request_json():
    return {
        "rgb_main_b64": jpeg_b64(Image.new("RGB", (40, 20), (60, 0, 0))),
        "rgb_wrist_b64": jpeg_b64(Image.new("RGB", (20, 40), (0, 60, 0))),
        "proprio": [0.1, 0.2, 0.3, -0.1, -0.2, -0.3, 0.04, -0.04],
        "prompt": "Pick up the black bowl.",
        "precision": "nvfp4",
        "denoise_steps": 3,
    }


def test_groot_http_contract(serve_app, request_json):
    result = PolicyActResponse(action_chunk=[[0] * 7], infer_ms=5, vram_mb=1)
    engine = SimpleNamespace(
        name=groot_server.CHECKPOINT,
        device="cuda:0",
        precisions=("bf16", "fp8", "int8", "nvfp4"),
        precision_backends=lambda: {"bf16": "pytorch", "nvfp4": "tensorrt"},
        act=Mock(return_value=result),
    )
    with serve_app(groot_server.create_app(engine)) as url, httpx.Client(base_url=url) as http:
        response = http.get("/health")
        assert PolicyHealthResponse.model_validate_json(response.content).precisions == (
            "bf16",
            "fp8",
            "int8",
            "nvfp4",
        )
        assert response.headers["X-Policy-Backend-nvfp4"] == "tensorrt"
        for mode in engine.precisions:
            response = http.post("/act", json=request_json | {"precision": mode})
            assert response.status_code == 200
            assert PolicyActResponse.model_validate_json(response.content) == result
            request, main, wrist = engine.act.call_args.args
            assert request.precision == mode and request.denoise_steps == 3
            assert main.shape == wrist.shape == (256, 256, 3)
            assert main[0, 0, 0] > 55 and wrist[0, 0, 1] > 55

        engine.precisions = ("bf16", "int8")
        engine.act.reset_mock()
        for invalid in (
            {},
            {"precision": "int8", "proprio": [0]},
            {"precision": "int8", "rgb_wrist_b64": "??"},
            {"precision": "int8", "denoise_steps": 11},
        ):
            assert http.post("/act", json=request_json | invalid).status_code == 422
        engine.act.assert_not_called()


def test_groot_policy_uses_libero_modalities_seed_and_denoise(request_json):
    engine = groot_server.GrootEngine.__new__(groot_server.GrootEngine)
    rng = Mock(return_value=nullcontext())
    seed = Mock()
    engine.torch = SimpleNamespace(
        cuda=SimpleNamespace(device=lambda _: nullcontext(), manual_seed=seed),
        random=SimpleNamespace(fork_rng=rng),
    )
    engine.device, engine._seed = "cuda:1", 27
    engine.model = SimpleNamespace(action_head=SimpleNamespace())
    actions = {
        key: np.full((1, 2, 1), i, dtype=np.float32)
        for i, key in enumerate(groot_server.STATE_AXES)
    } | {"gripper": np.array([[[0.2], [0.8]]])}
    engine.policy = Mock()
    engine.policy.get_action.return_value = (actions, {})
    main = np.zeros((256, 256, 3), dtype=np.uint8)
    wrist = np.full_like(main, 255)
    request = PolicyActRequest.model_validate(request_json)
    chunk = engine._predict_actions(request, main, wrist)
    assert chunk == [[0, 1, 2, 3, 4, 5, 1], [0, 1, 2, 3, 4, 5, -1]]
    assert engine.model.action_head.num_inference_timesteps == 3
    rng.assert_called_once_with(devices=["cuda:1"])
    seed.assert_called_once_with(27)
    engine.policy.reset.assert_called_once()
    (obs,) = engine.policy.get_action.call_args.args
    assert obs["language"] == {groot_server.LANGUAGE_KEY: [[request.prompt]]}
    np.testing.assert_array_equal(obs["video"]["image"][0, 0], main)
    np.testing.assert_array_equal(obs["video"]["wrist_image"][0, 0], wrist)
    axes = np.concatenate([obs["state"][key] for key in groot_server.ACTION_KEYS], axis=-1)
    np.testing.assert_allclose(axes, [[request.proprio]])
    assert axes.dtype == np.float32


def test_groot_downloads_only_the_selected_checkpoint(monkeypatch, tmp_path):
    download = Mock(return_value=str(tmp_path))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    path = groot_server.checkpoint_path(groot_server.CHECKPOINT, "sha", "libero_goal")
    assert path == tmp_path / "libero_goal"
    assert download.call_args.args == (groot_server.CHECKPOINT,)
    assert download.call_args.kwargs["revision"] == "sha"
    patterns = download.call_args.kwargs["allow_patterns"]
    assert "libero_goal/model*.safetensors" in patterns
    assert all(pattern.startswith("libero_goal/") for pattern in patterns)
    path.mkdir()
    download.reset_mock()
    assert groot_server.checkpoint_path(str(tmp_path), None, "libero_goal") == path
    download.assert_not_called()


def test_groot_unavailable_runtime_has_actionable_error(monkeypatch):
    original_import = builtins.__import__

    def no_optional_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("gr00t"):
            raise ModuleNotFoundError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_optional_import)
    with pytest.raises(GPUUnavailable, match="CUDA-enabled PyTorch"):
        groot_server.GrootEngine()

    def fake_cuda_init(engine, **kwargs):
        engine.device = "cuda:0"

    monkeypatch.setattr(groot_server.CudaPolicyEngine, "__init__", fake_cuda_init)
    with pytest.raises(GPUUnavailable, match="Isaac GR00T N1.7"):
        groot_server.GrootEngine()


def test_groot_constructs_dit_signature_for_local_libero_checkpoint(monkeypatch, tmp_path):
    def init_cuda(engine, **kwargs):
        engine.device = "cuda:0"
        engine.torch = object()
        engine.capability = (12, 0)
        engine.precisions = ("bf16", "fp8", "int8")

    model = SimpleNamespace(
        action_head=SimpleNamespace(
            config=SimpleNamespace(
                use_alternate_vl_dit=True,
                strict_action_padding_mask=True,
            )
        )
    )
    gr00t = Mock(return_value=SimpleNamespace(model=model))
    monkeypatch.setitem(
        sys.modules, "gr00t.policy.gr00t_policy", SimpleNamespace(Gr00tPolicy=gr00t)
    )
    monkeypatch.setattr(groot_server.CudaPolicyEngine, "__init__", init_cuda)
    weights = Mock()
    monkeypatch.setattr(groot_server, "LinearWeights", weights)
    checkpoint = tmp_path / "libero_object"
    checkpoint.mkdir()
    result = groot_server.GrootEngine(checkpoint=str(tmp_path))
    gr00t.assert_called_once_with(
        embodiment_tag="LIBERO_PANDA",
        model_path=str(checkpoint),
        device="cuda:0",
    )
    assert result.name == str(checkpoint)
    assert result.precisions == ("bf16", "fp8", "int8")
    assert result.trt_component.path == "action_head.model"
    assert result.trt_component.inputs == (
        "hidden_states",
        "encoder_hidden_states",
        "timestep",
        "image_mask",
        "backbone_attention_mask",
        "hidden_attention_mask",
    )
    assert result.trt_component.sequence_inputs == (
        "encoder_hidden_states",
        "image_mask",
        "backbone_attention_mask",
    )
    assert weights.call_args.kwargs["prefixes"] == ("action_head.model.",)

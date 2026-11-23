from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from PIL import Image

from backend.chaos_sim.media import jpeg_b64
from backend.policy_servers import smolvla_server
from backend.shared.schemas import PolicyActResponse, PolicyHealthResponse


@pytest.fixture(scope="module")
def client(serve_app):
    result = PolicyActResponse(action_chunk=[[0] * 7, [1] * 7], infer_ms=18, vram_mb=1320)
    engine = SimpleNamespace(name=smolvla_server.CHECKPOINT, device="cuda:0", act=Mock())
    engine.act.return_value = result
    with (
        serve_app(smolvla_server.create_app(engine)) as url,
        httpx.Client(base_url=url, trust_env=False) as http,
    ):
        yield http, engine


@pytest.fixture
def request_json():
    return {
        "rgb_main_b64": jpeg_b64(Image.new("RGB", (32, 48), (50, 0, 0))),
        "rgb_wrist_b64": jpeg_b64(Image.new("RGB", (32, 48), (0, 50, 0))),
        "proprio": [0, 0, 0.4, 0, 0, 1.0, 0.05, -0.05],
        "prompt": "Place the bowl on the plate.",
        "precision": "int8",
        "denoise_steps": 2,
    }


def test_smolvla_http_contract_decodes_both_cameras_and_passes_settings(client, request_json):
    http, engine = client
    response = http.get("/health")
    assert response.status_code == 200
    health = PolicyHealthResponse.model_validate_json(response.content)
    assert health.model == engine.name
    assert health.precisions == ("bf16", "int8")
    assert health.device == "cuda:0"
    response = http.post("/act", json=request_json)
    assert response.status_code == 200
    assert PolicyActResponse.model_validate_json(response.content) == engine.act.return_value
    request, main, wrist = engine.act.call_args.args
    assert request.precision == "int8"
    assert request.denoise_steps == 2
    assert main.shape == wrist.shape == (256, 256, 3)
    assert main[0, 0, 0] > 45
    assert wrist[0, 0, 1] > 45


@pytest.mark.parametrize(
    "replacement",
    [{"precision": "nvfp4"}, {"proprio": [0, 0, 0]}, {"rgb_wrist_b64": "!!!!"}],
)
def test_smolvla_act_rejects_unsupported_precision_state_and_images(
    client, request_json, replacement
):
    http, engine = client
    engine.act.reset_mock()
    assert http.post("/act", json=request_json | replacement).status_code == 422
    engine.act.assert_not_called()

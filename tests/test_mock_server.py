import json
from time import perf_counter
from unittest.mock import AsyncMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.policy_servers import mock_server
from backend.shared.schemas import PolicyActResponse, PolicyHealthResponse

REQUEST = {
    "rgb_main_b64": "AAAA",
    "rgb_wrist_b64": "AAAA",
    "proprio": [0.0, 0.0, 0.0],
    "prompt": "Pick up the orange block.",
    "precision": "bf16",
    "denoise_steps": 10,
}
PRECISIONS = ("bf16", "fp8", "int8", "nvfp4")


@pytest.fixture
def client():
    with TestClient(mock_server.app) as client:
        yield client


def test_health_contract(client):
    response = client.get("/health")
    assert response.status_code == 200
    health = PolicyHealthResponse.model_validate_json(response.content)
    assert health.model == "mock"
    assert health.device == "cpu"
    assert health.precisions == PRECISIONS


def test_act_contract_and_measured_delay(client):
    started_at = perf_counter()
    response = client.post("/act", json=REQUEST)
    elapsed_ms = (perf_counter() - started_at) * 1000
    assert response.status_code == 200
    result = PolicyActResponse.model_validate_json(response.content)
    assert np.shape(result.action_chunk) == (8, 7)
    assert np.max(np.abs(result.action_chunk)) <= 1.0
    assert 45 <= result.infer_ms <= elapsed_ms
    assert result.vram_mb == 0.0


def test_precision_and_denoise_change_delay_and_jitter(client, monkeypatch):
    sleep = AsyncMock()
    monkeypatch.setattr(mock_server, "sleep", sleep)
    last_duration, last_jitter = float("inf"), 0.0

    for precision in PRECISIONS:
        response = client.post("/act", json=REQUEST | {"precision": precision})
        assert response.status_code == 200
        duration = sleep.await_args.args[0]
        jitter = np.std(response.json()["action_chunk"])
        assert duration < last_duration
        assert jitter > last_jitter
        last_duration, last_jitter = duration, jitter

    reduced = client.post("/act", json=REQUEST | {"precision": "nvfp4", "denoise_steps": 1})
    assert reduced.status_code == 200
    assert sleep.await_args.args[0] < last_duration
    assert np.std(reduced.json()["action_chunk"]) > last_jitter
    assert (
        client.post("/act", json=REQUEST | {"precision": "nvfp4", "denoise_steps": 1})
        .json()["action_chunk"]
        == reduced.json()["action_chunk"]
    )


@pytest.mark.parametrize(
    "replacement",
    [{"precision": "fp16"}, {"denoise_steps": 0}, {"denoise_steps": 11}, {"proprio": None}],
)
def test_act_rejects_invalid_requests(client, replacement):
    assert client.post("/act", json=REQUEST | replacement).status_code == 422


def test_act_rejects_numeric_overflow_with_a_json_error(client):
    content = json.dumps(REQUEST | {"proprio": ["overflow"]}).replace('"overflow"', "1e1000")
    response = client.post("/act", content=content, headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "proprio", 0]

import json
from typing import get_args

import pytest
from pydantic import ValidationError

from backend.shared.schemas import (
    CLIENT_MESSAGE_ADAPTER,
    SERVER_MESSAGE_ADAPTER,
    ChaosSkill,
    ConfigMessage,
    ControlMessage,
    FrameMessage,
    PolicyActRequest,
    PolicyActResponse,
    PolicyHealthResponse,
)

ACT_INPUT = {
    "rgb_main_b64": "AAAA",
    "rgb_wrist_b64": "AAAA",
    "proprio": [0.0, 1.0],
    "prompt": "Pick up the orange block.",
    "precision": "bf16",
    "denoise_steps": 10,
}


def test_policy_wire_fields_round_trip_and_are_frozen():
    request = PolicyActRequest.model_validate(ACT_INPUT)
    assert json.loads(request.model_dump_json()) == ACT_INPUT
    assert PolicyActRequest.model_validate_json(request.model_dump_json()) == request
    assert request.proprio == (0.0, 1.0)
    with pytest.raises(ValidationError, match="frozen"):
        request.precision = "fp8"

    health = {"model": "mock", "precisions": ["bf16", "fp8", "int8", "nvfp4"], "device": "cpu"}
    assert PolicyHealthResponse.model_validate(health).model_dump(mode="json") == health

    response = {"action_chunk": [[0.0] * 7], "infer_ms": 30.5, "vram_mb": 0.0}
    model = PolicyActResponse.model_validate(response)
    assert model.action_chunk == ((0.0,) * 7,)
    assert model.model_dump(mode="json") == response


@pytest.mark.parametrize(
    "replacement",
    [
        {"denoise_steps": 0},
        {"denoise_steps": 11},
        {"denoise_steps": "3"},
        {"precision": "fp16"},
        {"proprio": [True]},
        {"proprio": [float("nan")]},
        {"proprio": []},
        {"unexpected": 1},
    ],
)
def test_policy_request_rejects_invalid_values(replacement):
    with pytest.raises(ValidationError):
        PolicyActRequest.model_validate(ACT_INPUT | replacement)


@pytest.mark.parametrize(
    "action_chunk",
    [[], [[0.0] * 6], [[0.0] * 6 + [float("inf")]]],
)
def test_policy_response_rejects_invalid_action_chunks(action_chunk):
    with pytest.raises(ValidationError):
        PolicyActResponse(action_chunk=action_chunk, infer_ms=0, vram_mb=0)


@pytest.mark.parametrize("skill", get_args(ChaosSkill))
def test_chaos_skill_dispatches_to_its_frozen_params(skill):
    message = CLIENT_MESSAGE_ADAPTER.validate_json(
        json.dumps({"type": "chaos", "skill": skill, "params": {}})
    )
    assert message.skill == skill
    assert message.params.model_config["frozen"]
    assert CLIENT_MESSAGE_ADAPTER.validate_json(message.model_dump_json()) == message


@pytest.mark.parametrize(
    "message",
    [
        {"type": "chaos", "skill": "teleport", "params": {}},
        {"type": "chaos", "skill": "object_nudge", "params": {"brightness": 1}},
        {"type": "chaos", "skill": "arm_kick", "params": {"dx": 2}},
        {"type": "chaos", "skill": "lighting_shift", "params": {"brightness": -1}},
        {"type": "chaos", "skill": "camera_shake", "params": {"amplitude_px": 81}},
        {"type": "chaos", "skill": "occlusion_patch", "params": {"height": 0}},
        {"type": "chaos", "skill": "friction_glitch", "params": {"multiplier": -1}},
        {"type": "chaos", "skill": "latency_inject", "params": {"latency_ms": 301}},
        {"type": "chaos", "skill": "wrist_blackout", "params": {"duration_steps": 0}},
        {"type": "config", "latency_ms": -1},
        {"type": "config", "denoise_steps": 11},
        {"type": "control", "action": "stop"},
    ],
)
def test_invalid_websocket_messages_are_rejected(message):
    with pytest.raises(ValidationError):
        CLIENT_MESSAGE_ADAPTER.validate_python(message)


def test_partial_config_and_control_wire_messages():
    assert CLIENT_MESSAGE_ADAPTER.validate_json(
        '{"type":"config","precision":"nvfp4","denoise_steps":1}'
    ) == ConfigMessage(precision="nvfp4", denoise_steps=1)
    for action in ("reset", "pause", "resume", "export_gif"):
        assert CLIENT_MESSAGE_ADAPTER.validate_python(
            {"type": "control", "action": action}
        ) == ControlMessage(action=action)


def test_frame_wire_fields_round_trip():
    frame = {
        "type": "frame",
        "step": 1,
        "fps": 19.0,
        "infer_ms": 40.0,
        "precision": "int8",
        "denoise_steps": 3,
        "rgb_main_b64": "AAAA",
        "rgb_wrist_b64": "AAAA",
        "action_chunk": [[0.0] * 7],
        "active_chaos": ["lighting_shift"],
        "task_progress": 0.25,
        "status": "running",
    }
    message = SERVER_MESSAGE_ADAPTER.validate_json(json.dumps(frame))
    assert isinstance(message, FrameMessage)
    assert message.model_dump(mode="json") == frame

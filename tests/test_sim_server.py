import asyncio
import json
from base64 import b64decode
from contextlib import contextmanager
from io import BytesIO
from threading import Event
from time import perf_counter

import httpx
import pytest
from PIL import Image
from websockets.sync.client import connect

from backend.chaos_sim.media import Recording
from backend.chaos_sim.server import create_app
from backend.policy_servers import mock_server
from backend.shared.schemas import (
    SERVER_MESSAGE_ADAPTER,
    ErrorMessage,
    FrameMessage,
    GifMessage,
    PolicyActRequest,
)


def policy_response(_request):
    return httpx.Response(
        200,
        json={
            "action_chunk": [[0.5, 0, 0, 0, 0, 0, -1.0]],
            "infer_ms": 5.0,
            "vram_mb": 0.0,
        },
    )


@pytest.fixture
def sim_server(serve_app):
    @contextmanager
    def serve(policy_transport=None):
        transport = policy_transport or httpx.MockTransport(policy_response)
        app = create_app(policy_url="http://policy", policy_transport=transport)
        with serve_app(app) as url, httpx.Client(base_url=url, trust_env=False) as http:
            yield http, url.replace("http://", "ws://", 1) + "/ws/session"

    return serve


def send(websocket, **message):
    websocket.send(json.dumps(message))


def receive(websocket):
    return SERVER_MESSAGE_ADAPTER.validate_json(websocket.recv(timeout=3))


def receive_frame(websocket, predicate=lambda _: True, *, limit=40):
    for _ in range(limit):
        frame = receive(websocket)
        assert isinstance(frame, FrameMessage)
        if predicate(frame):
            return frame
    pytest.fail(f"No matching frame in {limit} messages")


def decode_image(value):
    return Image.open(BytesIO(b64decode(value, validate=True)))


def test_streams_dual_camera_frames_and_completes_pick_with_mock_policy(sim_server):
    transport = httpx.ASGITransport(app=mock_server.app)
    with sim_server(transport) as (_, ws_url), connect(ws_url, open_timeout=3) as websocket:
        start = receive_frame(websocket)
        assert start.step == 0
        assert start.status == "running"
        assert start.task_progress == 0.0
        assert start.action_chunk == ()
        with (
            decode_image(start.rgb_main_b64) as main,
            decode_image(start.rgb_wrist_b64) as wrist,
        ):
            assert main.size == wrist.size == (320, 240)
            assert main.format == wrist.format == "JPEG"
            assert main.tobytes() != wrist.tobytes()

        moving = receive_frame(websocket)
        assert moving.step == 1
        assert moving.fps > 0
        assert moving.infer_ms > 0
        assert len(moving.action_chunk) == 8
        assert moving.task_progress > start.task_progress

        success = receive_frame(websocket, lambda frame: frame.status == "success")
        assert success.step > moving.step
        assert success.task_progress == 1.0


def test_ws_config_pause_resume_and_timed_chaos_reach_policy(sim_server):
    requests = []

    def capture(request):
        requests.append(PolicyActRequest.model_validate_json(request.content))
        return policy_response(request)

    with (
        sim_server(httpx.MockTransport(capture)) as (_, ws_url),
        connect(ws_url, open_timeout=3) as websocket,
    ):
        assert receive_frame(websocket).precision == "bf16"
        send(websocket, type="control", action="pause")
        paused = receive_frame(websocket, lambda frame: frame.status == "paused")
        assert paused.fps == 0.0
        send(
            websocket,
            type="config",
            precision="nvfp4",
            denoise_steps=2,
            latency_ms=25,
            prompt="Lift the block.",
        )
        configured = receive_frame(websocket)
        assert configured.status == "paused"
        assert configured.step == paused.step
        assert configured.precision == "nvfp4"
        assert configured.denoise_steps == 2

        send(websocket, type="chaos", skill="wrist_blackout", params={"duration_steps": 2})
        blackout = receive_frame(websocket)
        assert blackout.active_chaos == ("wrist_blackout",)
        assert blackout.step == paused.step
        with decode_image(blackout.rgb_wrist_b64) as wrist:
            assert wrist.getbbox() is None

        send(
            websocket,
            type="chaos",
            skill="latency_inject",
            params={"latency_ms": 140, "duration_steps": 1},
        )
        delayed = receive_frame(websocket)
        assert delayed.active_chaos == ("wrist_blackout", "latency_inject")
        started_at = perf_counter()
        send(websocket, type="control", action="resume")
        first = receive_frame(websocket)
        assert perf_counter() - started_at >= 0.13
        assert first.step == paused.step + 1
        assert first.status == "running"
        assert first.active_chaos == ("wrist_blackout", "latency_inject")
        assert first.infer_ms == 145

        second = receive_frame(websocket)
        assert second.active_chaos == ("wrist_blackout",)
        assert second.infer_ms == 30
        third = receive_frame(websocket)
        assert third.active_chaos == ()
        assert third.precision == "nvfp4"
        assert third.step == first.step + 2
        with decode_image(third.rgb_wrist_b64) as wrist:
            assert wrist.getbbox() is not None

        changed = next(request for request in requests if request.precision == "nvfp4")
        assert changed.denoise_steps == 2
        assert changed.prompt == "Lift the block."
        with decode_image(changed.rgb_wrist_b64) as wrist:
            assert wrist.getbbox() is None


@pytest.mark.parametrize(
    "payload",
    ["{", b'{"type":"control","action":"reset"}', '{"type":"config","denoise_steps":0}'],
)
def test_invalid_ws_message_gets_error_and_stream_continues(sim_server, payload):
    with sim_server() as (_, ws_url), connect(ws_url, open_timeout=3) as websocket:
        receive_frame(websocket)
        websocket.send(payload)
        for _ in range(10):
            message = receive(websocket)
            if isinstance(message, ErrorMessage):
                break
        else:
            pytest.fail("No error received")
        assert message.code == "invalid_message"
        assert receive_frame(websocket).step > 0


@pytest.mark.parametrize("action", ["pause", "reset", "disconnect"])
def test_commands_and_disconnect_cancel_inflight_policy_without_stale_steps(sim_server, action):
    started = Event()
    cancelled = Event()

    async def slow_first_request(request):
        if not started.is_set():
            started.set()
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                cancelled.set()
                raise
        return policy_response(request)

    with (
        sim_server(httpx.MockTransport(slow_first_request)) as (_, ws_url),
        connect(ws_url, open_timeout=3) as websocket,
    ):
        assert receive_frame(websocket).step == 0
        assert started.wait(timeout=2)
        if action == "disconnect":
            websocket.close()
        else:
            send(websocket, type="control", action=action)
            frame = receive_frame(websocket)
            assert frame.step == 0
            assert frame.status == ("paused" if action == "pause" else "running")
            assert frame.action_chunk == ()
        assert cancelled.wait(timeout=2)
        if action == "pause":
            send(websocket, type="control", action="resume")
        if action != "disconnect":
            assert receive_frame(websocket).step == 1


@pytest.mark.parametrize(
    "failure",
    [httpx.Response(503), httpx.Response(200, json={"action_chunk": []})],
)
def test_policy_failure_keeps_session_alive_and_recovers(sim_server, failure):
    attempts = 0

    def flaky_policy(request):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return failure
        return policy_response(request)

    with (
        sim_server(httpx.MockTransport(flaky_policy)) as (_, ws_url),
        connect(ws_url, open_timeout=3) as websocket,
    ):
        assert receive_frame(websocket).step == 0
        error_frame = receive_frame(websocket)
        assert error_frame.status == "policy_error"
        assert error_frame.step == 0
        assert error_frame.action_chunk == ()
        recovered = receive_frame(websocket)
        assert recovered.status == "running"
        assert recovered.step == 1


def test_sessions_have_independent_configs_and_effects(sim_server):
    with sim_server() as (_, ws_url), connect(ws_url, open_timeout=3) as first:
        receive_frame(first)
        send(first, type="control", action="pause")
        paused = receive_frame(first, lambda frame: frame.status == "paused")
        send(first, type="config", precision="fp8")
        assert receive_frame(first).precision == "fp8"
        send(first, type="chaos", skill="wrist_blackout", params={})
        assert receive_frame(first).active_chaos == ("wrist_blackout",)

        with connect(ws_url, open_timeout=3) as second:
            start = receive_frame(second)
            assert start.status == "running"
            assert start.step == 0
            assert start.precision == "bf16"
            assert start.active_chaos == ()
            send(second, type="control", action="reset")
            receive_frame(second, lambda frame: frame.step == 0)

        send(first, type="config", denoise_steps=4)
        frame = receive_frame(first)
        assert frame.step == paused.step
        assert frame.precision == "fp8"
        assert frame.active_chaos == ("wrist_blackout",)


def test_http_cors_and_gif_exports_via_both_entry_points(sim_server):
    with sim_server() as (http, ws_url):
        headers = {"Origin": "http://localhost:5173"}
        preflight = http.options(
            "/api/export_gif",
            headers=headers | {"Access-Control-Request-Method": "POST"},
        )
        assert preflight.status_code == 200
        assert preflight.headers["access-control-allow-origin"] == "*"
        assert http.post("/api/export_gif").status_code == 404

        with connect(ws_url, open_timeout=3) as websocket:
            receive_frame(websocket, lambda frame: frame.step >= 3)
            send(websocket, type="control", action="pause")
            receive_frame(websocket, lambda frame: frame.status == "paused")
            send(websocket, type="chaos", skill="wrist_blackout", params={})
            assert receive_frame(websocket).active_chaos == ("wrist_blackout",)
            send(websocket, type="control", action="export_gif")
            message = receive(websocket)
            assert isinstance(message, GifMessage)
            http_gif = http.post("/api/export_gif", headers=headers)
            assert http_gif.status_code == 200
            assert http_gif.headers["content-type"] == "image/gif"
            assert "break-my-vla.gif" in http_gif.headers["content-disposition"]
            assert http_gif.headers["access-control-allow-origin"] == "*"
            assert http_gif.content == b64decode(message.gif_b64, validate=True)
            with Image.open(BytesIO(http_gif.content)) as gif:
                assert gif.format == "GIF"
                assert gif.size == (320, 240)
                assert gif.n_frames >= 4

            send(websocket, type="control", action="reset")
            reset = receive_frame(websocket)
            assert reset.status == "paused"
            assert reset.step == 0
            assert reset.task_progress == 0
            assert reset.active_chaos == ()

        latest = http.post("/api/export_gif")
        assert latest.status_code == 200
        with Image.open(BytesIO(latest.content)) as gif:
            assert gif.n_frames == 1


def test_recording_keeps_only_latest_100_frames():
    recording = Recording()
    for index in range(105):
        recording.add(Image.new("RGB", (1, 1), (index, 0, 0)))
    with Image.open(BytesIO(asyncio.run(recording.export()))) as gif:
        assert gif.n_frames == 100
        assert gif.convert("RGB").getpixel((0, 0)) == (5, 0, 0)
        gif.seek(99)
        assert gif.convert("RGB").getpixel((0, 0)) == (104, 0, 0)

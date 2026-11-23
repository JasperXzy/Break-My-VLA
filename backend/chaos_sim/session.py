"""Own a WebSocket session's simulation state and interruptible policy calls."""

import asyncio
from base64 import b64encode
from collections.abc import Mapping
from dataclasses import dataclass, replace
from time import perf_counter

import httpx
from fastapi import WebSocket
from pydantic import ValidationError

from backend.chaos_sim.chaos import ChaosWrapper, Scene
from backend.chaos_sim.media import TARGET_FPS, Recording, jpeg_b64
from backend.chaos_sim.policies import fetch_policy_health
from backend.shared.schemas import (
    CLIENT_MESSAGE_ADAPTER,
    ConfigMessage,
    ControlMessage,
    ErrorMessage,
    FrameMessage,
    FrameStatus,
    GifMessage,
    PolicyActRequest,
    PolicyActResponse,
    Precision,
    ServerMessage,
)

POLICY_RETRY_SECONDS = 0.25


@dataclass(frozen=True, slots=True)
class SessionConfig:
    model: str
    precision: Precision = "bf16"
    denoise_steps: int = 10
    latency_ms: float = 0.0
    prompt: str = "Pick up the orange block."


@dataclass(frozen=True, slots=True)
class Inference:
    policy: PolicyActResponse
    injected_ms: float


class Session:
    def __init__(
        self,
        websocket: WebSocket,
        wrapper: ChaosWrapper,
        policy_client: httpx.AsyncClient,
        policy_servers: Mapping[str, str],
    ) -> None:
        self._websocket = websocket
        self._wrapper = wrapper
        self._policy_client = policy_client
        self._policy_servers = policy_servers
        default_config = SessionConfig(model=next(iter(policy_servers)))
        self._config = replace(
            default_config, prompt=getattr(wrapper.env, "prompt", default_config.prompt)
        )
        self._recording = Recording()
        self._paused = False
        self._policy_error = False
        self._inference: Inference | None = None
        self._last_step_at: float | None = None
        self._fps = 0.0

    async def run(self) -> None:
        await self._publish(self._wrapper.observe(), record=True)
        receive = asyncio.create_task(self._websocket.receive())
        infer: asyncio.Task[Inference | None] | None = None
        try:
            while True:
                if not self._paused and infer is None:
                    if self._last_step_at is None:
                        self._last_step_at = perf_counter()
                    infer = asyncio.create_task(self._infer())
                pending = (receive,) if infer is None else (receive, infer)
                done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

                # A state-changing command takes priority over an obsolete policy result.
                if receive in done:
                    event = receive.result()
                    if event["type"] == "websocket.disconnect":
                        return
                    receive = asyncio.create_task(self._websocket.receive())
                    invalidated = await self._handle_message(event.get("text"))
                    if invalidated and infer is not None:
                        infer.cancel()
                        await asyncio.gather(infer, return_exceptions=True)
                        infer = None

                if infer is not None and infer in done:
                    result = infer.result()
                    infer = None
                    await self._complete_step(result)
        finally:
            pending = (receive,) if infer is None else (receive, infer)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

    async def _infer(self) -> Inference | None:
        started_at = perf_counter()
        config = self._config
        observation = self._wrapper.observe().observation
        latency_ms = self._wrapper.latency_ms(config.latency_ms)
        request = PolicyActRequest(
            rgb_main_b64=jpeg_b64(observation.rgb_main),
            rgb_wrist_b64=jpeg_b64(observation.rgb_wrist),
            proprio=observation.proprio,
            prompt=config.prompt,
            precision=config.precision,
            denoise_steps=config.denoise_steps,
        )
        try:
            await asyncio.sleep(latency_ms / 1000)
            response = await self._policy_client.post(
                f"{self._policy_servers[config.model]}/act", json=request.model_dump(mode="json")
            )
            response.raise_for_status()
            result = PolicyActResponse.model_validate_json(response.content)
        except (httpx.HTTPError, ValidationError):
            await asyncio.sleep(POLICY_RETRY_SECONDS)
            return None
        await asyncio.sleep(max(0.0, 1 / TARGET_FPS - (perf_counter() - started_at)))
        return Inference(result, latency_ms)

    async def _complete_step(self, inference: Inference | None) -> None:
        self._inference = inference
        self._policy_error = inference is None
        if inference is None:
            self._restart_clock()
            await self._publish(self._wrapper.observe())
            return

        now = perf_counter()
        if self._last_step_at is not None:
            self._fps = 1 / (now - self._last_step_at)
        self._last_step_at = now
        scene = self._wrapper.step(inference.policy.action_chunk[0])
        await self._publish(scene, record=True)

    async def _handle_message(self, payload: str | None) -> bool:
        try:
            message = CLIENT_MESSAGE_ADAPTER.validate_json(payload)
        except ValidationError as error:
            issue = error.errors(include_input=False, include_url=False)[0]
            field = ".".join(str(part) for part in issue["loc"])
            detail = f"{field}: {issue['msg']}" if field else issue["msg"]
            await self._send(ErrorMessage(code="invalid_message", message=detail))
            return False

        match message:
            case ConfigMessage():
                if not await self._configure(message):
                    return False
            case ControlMessage(action="export_gif"):
                await self._export_gif()
                return False
            case ControlMessage(action="reset"):
                self._wrapper.reset()
                self._recording.clear()
                self._inference = None
                self._policy_error = False
                self._restart_clock()
                await self._publish(self._wrapper.observe(), record=True)
                return True
            case ControlMessage(action="pause"):
                self._paused = True
                self._restart_clock()
            case ControlMessage(action="resume"):
                if not self._paused:
                    return False
                self._paused = False
            case _:
                self._wrapper.inject(message)

        if self._paused:
            await self._publish(self._wrapper.observe())
        return True

    async def _configure(self, message: ConfigMessage) -> bool:
        changes = message.model_dump(exclude={"type"}, exclude_none=True)
        config = replace(self._config, **changes)
        if message.model is not None:
            url = self._policy_servers.get(config.model)
            if url is None:
                await self._send(ErrorMessage(
                    code="unknown_model",
                    message=f"Unknown model {config.model!r}. Choose from: "
                    + ", ".join(self._policy_servers),
                ))
                return False
            try:
                policy = await fetch_policy_health(self._policy_client, url)
            except (httpx.HTTPError, ValidationError):
                await self._send(ErrorMessage(
                    code="policy_unavailable",
                    message=f"Model {config.model!r} is unavailable. Its /health check failed.",
                ))
                return False
            if config.precision not in policy.precisions:
                if message.precision is not None:
                    await self._send(ErrorMessage(
                        code="unsupported_precision",
                        message=f"Model {config.model!r} does not support {config.precision}. "
                        f"Choose from: {', '.join(policy.precisions)}",
                    ))
                    return False
                config = replace(config, precision=policy.precisions[0])
            self._websocket.app.state.active_model = config.model

        if config == self._config:
            return False
        self._config = config
        self._inference = None
        self._policy_error = False
        return True

    def _restart_clock(self) -> None:
        self._last_step_at = None
        self._fps = 0.0

    async def _export_gif(self) -> None:
        data = await self._recording.export()
        if data is None:
            await self._send(ErrorMessage(code="gif_unavailable", message="No recorded frames"))
            return
        await self._send(GifMessage(gif_b64=b64encode(data).decode("ascii")))

    async def _publish(self, scene: Scene, *, record: bool = False) -> None:
        observation = scene.observation
        status: FrameStatus = "running"
        if self._paused:
            status = "paused"
        elif self._policy_error:
            status = "policy_error"
        elif observation.task_progress >= 1.0:
            status = "success"

        inference = self._inference
        message = FrameMessage(
            step=self._wrapper.step_count,
            fps=self._fps,
            infer_ms=inference.policy.infer_ms + inference.injected_ms if inference else 0.0,
            precision=self._config.precision,
            denoise_steps=self._config.denoise_steps,
            rgb_main_b64=jpeg_b64(observation.rgb_main),
            rgb_wrist_b64=jpeg_b64(observation.rgb_wrist),
            action_chunk=inference.policy.action_chunk if inference else (),
            active_chaos=scene.active_chaos,
            task_progress=observation.task_progress,
            status=status,
        )
        if record:
            self._recording.add(observation.rgb_main)
            self._websocket.app.state.latest_recording = self._recording
        await self._send(message)

    async def _send(self, message: ServerMessage) -> None:
        await self._websocket.send_text(message.model_dump_json())

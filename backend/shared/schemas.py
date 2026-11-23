"""Frozen JSON contracts for the policy HTTP API and simulator WebSocket."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

Precision = Literal["bf16", "fp8", "int8", "nvfp4"]
DenoiseSteps = Annotated[int, Field(ge=1, le=10)]
LatencyMs = Annotated[float, Field(ge=0, le=300)]
EncodedImage = Annotated[str, Field(min_length=1)]
Proprio = Annotated[tuple[float, ...], Field(strict=False, min_length=1)]
# LIBERO uses translation xyz, rotation xyz, and gripper for one action.
Action = Annotated[
    tuple[float, float, float, float, float, float, float],
    Field(strict=False),
]
ActionChunk = Annotated[tuple[Action, ...], Field(strict=False)]
ChaosSkill = Literal[
    "object_nudge",
    "arm_kick",
    "friction_glitch",
    "lighting_shift",
    "camera_shake",
    "occlusion_patch",
    "latency_inject",
    "wrist_blackout",
]
FrameStatus = Literal["running", "paused", "success", "policy_error"]


class WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


class PolicyHealthResponse(WireModel):
    model: str
    precisions: Annotated[tuple[Precision, ...], Field(strict=False, min_length=1)]
    device: str


class SimHealthResponse(PolicyHealthResponse):
    env: str


class PolicyActRequest(WireModel):
    rgb_main_b64: EncodedImage
    rgb_wrist_b64: EncodedImage
    proprio: Proprio
    prompt: str
    precision: Precision
    denoise_steps: DenoiseSteps


class PolicyActResponse(WireModel):
    action_chunk: Annotated[ActionChunk, Field(min_length=1)]
    infer_ms: Annotated[float, Field(ge=0)]
    vram_mb: Annotated[float, Field(ge=0)]


class FrameMessage(WireModel):
    type: Literal["frame"] = "frame"
    step: Annotated[int, Field(ge=0)]
    fps: Annotated[float, Field(ge=0)]
    infer_ms: Annotated[float, Field(ge=0)]
    precision: Precision
    denoise_steps: DenoiseSteps
    rgb_main_b64: EncodedImage
    rgb_wrist_b64: EncodedImage
    action_chunk: ActionChunk
    active_chaos: Annotated[tuple[ChaosSkill, ...], Field(strict=False)]
    task_progress: Annotated[float, Field(ge=0, le=1)]
    status: FrameStatus


Offset = Annotated[float, Field(ge=-0.5, le=0.5)]


class ObjectNudgeParams(WireModel):
    dx: Offset = 0.18
    dy: Offset = -0.12
    dz: Offset = 0.0


class ArmKickParams(WireModel):
    dx: Offset = -0.22
    dy: Offset = 0.10
    dz: Offset = 0.12


class TimedParams(WireModel):
    duration_steps: Annotated[int, Field(ge=1, le=600)] = 30


class FrictionGlitchParams(TimedParams):
    multiplier: Annotated[float, Field(ge=0, le=3)] = 0.10


class LightingShiftParams(TimedParams):
    brightness: Annotated[float, Field(ge=0, le=2.5)] = 0.45
    saturation: Annotated[float, Field(ge=0, le=2.5)] = 0.65


class CameraShakeParams(TimedParams):
    amplitude_px: Annotated[int, Field(ge=0, le=80)] = 14


class OcclusionPatchParams(TimedParams):
    x: Annotated[float, Field(ge=0, lt=1)] = 0.36
    y: Annotated[float, Field(ge=0, lt=1)] = 0.30
    width: Annotated[float, Field(gt=0, le=1)] = 0.29
    height: Annotated[float, Field(gt=0, le=1)] = 0.25


class LatencyInjectParams(TimedParams):
    latency_ms: LatencyMs = 160.0


class ChaosCommand(WireModel):
    type: Literal["chaos"] = "chaos"


class ObjectNudge(ChaosCommand):
    skill: Literal["object_nudge"]
    params: ObjectNudgeParams = ObjectNudgeParams()


class ArmKick(ChaosCommand):
    skill: Literal["arm_kick"]
    params: ArmKickParams = ArmKickParams()


class FrictionGlitch(ChaosCommand):
    skill: Literal["friction_glitch"]
    params: FrictionGlitchParams = FrictionGlitchParams()


class LightingShift(ChaosCommand):
    skill: Literal["lighting_shift"]
    params: LightingShiftParams = LightingShiftParams()


class CameraShake(ChaosCommand):
    skill: Literal["camera_shake"]
    params: CameraShakeParams = CameraShakeParams()


class OcclusionPatch(ChaosCommand):
    skill: Literal["occlusion_patch"]
    params: OcclusionPatchParams = OcclusionPatchParams()


class LatencyInject(ChaosCommand):
    skill: Literal["latency_inject"]
    params: LatencyInjectParams = LatencyInjectParams()


class WristBlackout(ChaosCommand):
    skill: Literal["wrist_blackout"]
    params: TimedParams = TimedParams()


ChaosMessage = Annotated[
    ObjectNudge
    | ArmKick
    | FrictionGlitch
    | LightingShift
    | CameraShake
    | OcclusionPatch
    | LatencyInject
    | WristBlackout,
    Field(discriminator="skill"),
]


class ConfigMessage(WireModel):
    type: Literal["config"] = "config"
    precision: Precision | None = None
    denoise_steps: DenoiseSteps | None = None
    latency_ms: LatencyMs | None = None
    prompt: str | None = None


class ControlMessage(WireModel):
    type: Literal["control"] = "control"
    action: Literal["reset", "pause", "resume", "export_gif"]


ClientMessage = Annotated[
    ChaosMessage | ConfigMessage | ControlMessage,
    Field(discriminator="type"),
]
CLIENT_MESSAGE_ADAPTER = TypeAdapter(ClientMessage)


class GifMessage(WireModel):
    type: Literal["gif"] = "gif"
    gif_b64: Annotated[str, Field(min_length=1)]


class ErrorMessage(WireModel):
    type: Literal["error"] = "error"
    code: Literal["invalid_message", "gif_unavailable"]
    message: str


ServerMessage = Annotated[FrameMessage | GifMessage | ErrorMessage, Field(discriminator="type")]
SERVER_MESSAGE_ADAPTER = TypeAdapter(ServerMessage)

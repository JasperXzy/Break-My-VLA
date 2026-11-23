"""Environment wrapper for physical, visual, and systems perturbations."""

from dataclasses import dataclass, replace
from math import cos, sin

from PIL import Image, ImageDraw, ImageEnhance

from backend.chaos_sim.env import Env, Observation
from backend.shared.schemas import (
    Action,
    ArmKick,
    CameraShake,
    ChaosMessage,
    ChaosSkill,
    FrictionGlitch,
    LatencyInject,
    LightingShift,
    ObjectNudge,
    OcclusionPatch,
    WristBlackout,
)


@dataclass(slots=True)
class ActiveEffect:
    command: ChaosMessage
    steps_left: int


@dataclass(frozen=True, slots=True)
class Scene:
    observation: Observation
    active_chaos: tuple[ChaosSkill, ...]


class ChaosWrapper:
    def __init__(self, env: Env) -> None:
        self.env = env
        self._effects: dict[ChaosSkill, ActiveEffect] = {}
        self.reset()

    def reset(self) -> None:
        self.env.reset()
        self.step_count = 0
        self._effects.clear()

    def inject(self, command: ChaosMessage) -> None:
        if isinstance(command, ObjectNudge):
            params = command.params
            self.env.nudge_object((params.dx, params.dy, params.dz))
            steps = 1
        elif isinstance(command, ArmKick):
            params = command.params
            self.env.kick_arm((params.dx, params.dy, params.dz))
            steps = 1
        else:
            steps = command.params.duration_steps
        self._effects[command.skill] = ActiveEffect(command, steps)

    def latency_ms(self, baseline: float) -> float:
        command = self._command("latency_inject")
        if isinstance(command, LatencyInject):
            return command.params.latency_ms
        return baseline

    def observe(self) -> Scene:
        observation = self.env.observe()
        main, wrist = observation.rgb_main, observation.rgb_wrist

        lighting = self._command("lighting_shift")
        if isinstance(lighting, LightingShift):
            params = lighting.params
            main = ImageEnhance.Brightness(main).enhance(params.brightness)
            wrist = ImageEnhance.Brightness(wrist).enhance(params.brightness)
            main = ImageEnhance.Color(main).enhance(params.saturation)
            wrist = ImageEnhance.Color(wrist).enhance(params.saturation)

        shake = self._command("camera_shake")
        if isinstance(shake, CameraShake):
            amplitude = shake.params.amplitude_px
            dx = round(amplitude * cos(self.step_count * 1.9))
            dy = round(amplitude * sin(self.step_count * 2.3 + 0.7))
            main = main.transform(main.size, Image.Transform.AFFINE, (1, 0, -dx, 0, 1, -dy))
            wrist = wrist.transform(wrist.size, Image.Transform.AFFINE, (1, 0, dx, 0, 1, dy))

        patch = self._command("occlusion_patch")
        if isinstance(patch, OcclusionPatch):
            params = patch.params
            width, height = main.size
            x, y = int(params.x * width), int(params.y * height)
            right = x + max(1, round(params.width * width)) - 1
            bottom = y + max(1, round(params.height * height)) - 1
            main = main.copy()
            ImageDraw.Draw(main).rectangle((x, y, right, bottom), fill="black")

        if isinstance(self._command("wrist_blackout"), WristBlackout):
            wrist = Image.new("RGB", wrist.size, "black")

        return Scene(
            replace(observation, rgb_main=main, rgb_wrist=wrist),
            tuple(self._effects),
        )

    def step(self, action: Action) -> Scene:
        friction = self._command("friction_glitch")
        multiplier = friction.params.multiplier if isinstance(friction, FrictionGlitch) else 1.0
        self.env.step(action, friction=multiplier)
        self.step_count += 1
        scene = self.observe()
        for skill in scene.active_chaos:
            effect = self._effects[skill]
            effect.steps_left -= 1
            if effect.steps_left == 0:
                del self._effects[skill]
        return scene

    def _command(self, skill: ChaosSkill) -> ChaosMessage | None:
        effect = self._effects.get(skill)
        return effect.command if effect else None

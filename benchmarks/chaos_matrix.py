"""Define the five Auto-Chaos dimensions and their stress levels."""

from dataclasses import dataclass

from backend.shared.schemas import (
    ArmKick,
    ArmKickParams,
    CameraShake,
    CameraShakeParams,
    ChaosMessage,
    FrictionGlitch,
    FrictionGlitchParams,
    LatencyInject,
    LatencyInjectParams,
    LightingShift,
    LightingShiftParams,
    ObjectNudge,
    ObjectNudgeParams,
    OcclusionPatch,
    OcclusionPatchParams,
    Precision,
    TimedParams,
    WristBlackout,
)

AXES = ("clean", "physical", "visual", "systems", "quantization")
FULL_STEPS = 10
DENOISE_BY_LEVEL = (FULL_STEPS, 8, 6, 4, 2, 1)
PRECISION_BITS: dict[Precision, int] = {"bf16": 16, "fp8": 8, "int8": 8, "nvfp4": 4}


@dataclass(frozen=True, slots=True)
class Case:
    axis: str
    level: int
    nominal_precision: Precision
    precision: Precision
    denoise_steps: int = FULL_STEPS
    command: ChaosMessage | None = None

    @property
    def tag(self) -> str:
        if self.command is not None:
            return self.command.skill
        if self.axis == "quantization":
            return "precision_drop" if self.precision != self.nominal_precision else "low_denoise"
        return "clean"


def perturbations(level: int) -> tuple[tuple[str, ChaosMessage], ...]:
    severity = level / 5
    duration = 6 * level
    return (
        (
            "physical",
            ObjectNudge(
                skill="object_nudge",
                params=ObjectNudgeParams(dx=0.12 * severity, dy=-0.08 * severity),
            ),
        ),
        (
            "physical",
            ArmKick(
                skill="arm_kick",
                params=ArmKickParams(dx=-0.22 * severity, dy=0.10 * severity, dz=0.12 * severity),
            ),
        ),
        (
            "physical",
            FrictionGlitch(
                skill="friction_glitch",
                params=FrictionGlitchParams(
                    multiplier=1 - 0.90 * severity, duration_steps=duration
                ),
            ),
        ),
        (
            "visual",
            LightingShift(
                skill="lighting_shift",
                params=LightingShiftParams(
                    brightness=1 - 0.82 * severity,
                    saturation=1 - 0.35 * severity,
                    duration_steps=duration,
                ),
            ),
        ),
        (
            "visual",
            CameraShake(
                skill="camera_shake",
                params=CameraShakeParams(
                    amplitude_px=round(14 * severity), duration_steps=duration
                ),
            ),
        ),
        (
            "visual",
            OcclusionPatch(
                skill="occlusion_patch",
                params=OcclusionPatchParams(
                    width=0.35 * severity,
                    height=0.30 * severity,
                    duration_steps=duration,
                ),
            ),
        ),
        (
            "systems",
            LatencyInject(
                skill="latency_inject",
                params=LatencyInjectParams(latency_ms=150.0, duration_steps=duration),
            ),
        ),
        (
            "systems",
            WristBlackout(skill="wrist_blackout", params=TimedParams(duration_steps=duration)),
        ),
    )


def stress_matrix(precisions: tuple[Precision, ...], levels: tuple[int, ...]) -> list[Case]:
    cases = []
    for nominal in precisions:
        cases.append(Case("clean", 0, nominal, nominal))
        lower = next(
            (
                mode
                for mode in PRECISION_BITS
                if mode in precisions and PRECISION_BITS[mode] < PRECISION_BITS[nominal]
            ),
            nominal,
        )
        for level in levels:
            if level == 0:
                continue
            cases.extend(
                Case(axis, level, nominal, nominal, command=command)
                for axis, command in perturbations(level)
            )
            cases.append(Case("quantization", level, nominal, lower, DENOISE_BY_LEVEL[level]))
    # Keep each weight tier active across adjacent rollouts.
    return sorted(cases, key=lambda case: precisions.index(case.precision))

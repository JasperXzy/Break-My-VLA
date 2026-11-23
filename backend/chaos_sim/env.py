"""Environment boundary used by the simulator and future LIBERO integration."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from PIL import Image

from backend.shared.schemas import Action, CameraShakeParams, LightingShiftParams

IMAGE_SIZE = (320, 240)
Vector3 = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class Observation:
    rgb_main: Image.Image
    rgb_wrist: Image.Image
    proprio: tuple[float, ...]
    task_progress: float


class Env(Protocol):
    def reset(self) -> None: ...

    def observe(self) -> Observation: ...

    def step(self, action: Action, *, friction: float) -> None: ...

    def nudge_object(self, offset: Vector3) -> None: ...

    def kick_arm(self, impulse: Vector3) -> None: ...


@runtime_checkable
class NativeCameraEffects(Protocol):
    def set_camera_effects(
        self,
        lighting: LightingShiftParams | None,
        shake: CameraShakeParams | None,
        step: int,
    ) -> None: ...

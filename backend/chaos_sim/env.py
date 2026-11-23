"""Environment boundary used by the simulator and future LIBERO integration."""

from dataclasses import dataclass
from typing import Protocol

from PIL import Image

from backend.shared.schemas import Action

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


class LiberoEnvAdapter(Env):
    """M2 entry point for adapting LIBERO observations and MuJoCo perturbations."""

    def __init__(self) -> None:
        raise NotImplementedError("LiberoEnvAdapter is reserved for M2")

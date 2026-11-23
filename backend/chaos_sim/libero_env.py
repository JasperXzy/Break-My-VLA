"""LIBERO adapter with native MuJoCo physics and dual camera observations."""

from importlib.util import find_spec
from json import dumps
from math import acos, cos, sin, sqrt
from os import environ
from pathlib import Path

import numpy as np
from PIL import Image

from backend.chaos_sim.env import Env, Observation, Vector3
from backend.shared.schemas import Action, CameraShakeParams, LightingShiftParams

CAMERAS = ("agentview", "robot0_eye_in_hand")
IMAGE_WIDTH = 256
SETTLE_ACTION = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)
ASSETS_REPO = "lerobot/libero-assets"
ASSETS_REVISION = "0b3ea86be5fe169d0fd036ae63d1070ec09e90f6"


class LiberoUnavailable(RuntimeError):
    """The optional native environment could not be imported."""


def libero_available() -> bool:
    return all(find_spec(package) is not None for package in ("libero", "mujoco", "torch"))


def configure_headless_libero(package: Path, config_dir: Path) -> None:
    """Supply the standard paths before LIBERO can prompt on first import."""
    config_dir.mkdir(parents=True, exist_ok=True)
    settings = {
        "benchmark_root": str(package),
        "bddl_files": str(package / "bddl_files"),
        "init_states": str(package / "init_files"),
        "datasets": str(package.parent / "datasets"),
        "assets": str(package / "assets"),
    }
    try:
        with (config_dir / "config.yaml").open("x") as config:
            config.write(dumps(settings))
    except FileExistsError:
        pass


def axis_angle(quat: np.ndarray) -> np.ndarray:
    """Convert the XYZW convention in LIBERO observations to LeRobot state."""
    w = float(np.clip(quat[3], -1.0, 1.0))
    denominator = sqrt(1.0 - w * w)
    if denominator < 1e-10:
        return np.zeros(3)
    return quat[:3] * (2.0 * acos(w) / denominator)


class LiberoEnvAdapter(Env):
    def __init__(self, env, initial_state, *, seed: int, settle_steps: int = 10) -> None:
        self._env = env
        self._initial_state = np.asarray(initial_state).copy()
        self._seed = seed
        self._settle_steps = settle_steps
        self.prompt: str = env.language_instruction
        self.success = False

        model = env.sim.model
        self._camera_ids = [model.camera_name2id(name) for name in CAMERAS]
        self._cam_pos = model.cam_pos[self._camera_ids].copy()
        self._cam_quat = model.cam_quat[self._camera_ids].copy()
        self._light_diffuse = model.light_diffuse.copy()
        self._camera_state = (None, None, 0)
        table_geoms, floor_geoms = [], []
        for i in range(model.ngeom):
            if not (model.geom_contype[i] or model.geom_conaffinity[i]):
                continue
            name = model.geom_id2name(i)
            body = model.body_id2name(model.geom_bodyid[i])
            if "table" in f"{body}/{name}":
                table_geoms.append(i)
            elif name == "floor" or body == "floor":
                floor_geoms.append(i)
        self._surface_geoms = table_geoms or floor_geoms
        self._friction = model.geom_friction[self._surface_geoms].copy()
        self._priority = model.geom_priority[self._surface_geoms].copy()
        self._force_body = model.site_bodyid[env.robots[0].eef_site_id]
        self._force = np.zeros(3)

        self.object_name, joint = self._find_free_joint()
        self._object_position = self._object_velocity = None
        if joint is not None:
            start, _ = model.get_joint_qpos_addr(joint)
            self._object_position = slice(start, start + 3)
            self._object_velocity = slice(*model.get_joint_qvel_addr(joint))

    @classmethod
    def from_task(
        cls,
        suite: str = "libero_spatial",
        task_id: int = 0,
        *,
        initial_state_id: int = 0,
        seed: int = 0,
    ) -> "LiberoEnvAdapter":
        if not libero_available():
            raise LiberoUnavailable("Install hf-libero, MuJoCo and PyTorch to use LIBERO")
        environ.setdefault("MUJOCO_GL", "egl")
        package = Path(next(iter(find_spec("libero").submodule_search_locations))) / "libero"
        config_dir = Path(environ.get("LIBERO_CONFIG_PATH", Path.home() / ".libero"))
        configure_headless_libero(package, config_dir)
        try:
            from libero.libero import benchmark
            from libero.libero.envs import OffScreenRenderEnv
        except (ImportError, OSError) as error:
            raise LiberoUnavailable(f"Could not import LIBERO/MuJoCo: {error}") from error

        suites = benchmark.get_benchmark_dict()
        if suite not in suites:
            raise ValueError(f"Unknown LIBERO suite {suite!r}")
        tasks = suites[suite]()
        if not 0 <= task_id < tasks.get_num_tasks():
            raise ValueError(f"Invalid task {task_id} for {suite}")
        states = tasks.get_task_init_states(task_id)
        if not 0 <= initial_state_id < len(states):
            raise ValueError(f"Invalid initial state {initial_state_id} for {suite} task {task_id}")
        env = OffScreenRenderEnv(
            bddl_file_name=tasks.get_task_bddl_file_path(task_id),
            use_camera_obs=False,
            hard_reset=False,
            ignore_done=True,
            camera_names=list(CAMERAS),
            camera_heights=IMAGE_WIDTH,
            camera_widths=IMAGE_WIDTH,
        )
        try:
            return cls(env, states[initial_state_id], seed=seed)
        except BaseException:
            env.close()
            raise

    def _find_free_joint(self) -> tuple[str | None, str | None]:
        model = self._env.sim.model
        objects = self._env.env.objects_dict
        for name in dict.fromkeys([*self._env.obj_of_interest, *objects]):
            obj = objects.get(name)
            if obj is None:
                continue
            for joint in obj.joints:
                address = model.get_joint_qpos_addr(joint)
                if isinstance(address, tuple) and address[1] - address[0] == 7:
                    return name, joint
        return None, None

    def reset(self) -> None:
        self.set_camera_effects(None, None, 0)
        self._force.fill(0.0)
        self._env.seed(self._seed)
        self._env.reset()
        self._raw = self._env.set_init_state(self._initial_state.copy())
        for _ in range(self._settle_steps):
            self._raw, _, _, _ = self._env.step(np.asarray(SETTLE_ACTION))
        self.success = bool(self._env.check_success())

    def close(self) -> None:
        self._env.close()

    @property
    def object_position(self) -> Vector3 | None:
        if self._object_position is None:
            return None
        x, y, z = self._env.sim.data.qpos[self._object_position]
        return float(x), float(y), float(z)

    def nudge_object(self, offset: Vector3) -> None:
        if self._object_position is None:
            return
        sim = self._env.sim
        sim.data.qpos[self._object_position] += offset
        sim.data.qvel[self._object_velocity] = 0.0
        sim.forward()
        self.success = bool(self._env.check_success())

    def kick_arm(self, impulse: Vector3) -> None:
        # xfrc_applied holds force (N) and torque. Apply force for one control tick.
        self._force = np.clip(self._force + 75.0 * np.asarray(impulse), -37.5, 37.5)

    def step(self, action: Action, *, friction: float) -> None:
        sim = self._env.sim
        external_force = sim.data.xfrc_applied[self._force_body, :3]
        original_force = external_force.copy()
        external_force += self._force
        self._force.fill(0.0)

        if friction != 1.0:
            sim.model.geom_friction[self._surface_geoms] = self._friction * friction
            # With equal priorities MuJoCo takes the larger contact coefficient.
            sim.model.geom_priority[self._surface_geoms] = sim.model.geom_priority.max() + 1
        try:
            self._raw, _, _, _ = self._env.step(np.clip(action, -1.0, 1.0))
            self.success = bool(self._env.check_success())
        finally:
            external_force[:] = original_force
            if friction != 1.0:
                sim.model.geom_friction[self._surface_geoms] = self._friction
                sim.model.geom_priority[self._surface_geoms] = self._priority

    def set_camera_effects(
        self,
        lighting: LightingShiftParams | None,
        shake: CameraShakeParams | None,
        step: int,
    ) -> None:
        state = (lighting, shake, step if shake else 0)
        if state == self._camera_state:
            return
        sim = self._env.sim
        model = sim.model
        diffuse = self._light_diffuse
        if lighting:
            gray = diffuse @ np.array([0.299, 0.587, 0.114])
            diffuse = gray[:, None] + lighting.saturation * (diffuse - gray[:, None])
            diffuse = np.clip(diffuse * lighting.brightness, 0.0, 1.0)
        model.light_diffuse[:] = diffuse

        positions, quaternions = self._cam_pos, self._cam_quat
        if shake:
            amplitude = shake.amplitude_px
            x = cos(step * 1.9)
            y = sin(step * 2.3 + 0.7)
            displacement = (x * amplitude * 0.0008, y * amplitude * 0.0008, 0.0)
            positions = positions + np.array([displacement, np.negative(displacement)])
            half_angles = np.array([x, -x]) * amplitude / (2 * IMAGE_WIDTH)
            c, s = np.cos(half_angles), np.sin(half_angles)
            qw, qx, qy, qz = quaternions.T
            quaternions = np.column_stack(
                (qw * c - qy * s, qx * c - qz * s, qy * c + qw * s, qz * c + qx * s)
            )
        model.cam_pos[self._camera_ids] = positions
        model.cam_quat[self._camera_ids] = quaternions
        sim.forward()
        self._camera_state = state

    def _render(self, camera: str) -> Image.Image:
        raw = self._env.sim.render(width=IMAGE_WIDTH, height=IMAGE_WIDTH, camera_name=camera)
        # Match the training data orientation used by LeRobot's LIBERO processor.
        return Image.fromarray(raw[::-1, ::-1].copy())

    def observe(self) -> Observation:
        raw = self._raw
        proprio = np.concatenate(
            (raw["robot0_eef_pos"], axis_angle(raw["robot0_eef_quat"]), raw["robot0_gripper_qpos"])
        )
        return Observation(
            rgb_main=self._render(CAMERAS[0]),
            rgb_wrist=self._render(CAMERAS[1]),
            proprio=tuple(proprio.tolist()),
            task_progress=float(self.success),
        )

from types import SimpleNamespace

import numpy as np
import pytest

from backend.chaos_sim import libero_env
from backend.chaos_sim.chaos import ChaosWrapper
from backend.chaos_sim.libero_env import LiberoEnvAdapter, LiberoUnavailable
from backend.shared.schemas import CLIENT_MESSAGE_ADAPTER


class Model:
    def __init__(self):
        self.cam_pos = np.zeros((2, 3))
        self.cam_quat = np.array([[1.0, 0.0, 0.0, 0.0]] * 2)
        self.light_diffuse = np.array([[0.8, 0.6, 0.4]])
        self.geom_names = [None, "table_visual", "bowl_collision"]
        self.body_names = ["world", "living_room_table_col", "bowl"]
        self.geom_bodyid = [1, 1, 2]
        self.ngeom = len(self.geom_names)
        self.geom_contype = self.geom_conaffinity = np.array([1, 0, 1])
        self.geom_friction = np.ones((3, 3))
        self.geom_priority = np.zeros(3, dtype=int)
        self.site_bodyid = [1]

    def camera_name2id(self, name):
        return libero_env.CAMERAS.index(name)

    def geom_id2name(self, geom):
        return self.geom_names[geom]

    def body_id2name(self, body):
        return self.body_names[body]

    def get_joint_qpos_addr(self, name):
        assert name == "bowl_joint"
        return (4, 11)

    def get_joint_qvel_addr(self, name):
        assert name == "bowl_joint"
        return (2, 8)


class Sim:
    def __init__(self):
        self.model = Model()
        self.data = SimpleNamespace(
            qpos=np.zeros(11), qvel=np.ones(8), xfrc_applied=np.zeros((2, 6))
        )
        self.forwards = 0

    def forward(self):
        self.forwards += 1

    def render(self, *, camera_name, width, height):
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        camera_id = self.model.camera_name2id(camera_name)
        rgb[:80, :100, :] = round(self.model.light_diffuse[0, 0] * (200 if camera_id else 250))
        return rgb


class FakeLibero:
    language_instruction = "Place the bowl on the plate."

    def __init__(self):
        self.sim = Sim()
        self.robots = [SimpleNamespace(eef_site_id=0)]
        self.obj_of_interest = ["fixture", "bowl"]
        self.env = SimpleNamespace(objects_dict={"bowl": SimpleNamespace(joints=["bowl_joint"])})
        self.fail_step = False
        self.closed = False

    def seed(self, seed):
        self.selected_seed = seed

    def reset(self):
        self.sim.data.qpos[:] = 0.0
        self.sim.data.qvel[:] = 1.0

    def set_init_state(self, state):
        self.sim.data.qpos[:] = state
        return {
            "robot0_eef_pos": np.array([0.2, 0.3, 0.4]),
            "robot0_eef_quat": np.array([0.0, 0.0, 1.0, 0.0]),
            "robot0_gripper_qpos": np.array([0.05, -0.05]),
        }

    def step(self, action):
        self.step_action = action
        self.step_wrench = self.sim.data.xfrc_applied.copy()
        self.step_friction = self.sim.model.geom_friction.copy()
        self.step_priority = self.sim.model.geom_priority.copy()
        if self.fail_step:
            raise RuntimeError("Failure inside step")
        return self.set_init_state(self.sim.data.qpos.copy()), 0, False, {}

    def check_success(self):
        return self.sim.data.qpos[4] > 1.0

    def close(self):
        self.closed = True


@pytest.fixture
def wrapped():
    native = FakeLibero()
    adapter = LiberoEnvAdapter(native, np.zeros(11), seed=42, settle_steps=0)
    return ChaosWrapper(adapter), adapter, native


def inject(wrapper, skill, **params):
    wrapper.inject(
        CLIENT_MESSAGE_ADAPTER.validate_python({"type": "chaos", "skill": skill, "params": params})
    )


def test_native_adapter_import_and_unavailable_message_without_optional_deps(monkeypatch):
    monkeypatch.setattr(libero_env, "find_spec", lambda _: None)
    assert not libero_env.libero_available()
    with pytest.raises(LiberoUnavailable, match="hf-libero"):
        LiberoEnvAdapter.from_task()


def test_native_observation_and_reset(wrapped):
    wrapper, adapter, native = wrapped
    assert native.selected_seed == 42
    observation = wrapper.observe().observation
    assert observation.proprio == pytest.approx((0.2, 0.3, 0.4, 0, 0, np.pi, 0.05, -0.05))
    assert observation.rgb_main.size == observation.rgb_wrist.size == (256, 256)
    assert observation.rgb_main.getpixel((255, 255)) == (200, 200, 200)
    assert observation.rgb_main.getpixel((0, 0)) == (0, 0, 0)
    assert observation.task_progress == 0
    native.sim.data.qpos[4] = 1.5
    assert wrapper.step((0.0,) * 7).observation.task_progress == 1
    wrapper.reset()
    assert wrapper.observe().observation.task_progress == 0
    adapter.close()
    assert native.closed


def test_nudge_updates_free_joint_translation_clears_velocity_and_forwards(wrapped):
    wrapper, adapter, native = wrapped
    inject(wrapper, "object_nudge", dx=0.2, dy=-0.1, dz=0.04)
    assert adapter.object_name == "bowl"
    assert adapter.object_position == pytest.approx((0.2, -0.1, 0.04))
    np.testing.assert_array_equal(native.sim.data.qvel, [1, 1, 0, 0, 0, 0, 0, 0])
    assert native.sim.forwards == 1
    wrapper.reset()
    assert adapter.object_position == (0.0, 0.0, 0.0)


@pytest.mark.parametrize("fail", [False, True])
def test_arm_wrench_and_table_contact_are_scoped_to_native_step(wrapped, fail):
    wrapper, adapter, native = wrapped
    inject(wrapper, "arm_kick", dx=0.10, dy=0.0, dz=-0.2)
    inject(wrapper, "friction_glitch", multiplier=0.2, duration_steps=1)
    native.fail_step = fail
    if fail:
        with pytest.raises(RuntimeError, match="Failure inside step"):
            wrapper.step((2.0,) * 7)
    else:
        assert wrapper.step((2.0,) * 7).active_chaos == ("arm_kick", "friction_glitch")
    np.testing.assert_array_equal(native.step_action, np.ones(7))
    np.testing.assert_array_equal(native.step_wrench[1], [7.5, 0, -15, 0, 0, 0])
    np.testing.assert_array_equal(native.step_friction[:, 0], [0.2, 1, 1])
    np.testing.assert_array_equal(native.step_priority, [1, 0, 0])
    assert not native.sim.data.xfrc_applied.any()
    np.testing.assert_array_equal(native.sim.model.geom_friction, np.ones((3, 3)))
    native.fail_step = False
    adapter.step((0.0,) * 7, friction=1)
    assert not native.step_wrench.any()


def test_friction_glitch_uses_floor_for_floor_manipulation_tasks():
    native = FakeLibero()
    native.sim.model.body_names[1] = "floor"
    wrapper = ChaosWrapper(LiberoEnvAdapter(native, np.zeros(11), seed=42, settle_steps=0))
    inject(wrapper, "friction_glitch", multiplier=0.1)
    wrapper.step((0.0,) * 7)
    np.testing.assert_array_equal(native.step_friction[:, 0], [0.1, 1, 1])


def test_native_light_and_camera_restore_at_expiry(wrapped):
    wrapper, _, native = wrapped
    model = native.sim.model
    initial_light = model.light_diffuse.copy()
    initial_quat = model.cam_quat.copy()
    inject(wrapper, "lighting_shift", brightness=0.5, saturation=1, duration_steps=1)
    inject(wrapper, "camera_shake", amplitude_px=20, duration_steps=1)
    masked = wrapper.step((0.0,) * 7)
    assert masked.observation.rgb_main.getpixel((255, 255)) == (100, 100, 100)
    np.testing.assert_allclose(model.light_diffuse, initial_light * 0.5)
    assert model.cam_pos.any()
    assert not np.allclose(model.cam_quat, initial_quat)
    np.testing.assert_allclose(np.linalg.norm(model.cam_quat, axis=1), np.ones(2))

    assert wrapper.observe().active_chaos == ()
    np.testing.assert_array_equal(model.light_diffuse, initial_light)
    np.testing.assert_array_equal(model.cam_pos, np.zeros((2, 3)))
    np.testing.assert_array_equal(model.cam_quat, initial_quat)


def test_visual_and_systems_effects_use_native_camera_outputs(wrapped):
    wrapper, _, _ = wrapped
    inject(wrapper, "occlusion_patch", x=0.75, y=0.75, width=0.25, height=0.25)
    inject(wrapper, "wrist_blackout")
    inject(wrapper, "latency_inject", latency_ms=125)
    observation = wrapper.observe().observation
    assert observation.rgb_main.getpixel((255, 255)) == (0, 0, 0)
    assert observation.rgb_main.getpixel((175, 190)) == (200, 200, 200)
    assert observation.rgb_wrist.getbbox() is None
    assert wrapper.latency_ms(10) == 125

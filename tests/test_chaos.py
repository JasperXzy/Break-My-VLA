import pytest
from PIL import ImageChops

from backend.chaos_sim.chaos import ChaosWrapper
from backend.chaos_sim.cpu_env import CpuPickEnv
from backend.shared.schemas import CLIENT_MESSAGE_ADAPTER

NEUTRAL_ACTION = (0.0,) * 7


@pytest.fixture
def wrapper():
    return ChaosWrapper(CpuPickEnv())


def inject(wrapper, skill, **params):
    wrapper.inject(
        CLIENT_MESSAGE_ADAPTER.validate_python({"type": "chaos", "skill": skill, "params": params})
    )


def test_cpu_scene_has_two_views_and_reset_restores_initial_state(wrapper):
    initial = wrapper.observe().observation
    assert initial.rgb_main.size == initial.rgb_wrist.size == (320, 240)
    assert initial.rgb_main.mode == initial.rgb_wrist.mode == "RGB"
    assert ImageChops.difference(initial.rgb_main, initial.rgb_wrist).getbbox() is not None
    inject(wrapper, "wrist_blackout")
    wrapper.step((1.0, 0, 0, 0, 0, 0, -1.0))
    wrapper.reset()
    scene = wrapper.observe()
    assert wrapper.step_count == 0
    assert scene.active_chaos == ()
    assert scene.observation.proprio == initial.proprio
    assert scene.observation.task_progress == initial.task_progress == 0
    assert ImageChops.difference(scene.observation.rgb_wrist, initial.rgb_wrist).getbbox() is None


def test_object_nudge_teleports_block_and_marks_one_step(wrapper):
    initial = wrapper.observe().observation
    inject(wrapper, "object_nudge", dx=0.12, dy=-0.05, dz=0.16)
    scene = wrapper.observe()
    assert scene.observation.proprio[:3] == initial.proprio[:3]
    assert scene.observation.proprio[3:6] == pytest.approx(
        [initial.proprio[3] + 0.12, initial.proprio[4] - 0.05, initial.proprio[5] + 0.16],
    )
    assert ImageChops.difference(scene.observation.rgb_main, initial.rgb_main).getbbox() is not None
    assert wrapper.step(NEUTRAL_ACTION).active_chaos == ("object_nudge",)
    assert wrapper.observe().active_chaos == ()


def test_arm_kick_pushes_hand_with_a_damped_impulse(wrapper):
    initial = wrapper.observe().observation.proprio
    inject(wrapper, "arm_kick", dx=0.12, dy=0.0, dz=0.0)
    first = wrapper.step(NEUTRAL_ACTION)
    second = wrapper.step(NEUTRAL_ACTION)
    assert first.active_chaos == ("arm_kick",)
    assert first.observation.proprio[0] - initial[0] == pytest.approx(0.12)
    follow_through = second.observation.proprio[0] - first.observation.proprio[0]
    assert follow_through == pytest.approx(0.12 * 0.45)
    assert first.observation.proprio[3:6] == initial[3:6]
    assert second.active_chaos == ()


def test_friction_glitch_increases_table_slip(wrapper):
    baseline = ChaosWrapper(CpuPickEnv())
    for env in (wrapper, baseline):
        inject(env, "object_nudge", dx=0.1, dy=0.0)
    inject(wrapper, "friction_glitch", multiplier=0, duration_steps=2)

    first = wrapper.step(NEUTRAL_ACTION)
    baseline.step(NEUTRAL_ACTION)
    second = wrapper.step(NEUTRAL_ACTION)
    reference = baseline.step(NEUTRAL_ACTION)
    assert first.active_chaos == ("object_nudge", "friction_glitch")
    assert second.active_chaos == ("friction_glitch",)
    assert second.observation.proprio[3] > reference.observation.proprio[3]
    assert wrapper.observe().active_chaos == ()


def test_lighting_shift_changes_both_views_and_expires(wrapper):
    baseline = wrapper.observe().observation
    inject(wrapper, "lighting_shift", brightness=0, duration_steps=1)
    dark = wrapper.step(NEUTRAL_ACTION).observation
    assert dark.rgb_main.getbbox() is None
    assert dark.rgb_wrist.getbbox() is None
    restored = wrapper.observe().observation
    assert ImageChops.difference(restored.rgb_main, baseline.rgb_main).getbbox() is None
    assert ImageChops.difference(restored.rgb_wrist, baseline.rgb_wrist).getbbox() is None


def test_camera_shake_changes_both_views_across_steps(wrapper):
    initial = wrapper.observe().observation
    inject(wrapper, "camera_shake", amplitude_px=18)
    shaken = wrapper.observe().observation
    next_frame = wrapper.step(NEUTRAL_ACTION).observation
    assert ImageChops.difference(initial.rgb_main, shaken.rgb_main).getbbox() is not None
    assert ImageChops.difference(initial.rgb_wrist, shaken.rgb_wrist).getbbox() is not None
    assert ImageChops.difference(shaken.rgb_main, next_frame.rgb_main).getbbox() is not None
    assert next_frame.proprio == initial.proprio


def test_occlusion_patch_masks_only_the_main_view(wrapper):
    initial = wrapper.observe().observation
    inject(wrapper, "occlusion_patch", x=0.25, y=0.25, width=0.5, height=0.5)
    scene = wrapper.observe().observation
    assert scene.rgb_main.crop((80, 60, 240, 180)).getbbox() is None
    assert scene.rgb_main.getpixel((79, 60)) == initial.rgb_main.getpixel((79, 60))
    assert ImageChops.difference(initial.rgb_wrist, scene.rgb_wrist).getbbox() is None


def test_latency_injection_temporarily_overrides_configured_latency(wrapper):
    assert wrapper.latency_ms(40) == 40
    inject(wrapper, "latency_inject", latency_ms=185, duration_steps=2)
    assert wrapper.latency_ms(40) == 185
    wrapper.observe()
    wrapper.step(NEUTRAL_ACTION)
    assert wrapper.latency_ms(40) == 185
    assert wrapper.step(NEUTRAL_ACTION).active_chaos == ("latency_inject",)
    assert wrapper.latency_ms(40) == 40


def test_wrist_blackout_masks_only_the_wrist_view(wrapper):
    initial = wrapper.observe().observation
    inject(wrapper, "wrist_blackout", duration_steps=1)
    scene = wrapper.step(NEUTRAL_ACTION)
    assert scene.active_chaos == ("wrist_blackout",)
    assert scene.observation.rgb_wrist.getbbox() is None
    assert ImageChops.difference(initial.rgb_main, scene.observation.rgb_main).getbbox() is None
    assert wrapper.observe().observation.rgb_wrist.getbbox() is not None


def test_reinjecting_timed_skill_replaces_its_duration_and_parameters(wrapper):
    inject(wrapper, "latency_inject", latency_ms=30, duration_steps=10)
    wrapper.step(NEUTRAL_ACTION)
    inject(wrapper, "latency_inject", latency_ms=80, duration_steps=1)
    assert wrapper.latency_ms(0) == 80
    wrapper.step(NEUTRAL_ACTION)
    assert wrapper.latency_ms(0) == 0

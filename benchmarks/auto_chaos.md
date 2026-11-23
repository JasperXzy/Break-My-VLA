# M3 Auto-Chaos

Run a CPU smoke comparison from the repository root.

```bash
uv run python -m benchmarks.auto_chaos --mock \
  --models smolvla,pi05 --precisions bf16,fp8,int8 --tasks 0 --levels 0,5 \
  --output /tmp/m3-auto-chaos
```

Mock uses the existing pick-and-lift environment and the HTTP mock policy. All
task IDs exercise the same tabletop scene in this mode. Policy and injected
delays are recorded as simulated durations. Image effects run through the
simulator, and the mock policy steers using proprioception.

For a GPU run, use [the M3 notebook](../notebooks/break_my_vla_m3_auto_chaos.ipynb)
to set up both LeRobot runtimes. With a prepared runtime, call
`python -m benchmarks.auto_chaos` without `--mock`. The runner starts and stops
one local policy process at a time. Use `--policy-python pi05=/path/to/python`
when Pi0.5 has a separate environment. To use existing policy HTTP servers, set
`POLICY_SERVERS` to a comma-separated list of `model=url` entries. Unsupported
requested precisions appear in `metrics.json` as skipped, such as FP8 on an
A100. Pass `--pi05-nvfp4-backend module:factory --precisions bf16,nvfp4` to
evaluate a managed Pi0.5 server's optional TensorRT backend on SM 12.0.

`--tasks` accepts numeric task IDs within `--suite` or entries such as
`libero_object:0,libero_spatial:1`. The default suite is `libero_object`.
`--levels` accepts `0-5`, individual levels, or ranges such as `0,2-4`.
Each model, precision, and task has a level 0 clean control. Each selected
positive level runs each stress skill independently from the same initial
state and seed.

| Axis | Rollout settings |
| --- | --- |
| Clean | Nominal precision and 10 denoise steps. |
| Physical | `object_nudge`, `arm_kick`, and `friction_glitch` scale from one fifth of their level 5 strengths to full strength. |
| Visual | `lighting_shift`, `camera_shake`, and `occlusion_patch` scale with level. |
| Systems | `latency_inject` adds 150 ms per affected policy call. `wrist_blackout` blanks the wrist view. Each lasts six simulator steps per level. |
| Quantization | Levels 1 through 5 use 8, 6, 4, 2, and 1 denoise steps, respectively. The runner selects a lower bit width among the requested, available precisions when possible. |

The runner injects physical, visual, and systems effects at `--inject-step`
(default 0). Set `--initial-state-id`, `--seed`, `--max-steps`, and
`--chunk-actions` to control the episode and action replanning cadence.
The default LIBERO rollout runs up to 280 steps and executes up to ten
actions from each policy response. When the task lacks a movable object or
finishes before the injection, the affected stress rollout is marked
`not_exercised`.

- `metrics.json` records policy capabilities, episode settings, outcomes,
  applied commands, and per-call policy latency, injected latency, wall time,
  and peak allocated policy VRAM. The runner writes it after each model.
- `gallery.json` uses Gallery schema 1.0. Radar values are success ratios
  converted to percentages by the Gallery. The overall score averages measured
  axes. Clean policy calls supply the leaderboard latency and calls per second.
- `replays/` contains baseline and selected stress GIFs from the first task.
  The Gallery pairs the highest selected level's representative stress runs and
  the clean precision runs. Paths in the JSON are relative to the artifact root.

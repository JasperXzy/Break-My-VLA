# M4 Auto-Chaos and TensorRT export

Run a CPU smoke comparison from the repository root.

```bash
uv run python -m benchmarks.auto_chaos --mock \
  --models smolvla,pi05,groot --precisions bf16,fp8,int8,nvfp4 --tasks 0 --levels 0-5 \
  --output /tmp/m4-auto-chaos
uv run python -m scripts.export_tensorrt --mock --model groot --output /tmp/m4-tensorrt
```

Mock uses the existing pick-and-lift environment and the HTTP mock policy. All
task IDs exercise the same tabletop scene in this mode. Policy and injected
delays are recorded as simulated durations. Image effects run through the
simulator, and the mock policy steers using proprioception. Mock NVFP4 entries
use the simulated TensorRT label. The mock exporter writes JSON stage records
and a `profile.json` with simulated latency and action alignment.

## GPU runtime and export

The M4 notebook prepares Python 3.11 policy environments, captures calibration
and held-out observations from LIBERO, builds the available TensorRT tiers and
runs the matrix. Colab stores artifacts on Drive. NVFP4 requires SM 12.0
(G4 or RTX 5090). L4 runs BF16, FP8 and INT8 and records NVFP4 as skipped.
[M4 notebook](../notebooks/break_my_vla_m4_groot_trt_nvfp4.ipynb) ·
[NVIDIA CUDA GPU capabilities](https://developer.nvidia.com/cuda-gpus)

In a prepared policy environment, export one model with the command below.
Omitting the two JSONL options uses deterministic synthetic observations.
Each JSONL line uses the existing `/act` request fields with eight LIBERO
proprioception values. `--warmup`, `--iterations` and `--workspace-mib` control
the build and CUDA Event profiling.

```bash
python -m scripts.export_tensorrt --model groot --groot-suite libero_object \
  --precisions bf16,fp8,nvfp4 --calibration calibration.jsonl \
  --observations heldout.jsonl --output /tmp/m4-tensorrt
```

GR00T exports its diffusion transformer (DiT). Its NVFP4 recipe uses NVFP4
feed-forward projections and FP8 attention projections. SmolVLA and Pi0.5
export their denoising output projection. The rest of each policy uses
PyTorch. The exporter writes ONNX, serialized engines and a manifest under
`<output>/<model>/<precision>/`. Rebuild the plans on the target GPU with its
TensorRT runtime. `profile.json` reports CUDA Event latency for the exported
component, wall-clock policy latency, and end-to-end action alignment against
PyTorch BF16 on identical held-out observations and denoise settings. L1 is
mean absolute error across the entire chunk. Maximum absolute difference is
the largest error across all held-out chunks.
[Isaac GR00T LIBERO checkpoint](https://huggingface.co/nvidia/GR00T-N1.7-LIBERO) ·
[ModelOpt](https://github.com/NVIDIA/Model-Optimizer/tree/0.43.0) ·
[TensorRT Python runtime](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/python-api-docs.html)

With a prepared simulator and policy environments, run the complete matrix.

```bash
python -m benchmarks.auto_chaos --trt-root /tmp/m4-tensorrt \
  --policy-python pi05=/path/to/pi05/bin/python \
  --policy-python groot=/path/to/groot/bin/python \
  --tasks libero_object:0 --levels 0-5 --output /tmp/m4-auto-chaos
```

The runner starts and stops one local policy process at a time. To use existing
HTTP servers, set `POLICY_SERVERS` to comma-separated `model=url` entries.
Managed GR00T uses the checkpoint matching a single requested LIBERO suite.
Use `--groot-suite` to choose the GR00T checkpoint for a mixed-suite run.
Managed policies use available PyTorch tiers when TensorRT plans are absent or
incompatible, and `metrics.json` records unavailable requested precisions as
skipped. Pi0.5 also accepts the M3 `--pi05-nvfp4-backend module:factory` option.

## Rollouts and artifacts

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
  and peak PyTorch-allocated policy VRAM. The runner writes it after each model.
- `gallery.json` uses Gallery schema 1.0. Radar values are success ratios
  converted to percentages by the Gallery. The overall score averages measured
  axes. Clean policy calls supply the leaderboard latency and calls per second.
- `replays/` contains baseline and selected stress GIFs from the first task.
  The Gallery pairs the highest selected level's representative stress runs and
  the clean precision runs. Paths in the JSON are relative to the artifact root.

Run records and Gallery entries include optional `backend` and `latency.p99_ms`
when runtime metadata and latency samples are available. Each task's clean
alignment probe evaluates the same observation with the same prompt and ten
denoise steps at every tier against the server's BF16 tier. The Gallery averages
the resulting `action_mse_vs_bf16`, `action_l1_vs_bf16` and
`action_cos_vs_bf16` across tasks. Earlier policy and Gallery JSON contracts
remain compatible.

# Break-My-VLA M1 protocol

Run the two CPU services in separate terminals from the repository root.

```bash
uv run python -m backend.policy_servers.mock_server
uv run python -m backend.chaos_sim.server
```

The policy server listens at `http://127.0.0.1:8001` and the simulator at
`http://127.0.0.1:8000`. Set `POLICY_SERVER_URL` on the simulator to select another
HTTP policy server. The simulator allows cross-origin HTTP requests.

The frozen Pydantic wire models and discriminated WebSocket unions live in
[`schemas.py`](schemas.py). The image placeholders below stand for standard
Base64-encoded JPEG bytes without the data URI prefix. The CPU simulator emits
320×240 RGB images from both cameras.

## Policy HTTP API

### `GET /health`

```json
{"model":"mock","precisions":["bf16","fp8","int8","nvfp4"],"device":"cpu"}
```

### `POST /act`

Send JSON with all six request fields.

```json
{
  "rgb_main_b64": "<main JPEG>",
  "rgb_wrist_b64": "<wrist JPEG>",
  "proprio": [-0.43, 0.23, 0.43, 0.22, -0.18, 0.045, 0.0],
  "prompt": "Pick up the orange block.",
  "precision": "bf16",
  "denoise_steps": 10
}
```

```json
{
  "action_chunk": [
    [0.90, -0.90, -0.82, 0.01, 0.00, 0.00, -0.90],
    [0.84, -0.84, -0.77, 0.00, 0.01, 0.00, -0.90]
  ],
  "infer_ms": 52.4,
  "vram_mb": 0.0
}
```

Precision values are lowercase. `denoise_steps` is an integer from 1 through 10.
`proprio` is a nonempty numeric array. The CPU demo supplies hand XYZ, block XYZ,
and a numeric grasp flag in that order. The mock uses those values to steer the
gripper and generates eight actions with precision- and step-dependent jitter.
It accepts both camera images and the prompt for HTTP parity with real policies.

Each action has seven finite values in translation XYZ, rotation XYZ, gripper
order. The CPU simulator uses translation and the gripper sign. Positive closes
the gripper and zero or negative opens it. The simulator executes the first
action per step and streams the full chunk. The mock measures its simulated
inference time in milliseconds and reports zero VRAM usage. Invalid requests
return HTTP 422.

## WebSocket `/ws/session`

Connect to `ws://127.0.0.1:8000/ws/session` and send JSON as text messages.
Each connection owns its simulation, configuration, effects, and recording.

### Server frame

```json
{
  "type": "frame",
  "step": 5,
  "fps": 18.9,
  "infer_ms": 52.4,
  "precision": "bf16",
  "denoise_steps": 10,
  "rgb_main_b64": "<main JPEG>",
  "rgb_wrist_b64": "<wrist JPEG>",
  "action_chunk": [[0.33, -0.18, -0.45, 0.01, 0.00, 0.00, -0.90]],
  "active_chaos": ["camera_shake"],
  "task_progress": 0.31,
  "status": "running"
}
```

Step 0 is the initial or reset frame. The simulator targets up to 20 steps per
second. `fps` reflects the measured step rate and `infer_ms` is the policy's
reported inference time plus applied injected latency. The chunk is empty
before the first inference and on policy errors. `task_progress` is in
`[0, 1]` and reaches 1 when the block reaches the lift height. The session
keeps stepping after `success`, so new perturbations can affect the grasp.
The other statuses are `running`, `paused`, and `policy_error`. A failed
policy request produces a `policy_error` frame and retries at the current step.

### Client config and controls

```json
{"type":"config","precision":"fp8","denoise_steps":3,"latency_ms":40,"prompt":"Lift the block."}
```

Each config field is optional. Omitted or null fields retain their current
values. Defaults are `bf16`, 10 denoise steps, 0 ms added latency, and
`Pick up the orange block.` Added latency accepts 0 through 300 ms. Precision
and denoise settings form the fourth chaos category.

```json
{"type":"control","action":"pause"}
```

- `pause` sends a paused frame and suspends stepping. Config changes and skill
  injections during pause send a fresh preview at the current step.
- `resume` restarts stepping.
- `reset` starts a new episode at step 0 and clears effects and history.
  Playback mode and config carry over to the new episode.
- `export_gif` sends the current connection's replay as a `gif` message.

### Client chaos

```json
{"type":"chaos","skill":"camera_shake","params":{"amplitude_px":12,"duration_steps":30}}
```

Use `"params":{}` for defaults. Timed skills accept `duration_steps` from 1
through 600 and default to 30 successful simulation steps. Reapplying a timed
skill replaces its parameters and restarts its duration. Each effect appears
in `active_chaos` for its configured duration. The instant physical skills
mark the next step and can leave the block or gripper in motion.

| Category | Skill | Parameters and effect |
| --- | --- | --- |
| Physical | `object_nudge` | `dx`, `dy`, `dz` default to 0.18, -0.12, 0.0. Each is in [-0.5, 0.5] workspace units. Moves the block immediately and marks the next step. |
| Physical | `arm_kick` | `dx`, `dy`, `dz` default to -0.22, 0.10, 0.12. Each is in [-0.5, 0.5] workspace units per step. Applies a damped gripper impulse starting on the next step. |
| Physical | `friction_glitch` | `multiplier` in [0, 3], default 0.10. The baseline is 1. Lower values increase table slip. |
| Visual | `lighting_shift` | `brightness` and `saturation` in [0, 2.5], defaults 0.45 and 0.65. Applies to both views. |
| Visual | `camera_shake` | `amplitude_px` in [0, 80], default 14. Shifts both views with per-step jitter. |
| Visual | `occlusion_patch` | `x`, `y`, `width`, `height` default to 0.36, 0.30, 0.29, 0.25. The top-left coordinates are in [0, 1) and the dimensions in (0, 1]. A black rectangle covers that area of the main view and clips at its edges. |
| Systems | `latency_inject` | `latency_ms` in [0, 300], default 160. Temporarily overrides the configured added latency before each policy call. |
| Systems | `wrist_blackout` | `duration_steps` controls how long the wrist view is black. |

### Server GIF and error messages

```json
{"type":"gif","gif_b64":"<GIF bytes in Base64>"}
```

```json
{"type":"error","code":"invalid_message","message":"config.denoise_steps: Input should be greater than or equal to 1"}
```

An invalid message produces an error and leaves the connection open.
The `gif_unavailable` error code denotes an empty recording.

## HTTP GIF export

`POST /api/export_gif` takes no body and returns an `image/gif` attachment
named `break-my-vla.gif`. It exports the session that most recently recorded a
frame, even after that session closes. Before any frame is recorded it returns
HTTP 404. Both export paths use up to the last 100 main-view frames, including
the initial scene and successful simulation steps. Replays play at 20 fps.

## Environment hook

The simulator accepts an `env_factory` through
`backend.chaos_sim.server.create_app`. Each factory call supplies an `Env` with
reset, observation, stepping, object nudge, and arm impulse methods. M2 can
implement `LiberoEnvAdapter` against that interface and keep the HTTP and
WebSocket contracts.

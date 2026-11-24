# Break-My-VLA Frontend & GitHub Pages Showcase Guide

This document covers the **Break-My-VLA** M4 **TensorRT & Quantization Frontier Arena** frontend (`frontend/`), full-stack Live WebSocket verification, multi-model hot-switching, and the static **Multi-VLA Head-to-Head Arena** (`#gallery`) designed for zero-GPU GitHub Pages browsing.

## 1. Two Interactive Views

| View | URL Hash | Purpose |
| --- | --- | --- |
| **Live Arcade (`LIVE ARCADE`)** | `#arcade` (default) | Real-time dual-camera console (`MAIN VIEW` + `WRIST CAM` with TV static snow during `wrist_blackout`), top-bar **Multi-VLA Model Switcher** (hot-swapping models via WS `{ type: "config", model }` with live parameter count & latency baseline chips), action chunk telemetry, latency spark bars, 4-category God-Mode Chaos Console (hotkeys `1`–`8`, `Space`, `R`), and live event log. Connects to `ws://localhost:8000/ws/session` with automatic reconnect on transient drops and Local Demo fallback when offline. |
| **TensorRT & Quantization Frontier Arena (`MULTI-VLA ARENA`)** | `#gallery` or `#gallery?dataset=<file>&a=<id>&b=<id>` | Static analytics and failure-replay arena powered by `benchmarks/results/*.json` (prioritizing verified real GPU suites first with a **`REAL GPU VERIFIED`** badge). Features a **Head-to-Head 1v1 Comparison Card** (5-axis Δ, latency speedup ratio, VRAM savings, M4 **Action-Chunk Precision Drift** & **P99 Tail Latency** panels, URL hash deep-link **`COPY SHARE LINK`**, and one-click Copy/Export Markdown tables), a **Quantization Pareto Frontier** scatter plot (`33ms / 30Hz` & `50ms / 20Hz` real-time redlines), a **Chaos Survival Matrix** heatmap, a 5-dimensional stress resilience radar chart, sortable latency/throughput bars, an interactive leaderboard, and a synchronized dual-screen GIF comparison player. |

## 2. M4 TensorRT & Quantization Frontier Features

### Quantization Pareto Frontier (`ParetoFrontierChart.tsx`)
- **Axes**: X-axis plots mean inference latency (`ms`, lower is faster) with two real-time robot control redlines at **`33ms / 30Hz`** and **`50ms / 20Hz`**; Y-axis plots composite resilience score (`0–100`).
- **Visual Encoding**:
  - **Color by Precision**: `BF16` (emerald), `FP8` (cyan), `INT8` (amber), `NVFP4` (fuchsia).
  - **Shape by Inference Backend**: **PyTorch** (`●` circle) vs **TensorRT** (`◆` diamond).
  - **Auto-Computed Frontier**: Non-dominated configurations minimizing latency and maximizing resilience are automatically computed via `computeParetoFrontier()` and connected with a highlighted Pareto curve.
- **Interactive Inspection & Pairing**: Hovering or focusing any point displays its backend, latency (`Mean / P95 / P99`), FPS, VRAM, and BF16 action-chunk alignment (`Cosine` & `MSE`). Clicking any scatter point (or frontier quick-pick row) immediately pairs it into the 1v1 Head-to-Head Arena.

### Action-Chunk Precision Drift, P99 Tail Latency & Deep-Link Sharing (`HeadToHeadCard.tsx`)
- **Action-Chunk Precision Drift**: When compared entries include `alignment` (`action_cos_vs_bf16`, `action_mse_vs_bf16`, `action_l1_vs_bf16`), the 1v1 card renders side-by-side drift bars with fidelity tier badges (`BF16 REF`, `ALIGNED`, `MILD DRIFT`, `HIGH DRIFT`) and Δ Cosine winner attribution.
- **P99 Tail Latency Comparison**: When entries include `latency.p99_ms`, the 1v1 card compares P99 tail latency against the `33ms (30Hz)` and `50ms (20Hz)` control ceilings alongside P99 speedup (`×`).
- **URL Hash Deep-Linking (`#gallery?dataset=...&a=...&b=...`)**: Visitors can click **`COPY SHARE LINK`** to copy a shareable deep-link URL to the exact dataset and 1v1 contender pairing. Loading a deep-link hash hydrates the Arena view, active dataset, and both contenders automatically.
- **Enhanced Markdown Export**: **`COPY MARKDOWN`** and **`EXPORT .MD`** output GitHub-Flavored Markdown including a contender spec summary table (`Backend`, `Mean / P95 / P99`, `Action Cosine (vs BF16)`, `Action MSE (vs BF16)`) and full 5-axis + latency/VRAM delta table.

## 3. Multi-Model Hot-Switching (`/health` & WS `config`)

### Live Backend Discovery (`GET /health`)
Upon WebSocket connection, the frontend queries `GET /health` derived from the active WebSocket endpoint and supports both:
- **Multi-model responses**: `{ "active_model": "pi0.5-libero", "models": ["smolvla-450m", "pi0.5-libero", ...] }` (or rich objects with `id`, `label`, `params_b`, `baseline_latency_ms`, `precisions`).
- **Legacy single-model responses** (backward compatible): `{ "model": "lerobot/smolvla_libero", "precisions": ["bf16", "int8"], "device": "cuda:0", "env": "LiberoEnvAdapter" }`.

Selecting a model in the top-bar dropdown dispatches `{ "type": "config", "model": "<model_id>" }` over WebSocket and updates the model label, parameter count pill, and baseline latency indicator in real time.

### Local Demo Mode Presets (`?mode=demo` or Offline Fallback)
In Demo Mode, the top-bar Model Switcher provides 4 differentiated VLA presets with distinct latency baselines, tracking gains, and chaos sensitivities:
1. **SmolVLA 450M** (`450M` · `24ms` BF16 baseline) — Ultra-fast compact edge VLA; higher sensitivity to physical nudge & 4-bit noise.
2. **Pi0.5** (`3.0B` · `58ms` BF16 baseline) — Balanced flow-matching generalist with strong wrist-blackout recovery.
3. **Isaac GR00T** (`2.2B` · `42ms` BF16 baseline) — High-bandwidth visuomotor control with strong physical impulse damping.
4. **OpenVLA-OFT** (`7.0B` · `74ms` BF16 baseline) — Heavyweight semantic policy with high visual chaos resilience.

## 4. Local Full-Stack Startup & Live E2E Verification

Run the three services from the repository root:

```bash
# Terminal 1: CPU Mock Policy Server (port 8001)
uv run python -m backend.policy_servers.mock_server

# Terminal 2: Chaos Sim WebSocket Server (port 8000)
uv run python -m backend.chaos_sim.server

# Terminal 3: Vite Frontend Dev Server (port 5173)
cd frontend && npm run dev
```

Run automated Live WebSocket + Vite asset verification:

```bash
cd frontend && npm run verify:live
```

The verifier validates:
- `GET http://127.0.0.1:8001/health` and Vite static JSON/GIF assets
- Live `/ws/session` frame streaming (`rgb_main_b64`, `rgb_wrist_b64`, `action_chunk`, `fps`, `infer_ms`)
- All 8 chaos skills (`object_nudge`, `arm_kick`, `friction_glitch`, `lighting_shift`, `camera_shake`, `occlusion_patch`, `latency_inject`, `wrist_blackout`)
- Runtime `precision` hot-swap (`bf16` / `fp8` / `int8` / `nvfp4`) and `denoise_steps` (`1–10`)
- `latency_ms` & `prompt` updates, `pause` / `resume` / `reset`, WebSocket & HTTP `export_gif`
- Disconnect and reconnect recovery with zero protocol errors

## 5. Automated Frontend Checks

```bash
cd frontend
npm run lint     # ESLint across src/ and scripts/
npm run test     # Vitest unit, contract, and React component tests (jsdom 26)
npm run build    # TypeScript strict check (tsc -b) + Vite production bundle
npm run check    # Runs lint + test + build together
```

## 6. Static JSON Schema (M4 Backward-Compatible) & Real-GPU Prioritization

At build and dev time, `vite.benchmark-plugin.ts` automatically discovers JSON files in:
1. `<repo>/benchmarks/results/*.json` (highest priority; automatically copied to `frontend/dist/benchmarks/results/` on build)
2. `frontend/public/benchmarks/results/*.json` (built-in static GitHub Pages showcase datasets, including the M4 TensorRT/GR00T/NVFP4 showcase `auto_chaos_m4_showcase.json`)

Each entry in `entries[]` supports the backward-compatible M4 schema extensions:

```json
{
  "id": "groot-nvfp4-trt",
  "model": "GR00T-N1.5-2.2B",
  "family": "NVIDIA Isaac GR00T DiT",
  "backend": "tensorrt",
  "precision": "nvfp4",
  "denoise_steps": 4,
  "params_b": 2.2,
  "vram_gb": 2.8,
  "overall_score": 77.6,
  "latency": { "mean_ms": 12.6, "p50_ms": 12.3, "p95_ms": 14.4, "p99_ms": 15.5, "fps": 79.4 },
  "radar_scores": { "clean": 91, "physical": 77, "visual": 75, "systems": 78, "quantization": 67 },
  "alignment": {
    "action_mse_vs_bf16": 0.00342,
    "action_l1_vs_bf16": 0.0412,
    "action_cos_vs_bf16": 0.9846
  }
}
```

All M4 fields (`backend`, `latency.p99_ms`, and `alignment`) are optional, so M1/M2/M3 benchmark JSONs load cleanly with zero migration. Datasets remain sorted by provenance so **real GPU measured suites** (`01_smolvla_libero_l4_real.json`, tagged **`REAL GPU VERIFIED`**) are always prioritized first ahead of **`SHOWCASE / REFERENCE`** and **`CPU MOCK BASELINE`** suites.

## 7. GitHub Pages Static Deployment

`frontend/vite.config.ts` sets `base: './'` so `frontend/dist/` deploys cleanly under either a custom domain or a repository subpath (`https://<user>.github.io/Break-My-VLA/`). On `*.github.io` hosts (or with `?mode=demo`), the Arcade view launches immediately in Local Demo Mode without waiting for a local WebSocket server, and visitors can link directly to `#gallery` or deep-link to a specific 1v1 comparison via `#gallery?dataset=...&a=...&b=...`.

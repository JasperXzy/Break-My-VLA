# Break-My-VLA Frontend & GitHub Pages Showcase Guide

This document covers the **Break-My-VLA** M3 **Multi-VLA Arena** frontend (`frontend/`), full-stack Live WebSocket verification, multi-model hot-switching, and the static **Multi-VLA Head-to-Head Arena** (`#gallery`) designed for zero-GPU GitHub Pages browsing.

## 1. Two Interactive Views

| View | URL Hash | Purpose |
| --- | --- | --- |
| **Live Arcade (`LIVE ARCADE`)** | `#arcade` (default) | Real-time dual-camera console (`MAIN VIEW` + `WRIST CAM` with TV static snow during `wrist_blackout`), top-bar **Multi-VLA Model Switcher** (hot-swapping models via WS `{ type: "config", model }` with live parameter count & latency baseline chips), action chunk telemetry, latency spark bars, 4-category God-Mode Chaos Console (hotkeys `1`–`8`, `Space`, `R`), and live event log. Connects to `ws://localhost:8000/ws/session` with automatic reconnect on transient drops and Local Demo fallback when offline. |
| **Multi-VLA Head-to-Head Arena (`MULTI-VLA ARENA`)** | `#gallery` | Static analytics and failure-replay arena powered by `benchmarks/results/*.json` (prioritizing verified real GPU suites first with a **`REAL GPU VERIFIED`** badge). Features a **Head-to-Head 1v1 Comparison Card** (5-axis Δ, latency speedup ratio, VRAM savings, winner verdict, and one-click Copy/Export Markdown table), a **Chaos Survival Matrix** heatmap, a 5-dimensional stress resilience radar chart, sortable latency/throughput comparison bars (`50ms / 20Hz` reference line), an interactive quantization leaderboard, and a synchronized dual-screen GIF comparison player. |

## 2. Multi-Model Hot-Switching (`/health` & WS `config`)

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

## 3. Local Full-Stack Startup & Live E2E Verification

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

## 4. Automated Frontend Checks

```bash
cd frontend
npm run lint     # ESLint across src/ and scripts/
npm run test     # Vitest unit, contract, and React component tests (jsdom 26)
npm run build    # TypeScript strict check (tsc -b) + Vite production bundle
npm run check    # Runs lint + test + build together
```

## 5. Static JSON Schema & Real-GPU Prioritization (`benchmarks/results/*.json`)

At build and dev time, `vite.benchmark-plugin.ts` automatically discovers JSON files in:
1. `<repo>/benchmarks/results/*.json` (highest priority; automatically copied to `frontend/dist/benchmarks/results/` on build)
2. `frontend/public/benchmarks/results/*.json` (built-in static GitHub Pages showcase datasets)

Datasets are automatically classified and sorted by provenance so **real GPU measured suites** (e.g., `01_smolvla_libero_l4_real.json`, tagged **`REAL GPU VERIFIED`**) are selected by default ahead of **`SHOWCASE / REFERENCE`** and **`CPU MOCK BASELINE`** suites. Visitors can also click **LOAD JSON** inside the Arena to inspect any local benchmark result file directly in the browser, compare any 2 `model × precision` rows in the **Head-to-Head 1v1 Card**, inspect weakest-link failures in the **Chaos Survival Matrix**, and click **COPY MARKDOWN** or **EXPORT .MD** to embed comparison tables directly into READMEs or blog posts.

## 6. GitHub Pages Static Deployment

`frontend/vite.config.ts` sets `base: './'` so `frontend/dist/` deploys cleanly under either a custom domain or a repository subpath (`https://<user>.github.io/Break-My-VLA/`). On `*.github.io` hosts (or with `?mode=demo`), the Arcade view launches immediately in Local Demo Mode without waiting for a local WebSocket server, and visitors can link directly to `#gallery` for the Multi-VLA Head-to-Head Arena.

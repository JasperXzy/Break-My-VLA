# Break-My-VLA Frontend & GitHub Pages Showcase Guide

This document covers the **Break-My-VLA** frontend (`frontend/`), full-stack Live WebSocket verification, and the static **Benchmark & Replay Gallery** (`#gallery`) designed for zero-GPU GitHub Pages browsing.

## 1. Two Interactive Views

| View | URL Hash | Purpose |
| --- | --- | --- |
| **Live Arcade (`LIVE ARCADE`)** | `#arcade` (default) | Real-time dual-camera console (`MAIN VIEW` + `WRIST CAM` with TV static snow during `wrist_blackout`), action chunk telemetry, latency spark bars, 4-category God-Mode Chaos Console (hotkeys `1`–`8`, `Space`, `R`), and live event log. Connects to `ws://localhost:8000/ws/session` with automatic reconnect on transient drops and Local Demo fallback when offline. |
| **Benchmark & Replay Gallery (`BENCHMARK & REPLAY`)** | `#gallery` | Static analytics and failure-replay gallery powered by `benchmarks/results/*.json` (and bundled static showcases). Features a 5-dimensional stress resilience radar chart, sortable latency/throughput comparison bars with a `50ms / 20Hz` reference line, an interactive quantization leaderboard (`BF16` / `FP8` / `INT8` / `NVFP4`), and a synchronized dual-screen GIF comparison player. |

## 2. Local Full-Stack Startup & Live E2E Verification

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

## 3. Automated Frontend Checks

```bash
cd frontend
npm run lint     # ESLint across src/ and scripts/
npm run test     # Vitest unit, contract, and React component tests
npm run build    # TypeScript strict check (tsc -b) + Vite production bundle
npm run check    # Runs lint + test + build together
```

## 4. Static JSON Schema (`benchmarks/results/*.json`)

At build and dev time, `vite.benchmark-plugin.ts` automatically discovers JSON files in:
1. `<repo>/benchmarks/results/*.json` (highest priority; automatically copied to `frontend/dist/benchmarks/results/` on build)
2. `frontend/public/benchmarks/results/*.json` (built-in static GitHub Pages showcase datasets)

Visitors can also click **LOAD JSON** inside the Gallery to inspect any local benchmark result file directly in the browser.

Canonical JSON format:

```json
{
  "schema_version": "1.0",
  "suite_name": "Auto-Chaos Level 0-5 Quantization Matrix",
  "generated_at": "2026-11-23T09:30:00Z",
  "hardware": "NVIDIA Blackwell G4 96GB",
  "notes": "Optional summary notes",
  "entries": [
    {
      "id": "pi05-nvfp4",
      "model": "pi0.5-libero",
      "family": "OpenPI Flow-Matching",
      "precision": "nvfp4",
      "denoise_steps": 3,
      "params_b": 3.0,
      "vram_gb": 3.5,
      "overall_score": 68.8,
      "latency": { "mean_ms": 17.4, "p50_ms": 16.9, "p95_ms": 21.3, "fps": 57.5 },
      "radar_scores": {
        "clean": 86,
        "physical": 63,
        "visual": 66,
        "systems": 72,
        "quantization": 57
      }
    }
  ],
  "replays": [
    {
      "id": "nudge-bf16-vs-nvfp4",
      "title": "Physical Nudge Collapse · BF16 vs NVFP4",
      "chaos_tag": "object_nudge",
      "prompt": "pick up the crimson block and place it in the cyan bowl",
      "baseline_label": "pi0.5-libero · BF16 (10 steps)",
      "baseline_precision": "bf16",
      "baseline_latency_ms": 58.2,
      "baseline_gif": "./replays/bf16_clean_pass.gif",
      "baseline_outcome": "pass",
      "degraded_label": "pi0.5-libero · NVFP4 (3 steps)",
      "degraded_precision": "nvfp4",
      "degraded_latency_ms": 17.4,
      "degraded_gif": "./replays/nvfp4_nudge_fail.gif",
      "degraded_outcome": "fail",
      "failure_note": "Summary of failure mode under quantization + chaos."
    }
  ]
}
```

## 5. GitHub Pages Static Deployment

`frontend/vite.config.ts` sets `base: './'` so `frontend/dist/` deploys cleanly under either a custom domain or a repository subpath (`https://<user>.github.io/Break-My-VLA/`). On `*.github.io` hosts (or with `?mode=demo`), the Arcade view launches immediately in Local Demo Mode without waiting for a local WebSocket server, and visitors can link directly to `#gallery` for the Benchmark & Replay Gallery.

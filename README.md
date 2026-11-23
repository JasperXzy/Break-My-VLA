# Break-My-VLA

Interactive physical/visual/systems chaos playground and closed-loop robustness benchmark for Vision-Language-Action (VLA) models (`Pi0.5`, `Isaac GR00T`, `SmolVLA`, `OpenVLA-OFT`) across `BF16` / `FP8` / `INT8` / `NVFP4` precision modes.

## Architecture

- `backend/chaos_sim/` — FastAPI + MuJoCo/LIBERO environment with real-time `ChaosWrapper` perturbations
- `backend/policy_servers/` — Process-isolated VLA inference servers (`POST /act`) supporting PyTorch & TensorRT
- `backend/shared/` — Shared request/response & telemetry protocol schemas
- `frontend/` — Real-time Web Arcade UI (dual camera stream, chaos controls, action-chunk & latency HUD)
- `benchmarks/` — Auto-Chaos reproducible evaluation suites (Levels 0–5)
- `scripts/` — `colab run --gpu G4` job entrypoints and TensorRT export scripts

See [`SPEC.md`](SPEC.md) for architecture and API contracts, and [`AGENTS.md`](AGENTS.md) for Codex / Antigravity multi-agent workflow rules.

# Break-My-VLA Technical Specification

## 1. Goal
Build a real-time interactive web sandbox and reproducible Auto-Chaos benchmark where users (or automated suites) inject live physical, visual, systems, and quantization perturbations while open-source VLA policies attempt closed-loop manipulation tasks in MuJoCo / LIBERO.

## 2. Decoupled Client-Server Architecture
To avoid dependency conflicts across VLA frameworks (`openpi`, `lerobot`, `GR00T`, `openvla-oft`, `TensorRT`), the system is strictly split into three decoupled components:

1. **Frontend Arcade UI (`frontend/`)**
   - Vite + React + TypeScript + Tailwind
   - Connects to Chaos Sim Server via WebSocket (`/ws/session`) for live dual-camera frames (third-person + wrist), telemetry (latency, Hz, action chunk trajectory, task success), and live chaos triggers.
2. **Chaos Sim Server (`backend/chaos_sim/`)**
   - FastAPI + MuJoCo / LIBERO (`ChaosWrapper`)
   - Owns simulation stepping, perturbation injection, latency/dropout queues, episode recording (GIF/MP4 + failure replay JSON), and calls Policy Server via HTTP/ZeroMQ.
3. **Isolated Policy Servers (`backend/policy_servers/`)**
   - Each model family runs in its own isolated `uv`/Docker environment and exposes a uniform HTTP contract:
     - `GET /health` -> `{ "model": "pi05_libero", "precisions": ["bf16", "fp8", "nvfp4"], "device": "cuda:0" }`
     - `POST /act` -> takes `{ "rgb_main_b64", "rgb_wrist_b64", "proprio", "prompt", "precision", "denoise_steps" }` and returns `{ "action_chunk": [[...]], "infer_ms": float, "vram_mb": float }`

## 3. Four Chaos Skill Categories
1. **Physical Chaos**: object nudge / teleport (`qpos`), end-effector external impulse (`xfrc_applied`), table friction / gravity glitch.
2. **Visual Chaos**: dynamic lighting & color shift (`light_diffuse`, `geom_rgba`), camera extrinsic jitter (`cam_pos`, `cam_quat`), random occlusion patch / Gaussian noise / motion blur.
3. **Systems Chaos**: configurable inference latency injection (`0–300 ms`), wrist camera blackout (`sensor_dropout`).
4. **Precision & Denoise Chaos**: runtime switching between `BF16` / `FP8` / `INT8` / `NVFP4` and Flow-Matching denoise steps (`10 -> 3 -> 1`).

## 4. MVP Milestones
- **M1 (Vertical Slice)**: Mock Policy Server + LIBERO `ChaosWrapper` (5 core perturbations) + FastAPI WebSocket streaming + contract tests passing on `gcp-dev` CPU.
- **M2 (Real VLA on G4 / 5090)**: Connect `pi05_libero` and `SmolVLA` policy servers via `colab run --gpu G4` (or local RTX 5090 over Tailscale).
- **M3 (Arcade Frontend)**: Live dual-camera UI, 4 chaos panels, real-time latency & action-chunk HUD, one-click GIF replay export.
- **M4 (Quantization & Auto-Chaos Suite)**: TensorRT `BF16` / `FP8` / `NVFP4` comparison + Level 0–5 Auto-Chaos benchmark report & static GitHub Pages showcase.

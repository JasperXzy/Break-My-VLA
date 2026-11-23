# AGENTS.md — Break-My-VLA Multi-Agent Guide

This repository is developed collaboratively by **Codex CLI (`codex`)** and **Antigravity CLI (`agy`)** on `gcp-dev` (4 vCPU, 16 GB RAM, Spot CPU host) paired with **Colab G4 (96 GB Blackwell)** via `colab` CLI and an optional local **RTX 5090** over Tailscale.

## 1. Role & Ownership Split
- **Codex CLI (`codex`)**: Backend architecture, `backend/chaos_sim/`, `backend/policy_servers/`, `backend/shared/`, TensorRT / NVFP4 scripts, Auto-Chaos benchmark runner, and strict backend/concurrency code review.
- **Antigravity CLI (`agy`)**: `frontend/` Arcade UI & interactive visualizations, integration testing, browser visual verification, README/docs, and cross-reviewing API ergonomics against `SPEC.md`.
- **Conflict Prevention**: Never edit the same branch simultaneously from two agents. Use `git worktree` (e.g., `git worktree add ../Break-My-VLA-ui -b feat/frontend`) and synchronize solely through `SPEC.md` and `backend/shared/` schemas.

## 2. Compute & Cost Guardrails (CRITICAL)
- `gcp-dev` is a **CPU-only** host. Never run heavy VLA PyTorch/TensorRT GPU inference on `gcp-dev` CPU; use lightweight mock servers and unit tests on `gcp-dev`.
- For GPU workloads (VLA inference, TensorRT `FP8`/`NVFP4` compilation, Auto-Chaos rollouts), use **`colab run --gpu G4 <script.py>`** (ephemeral VM that auto-tears down after completion) rather than leaving idle `colab new` sessions running.
- Check compute balance anytime via `colab usage`.

## 3. Environment & Verification Commands
- Python tooling: `uv` (prefer `uv run pytest`, `uv run ruff check .`)
- Frontend tooling: `npm` in `frontend/` (`npm run lint`, `npm run build`)
- Always run relevant unit/contract tests before committing changes.
- Keep commits atomic and scoped to a single feature or fix.

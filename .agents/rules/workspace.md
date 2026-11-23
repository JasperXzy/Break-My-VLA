---
trigger: always_on
---

# Break-My-VLA Workspace Rules

- Treat `SPEC.md` as the single source of truth for WebSocket and HTTP `/act` schemas.
- Treat `AGENTS.md` as the shared operating protocol across Antigravity (`agy`) and Codex (`codex`).
- Never launch persistent Colab GPU sessions without stopping them; prefer `colab run --gpu G4` for ephemeral GPU jobs.
- Never commit model weights (`*.pth`, `*.safetensors`, `*.onnx`, `*.engine`) or raw large videos to Git.

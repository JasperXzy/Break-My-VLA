# M2 LIBERO and SmolVLA smoke comparison

From the repository root, run the following command with an authenticated
Colab CLI account.

```bash
python3 scripts/colab_m2_smolvla.py launch
```

The launcher packages the checked-out backend and benchmark, starts
`colab run --gpu G4`, and retrieves the GIFs and metrics to
`benchmarks/results/m2_smoke/` with a SHA-256 check. Colab releases the GPU
when the job ends. Setup and inference run on the remote VM.
If a Colab L4 is the selected fallback, pass `launch --gpu L4`. The metrics
record the actual accelerator.

Defaults are LIBERO Object task 0, initial-state index 0, seed 0, a 280-step
budget, and up to 10 executed actions between policy calls. Pass options such
as `--suite libero_spatial --task-id 0 --initial-state-id 0` to the launcher
to use a different task. The runner starts the SmolVLA HTTP server in its own
process and queries it with both rendered cameras, the task instruction, and
the current eight-element robot state.

| Episode | Precision and denoise steps | Intervention |
| --- | --- | --- |
| Baseline | BF16 autocast, 10 | — |
| Object nudge | BF16 autocast, 10 | Translate the goal object's free joint by +0.12 m X and −0.08 m Y at step 20 |
| Lighting shift | BF16 autocast, 10 | Set diffuse brightness to 0.18 and saturation to 0.7 at step 20 |
| INT8 and low denoise | INT8 weights, 2 | Quantize MLP and projection weights per output channel and use BF16 arithmetic |

Each episode starts from the same initial state with the same seeded flow
noise. A new observation is sent at each replan and immediately after a
scheduled intervention. Each run records the LIBERO success predicate, steps,
number of HTTP policy calls, inference latency, peak PyTorch allocation in
the policy process, and both cameras in the replay. Peak allocation includes
the resident BF16 and INT8 variants. The metrics also record the checkpoint,
asset revisions, source bundle hash, runtime versions, GPU, and renderer.

To use the live interface on an already configured GPU runtime, start these
services in separate terminals in the benchmark's Python environment.

```bash
python -m backend.policy_servers.smolvla_server
CHAOS_SIM_ENV=libero LIBERO_SUITE=libero_object python -m backend.chaos_sim.server
```

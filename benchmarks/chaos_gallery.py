"""Aggregate measured Auto-Chaos rollouts into the Gallery 1.0 format."""

import numpy as np

from backend.policy_servers.groot_server import CHECKPOINT as GROOT_CHECKPOINT
from backend.policy_servers.pi05_server import CHECKPOINT as PI05_CHECKPOINT
from backend.policy_servers.smolvla_server import CHECKPOINT as SMOLVLA_CHECKPOINT
from benchmarks.chaos_matrix import AXES, FULL_STEPS

MODEL_INFO = {
    SMOLVLA_CHECKPOINT: ("SmolVLA", 0.45),
    PI05_CHECKPOINT: ("Pi0.5", 3.6168),
    GROOT_CHECKPOINT: ("Isaac GR00T", 3.0),
}
SCORED = ("success", "step_limit")
ALIGNMENT_FIELDS = ("action_mse_vs_bf16", "action_l1_vs_bf16", "action_cos_vs_bf16")


def latency_summary(samples: list[float]) -> dict:
    if not samples:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "fps": 0.0}
    mean = float(np.mean(samples))
    return {
        "mean_ms": round(mean, 3),
        "p50_ms": round(float(np.percentile(samples, 50)), 3),
        "p95_ms": round(float(np.percentile(samples, 95)), 3),
        "p99_ms": round(float(np.percentile(samples, 99)), 3),
        "fps": round(1000 / mean, 3) if mean else 0.0,
    }


def entry(
    alias: str, model_name: str, precision: str, runs: list[dict], *, mock: bool, backend=None
) -> dict:
    rates = {}
    for axis in AXES:
        scored = [
            run["success"]
            for run in runs
            if run["axis"] == axis
            and run["nominal_precision"] == precision
            and run["status"] in SCORED
        ]
        if scored:
            rates[axis] = sum(scored) / len(scored)
    clean = [run for run in runs if run["axis"] == "clean" and run["precision"] == precision]
    calls = [call for run in clean for call in run["calls"]]
    memory = [
        call["vram_mb"] for run in runs if run["precision"] == precision for call in run["calls"]
    ]
    checkpoint = GROOT_CHECKPOINT if model_name.startswith(f"{GROOT_CHECKPOINT}/") else model_name
    family, size = ("CPU Mock", 0.0) if mock else MODEL_INFO.get(checkpoint, (alias, 0.0))
    row = {
        "id": f"{alias}-{precision}",
        "model": f"mock/{alias}" if mock else model_name,
        "family": family,
        "precision": precision,
        "denoise_steps": FULL_STEPS,
        "params_b": size,
        "vram_gb": round(max(memory, default=0.0) / 1024, 3),
        "overall_score": round(sum(rates.values()) / len(rates), 6) if rates else 0.0,
        "latency": latency_summary([call["infer_ms"] for call in calls]),
        # Gallery v1 accepts ratios and converts them to display percentages.
        "radar_scores": {axis: round(rates.get(axis, 0.0), 6) for axis in AXES},
    }
    if backend is not None:
        row["backend"] = backend
    alignment = [run["alignment"] for run in clean if run.get("alignment")]
    if alignment:
        row["alignment"] = {
            key: float(np.mean([values[key] for values in alignment])) for key in ALIGNMENT_FIELDS
        }
    return row


def comparison(baseline: dict, degraded: dict, *, tag: str) -> dict:
    def replay_fields(run: dict) -> dict:
        latency = latency_summary([call["effective_infer_ms"] for call in run["calls"]])
        return {
            "label": (
                f"{run['model']} · {run['precision'].upper()} · "
                f"{run['denoise_steps']} denoise steps"
            ),
            "precision": run["precision"],
            "latency_ms": latency["p50_ms"],
            "gif": run["gif"],
            "outcome": "pass" if run["success"] else "fail",
        }

    fields = {
        f"{prefix}_{key}": value
        for prefix, run in (("baseline", baseline), ("degraded", degraded))
        for key, value in replay_fields(run).items()
    }
    name = tag.replace("_", " ").title()
    return {
        "id": degraded["id"],
        "title": f"{degraded['model']} · {name} · Level {degraded['level']}",
        "chaos_tag": tag,
        "prompt": degraded["prompt"],
        "failure_note": (
            f"Clean {fields['baseline_outcome']} in {baseline['steps']} steps. "
            f"Comparison {fields['degraded_outcome']} in {degraded['steps']} steps."
        ),
        **fields,
    }


def build_gallery(metrics: dict) -> dict:
    mock = metrics["mode"] == "cpu_mock"
    entries, replays = [], []
    for alias, policy in metrics["models"].items():
        runs = [run for run in metrics["runs"] if run["model"] == alias]
        for precision in policy["precisions"]:
            backend = policy.get("backends", {}).get(precision)
            entries.append(
                entry(alias, policy["health"]["model"], precision, runs, mock=mock, backend=backend)
            )
        recorded = [run for run in runs if run["gif"] and run["status"] in SCORED]
        baseline = {
            (run["task"], run["nominal_precision"]): run
            for run in recorded
            if run["axis"] == "clean"
        }
        for run in recorded:
            is_clean = run["axis"] == "clean"
            precision = policy["reference_precision"] if is_clean else run["nominal_precision"]
            clean = baseline.get((run["task"], precision))
            if clean is not None and clean is not run:
                tag = "precision_drop" if is_clean else run["tag"]
                replays.append(comparison(clean, run, tag=tag))

    notes = (
        "Radar axes use rollout success rates. The overall score averages measured axes. "
        "Unmeasured axes display as zero. Latency and FPS use clean policy calls. "
        "Replay latency includes injected delay."
    )
    if mock:
        notes += " CPU Mock uses one tabletop task and simulated policy and system delays."
    milestone = metrics.get("milestone", "M3")
    environment = "CPU Mock" if mock else "LIBERO"
    return {
        "schema_version": "1.0",
        "suite_name": f"{milestone} Auto-Chaos · {environment}",
        "generated_at": metrics["generated_at"],
        "hardware": metrics["hardware"],
        "notes": notes,
        "entries": entries,
        "replays": replays,
    }

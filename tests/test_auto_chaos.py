import json
import subprocess
import sys
from pathlib import Path
from typing import get_args

import httpx
import pytest
from PIL import Image

from backend.chaos_sim.chaos import ChaosWrapper
from backend.chaos_sim.cpu_env import CpuPickEnv
from backend.shared.schemas import ChaosSkill, PolicyActRequest
from benchmarks.auto_chaos import options, probe_alignment, read_health
from benchmarks.chaos_gallery import ALIGNMENT_FIELDS, build_gallery
from benchmarks.chaos_matrix import AXES, DENOISE_BY_LEVEL, stress_matrix

ROOT = Path(__file__).resolve().parents[1]


def test_matrix_has_each_skill_at_each_level_and_uses_available_precision():
    args = options(["--mock", "--levels", "0-5", "--tasks", "0,libero_spatial:1"])
    assert args.levels == tuple(range(6))
    assert [task.label for task in args.tasks] == ["libero_object:0", "libero_spatial:1"]
    cases = stress_matrix(("bf16", "int8"), args.levels)
    assert len(cases) == 2 * (1 + 5 * (len(get_args(ChaosSkill)) + 1))
    assert {case.axis for case in cases} == set(AXES)
    for level in range(1, 6):
        cases_at_level = [
            case for case in cases if case.level == level and case.nominal_precision == "bf16"
        ]
        assert {case.tag for case in cases_at_level if case.command} == set(get_args(ChaosSkill))
        quant = next(case for case in cases_at_level if case.axis == "quantization")
        assert quant.precision == "int8"
        assert quant.denoise_steps == DENOISE_BY_LEVEL[level]


def test_auto_chaos_cli_mock_writes_gallery_and_gif_pairs(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.auto_chaos",
            "--mock",
            "--models",
            "smolvla,pi05",
            "--precisions",
            "bf16,fp8",
            "--tasks",
            "0",
            "--levels",
            "0",
            "5",
            "--inject-step",
            "2",
            "--max-steps",
            "36",
            "--output",
            str(tmp_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    raw = json.loads((tmp_path / "metrics.json").read_text())
    gallery = json.loads((tmp_path / "gallery.json").read_text())
    reference = json.loads((ROOT / "benchmarks/results/01_smolvla_libero_l4_real.json").read_text())
    assert raw["mode"] == "cpu_mock"
    assert set(raw["models"]) == {"smolvla", "pi05"}
    assert len(raw["runs"]) == 40
    assert {run["level"] for run in raw["runs"]} == {0, 5}
    assert {run["status"] for run in raw["runs"]} <= {"success", "step_limit"}
    assert any(run["success"] for run in raw["runs"] if run["axis"] == "clean")

    delayed = [run for run in raw["runs"] if run["tag"] == "latency_inject"]
    assert len(delayed) == 4
    for run in delayed:
        assert run["injected_at_step"] == 2
        assert run["command"]["params"]["latency_ms"] == 150
        delayed_call = next(call for call in run["calls"] if call["step"] == 2)
        assert delayed_call["effective_infer_ms"] - delayed_call["infer_ms"] == 150
    assert {
        (run["nominal_precision"], run["precision"], run["denoise_steps"])
        for run in raw["runs"]
        if run["axis"] == "quantization"
    } == {("bf16", "fp8", 1), ("fp8", "fp8", 1)}

    assert gallery.keys() == reference.keys()
    assert gallery["schema_version"] == reference["schema_version"]
    assert len(gallery["entries"]) == 4
    for entry in gallery["entries"]:
        assert reference["entries"][0].keys() <= entry.keys()
        assert entry["model"].startswith("mock/")
        assert reference["entries"][0]["latency"].keys() <= entry["latency"].keys()
        assert entry["radar_scores"].keys() == reference["entries"][0]["radar_scores"].keys()
        assert 0 <= entry["overall_score"] <= 1
    assert {replay["chaos_tag"] for replay in gallery["replays"]} >= {
        "object_nudge",
        "lighting_shift",
        "latency_inject",
        "precision_drop",
    }
    for replay in gallery["replays"]:
        assert replay.keys() == reference["replays"][0].keys()
        for key in ("baseline_gif", "degraded_gif"):
            with Image.open(tmp_path / replay[key]) as gif:
                assert gif.format == "GIF"
                assert gif.n_frames > 1


def test_m4_mock_covers_all_models_precisions_and_stress_levels(tmp_path):
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.auto_chaos",
            "--mock",
            "--levels",
            "0-5",
            "--max-steps",
            "5",
            "--inject-step",
            "2",
            "--output",
            str(tmp_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    raw = json.loads((tmp_path / "metrics.json").read_text())
    gallery = json.loads((tmp_path / "gallery.json").read_text())
    assert raw["milestone"] == "M4" and raw["mode"] == "cpu_mock"
    assert set(raw["models"]) == {"smolvla", "pi05", "groot"}
    assert raw["levels"] == list(range(6))
    assert len(raw["runs"]) == 3 * 4 * (1 + 5 * (len(get_args(ChaosSkill)) + 1))
    assert {run["axis"] for run in raw["runs"]} == set(AXES)
    for model in raw["models"].values():
        assert model["backends"]["nvfp4"] == "tensorrt"
        assert model["skipped_precisions"] == []
    for record in raw["runs"]:
        assert record["backend"] == raw["models"][record["model"]]["backends"][record["precision"]]
        assert record["latency"]["p99_ms"] >= record["latency"]["p95_ms"]
        if record["axis"] == "clean":
            assert record["alignment"].keys() == set(ALIGNMENT_FIELDS)
        else:
            assert "alignment" not in record
        if record["axis"] == "quantization" and record["nominal_precision"] in {"fp8", "int8"}:
            assert record["precision"] == "nvfp4"
            assert record["denoise_steps"] == DENOISE_BY_LEVEL[record["level"]]

    assert gallery["schema_version"] == "1.0" and len(gallery["entries"]) == 12
    assert gallery["suite_name"].startswith("M4 Auto-Chaos")
    assert gallery["replays"]
    for entry in gallery["entries"]:
        mode = entry["precision"]
        assert entry["latency"]["p99_ms"] > 0
        alignment = entry["alignment"]
        assert -1 <= alignment["action_cos_vs_bf16"] <= 1
        assert (alignment["action_l1_vs_bf16"] == 0) == (mode == "bf16")
        assert (alignment["action_mse_vs_bf16"] == 0) == (mode == "bf16")
        assert entry["backend"] == ("tensorrt" if mode == "nvfp4" else "pytorch")


def test_policy_alignment_reuses_identical_observations_and_ignores_incompatible_chunks():
    requests = []

    def act(request):
        wire = PolicyActRequest.model_validate_json(request.content)
        requests.append(wire)
        actions = {"bf16": [[1] * 7], "fp8": [[2] * 7], "nvfp4": [[2] * 7] * 2}
        return httpx.Response(
            200,
            json={
                "action_chunk": actions[wire.precision],
                "infer_ms": 0.1,
                "vram_mb": 0.0,
            },
        )

    with httpx.Client(transport=httpx.MockTransport(act), base_url="http://policy") as client:
        align = probe_alignment(client, ChaosWrapper(CpuPickEnv()), ("fp8", "nvfp4"))
    assert [request.precision for request in requests] == ["bf16", "fp8", "nvfp4"]
    assert {request.denoise_steps for request in requests} == {10}
    assert requests[0].model_copy(update={"precision": "fp8"}) == requests[1]
    assert requests[0].model_copy(update={"precision": "nvfp4"}) == requests[2]
    assert list(align) == ["fp8"]
    assert align["fp8"] == pytest.approx(
        {
            "action_mse_vs_bf16": 1,
            "action_l1_vs_bf16": 1,
            "action_cos_vs_bf16": 1,
        }
    )


def test_legacy_health_and_gallery_records_can_omit_m4_metadata():
    wire = {"model": "old-model", "device": "cuda:0", "precisions": ["bf16"]}
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json=wire))
    with httpx.Client(transport=transport, base_url="http://policy") as client:
        health, backends = read_health(client)
    assert health.model_dump(mode="json") == wire and backends == {}

    metric = {
        "mode": "libero",
        "hardware": "test",
        "generated_at": "2026-11-24T00:00:00Z",
        "models": {"legacy": {"health": wire, "precisions": ["bf16"]}},
        "runs": [
            {
                "model": "legacy",
                "axis": "clean",
                "precision": "bf16",
                "nominal_precision": "bf16",
                "status": "success",
                "success": True,
                "calls": [{"infer_ms": 42.0, "vram_mb": 1024}],
                "gif": None,
            }
        ],
    }
    gallery = build_gallery(metric)
    assert gallery["suite_name"] == "M3 Auto-Chaos · LIBERO"
    (row,) = gallery["entries"]
    assert row["overall_score"] == 1.0 and row["latency"]["p50_ms"] == 42
    assert "alignment" not in row and "backend" not in row

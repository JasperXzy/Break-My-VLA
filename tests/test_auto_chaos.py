import json
import subprocess
import sys
from pathlib import Path
from typing import get_args

from PIL import Image

from backend.shared.schemas import ChaosSkill
from benchmarks.auto_chaos import options
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
        assert entry.keys() == reference["entries"][0].keys()
        assert entry["model"].startswith("mock/")
        assert entry["latency"].keys() == reference["entries"][0]["latency"].keys()
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

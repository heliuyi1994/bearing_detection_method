"""CLI 冒烟测试：simulate / diagnose 子命令跑通且产出文件、结论正确。"""

import json
from pathlib import Path

import pytest

from bearing_fault.cli import main


class TestSimulate:
    def test_outputs_created(self, tmp_path: Path, capsys):
        rc = main(["simulate", "--fault", "outer", "--duration", "1.0",
                   "--out", str(tmp_path)])
        assert rc == 0
        assert (tmp_path / "signal.npz").exists()
        assert (tmp_path / "waveform.png").exists()
        assert "107.36" in capsys.readouterr().out  # BPFO 理论值


class TestDiagnose:
    @pytest.mark.parametrize("fault", ["inner", "outer", "ball", "cage"])
    def test_verdict_and_outputs(self, fault, tmp_path: Path):
        rc = main(["diagnose", "--fault", fault, "--seed", "1",
                   "--out", str(tmp_path)])
        assert rc == 0
        for name in ("waveform.png", "envelope_spectrum.png", "scores.png", "summary.json"):
            assert (tmp_path / name).exists(), name
        summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
        assert summary["verdict"] == fault
        assert summary["true_fault"] == fault
        assert summary["peak_relative_error"] < 0.02

    def test_diagnose_from_saved_npz(self, tmp_path: Path):
        sim_dir = tmp_path / "sim"
        main(["simulate", "--fault", "inner", "--out", str(sim_dir)])
        diag_dir = tmp_path / "diag"
        rc = main(["diagnose", "--input", str(sim_dir / "signal.npz"),
                   "--out", str(diag_dir)])
        assert rc == 0
        summary = json.loads((diag_dir / "summary.json").read_text(encoding="utf-8"))
        assert summary["verdict"] == "inner"
        assert summary["true_fault"] == "inner"

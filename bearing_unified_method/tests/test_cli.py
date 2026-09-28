# -*- coding: utf-8 -*-
"""报告输出与 CLI 冒烟测试（v2 规则）。"""

import json

import pytest

from bearing_unified.cli import main as cli_main
from bearing_unified.diagnose import diagnose
from bearing_unified.loaders import save_signal_csv
from bearing_unified.report import build_json_payload, build_text_report, write_reports
from bearing_unified.simulate import simulate


@pytest.fixture(scope="module")
def bpf_result():
    x, info = simulate("BPFI", fs=16000, duration=3.0, rpm=1150,
                       bearing="FIELD7", snr_db=8, resonance_hz=2200, seed=100)
    return diagnose(x, fs=16000, bearing="FIELD7"), info


class TestReport:
    def test_text_report_content(self, bpf_result):
        res, info = bpf_result
        text = build_text_report(res, "unit-test")
        assert "诊断报告" in text and "v2" in text
        assert "置信度" in text and "峭度" in text and "盲梳检" in text
        assert "外圈" in text and "内圈" in text and "滚珠" in text
        assert str(int(info["fs"])) in text

    def test_json_payload_serializable(self, bpf_result):
        res, _ = bpf_result
        payload = build_json_payload(res, "unit-test")
        s = json.dumps(payload, ensure_ascii=False)
        d = json.loads(s)
        assert d["rules"] == "v2" and "verdict" in d
        assert set(d["confidence"]) >= {"outer", "inner", "ball"}
        assert "combs" in d and "detail" in d["combs"]

    def test_write_reports_creates_files(self, bpf_result, tmp_path):
        res, _ = bpf_result
        paths = write_reports(res, tmp_path / "out")
        assert paths["png"].stat().st_size > 30_000
        assert paths["txt"].exists() and paths["json"].exists()


class TestCLI:
    def test_simulate_analyze_roundtrip(self, tmp_path, capsys, monkeypatch):
        sim_csv = tmp_path / "sim.csv"
        monkeypatch.chdir(tmp_path)
        rc = cli_main(["simulate", "--fault", "BPFI", "--snr", "10",
                       "--seed", "5", "--out", str(sim_csv)])
        assert rc == 0 and sim_csv.exists()
        out = capsys.readouterr().out
        assert "BPFI" in out and "84" in out  # FIELD7@1150 的 BPFI≈84.7 Hz

        rc = cli_main(["analyze", str(sim_csv), "--fs", "16000",
                       "--bearing", "FIELD7", "--out", str(tmp_path / "rep")])
        assert rc == 0
        captured = capsys.readouterr()
        assert "判定" in captured.out and "置信度" in captured.out
        rep = tmp_path / "rep"
        assert (rep / "diagnosis_report.png").exists()
        assert (rep / "diagnosis_report.json").exists()
        verdict = json.loads((rep / "diagnosis_report.json").read_text())
        assert verdict["verdict"] in ("BPFI", "normal")
        assert set(verdict["confidence"]) >= {"outer", "inner", "ball"}

    def test_bands_subcommand(self, capsys):
        rc = cli_main(["bands", "--bearing", "FIELD7", "--rpm-window", "1100:1200"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "BPFO" in out and "BPFI" in out and "BSF" in out
        assert "47.2" in out or "47.3" in out  # 外圈带下沿

    def test_bands_custom_window_recomputes(self, capsys):
        rc = cli_main(["bands", "--bearing", "FIELD7", "--rpm-window", "900:1000"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "38.6" in out or "38.7" in out  # 2.57832×15≈38.67 外圈下沿

    def test_bearings_listing(self, capsys):
        assert cli_main(["bearings", "--rpm", "1150"]) == 0
        out = capsys.readouterr().out
        assert "SKF6205" in out and "BPFO" in out and "FIELD7" in out

    def test_missing_input_error(self, capsys):
        rc = cli_main(["analyze", "/nonexistent.csv", "--fs", "1000"])
        assert rc == 2
        assert "错误" in capsys.readouterr().err

    def test_invalid_bearing_error(self, capsys, tmp_path):
        import numpy as np
        p = tmp_path / "s.npy"
        np.save(p, np.random.default_rng(0).standard_normal(5000))
        rc = cli_main(["analyze", str(p), "--fs", "1000", "--bearing", "NOPE999"])
        assert rc == 2
        assert "未知轴承" in capsys.readouterr().err

    def test_invalid_rpm_window(self, capsys, tmp_path):
        import numpy as np
        p = tmp_path / "s.npy"
        np.save(p, np.random.default_rng(0).standard_normal(5000))
        rc = cli_main(["analyze", str(p), "--fs", "16000",
                       "--rpm-window", "1200:1100"])
        assert rc == 2


class TestSimulateCSVFormat:
    def test_csv_header_comment(self, tmp_path):
        x, _ = simulate("normal", duration=0.3, seed=2)
        p = tmp_path / "c.csv"
        save_signal_csv(p, x, 12000)
        assert p.read_text().startswith("# fs=12000")

"""报告输出与 CLI 冒烟测试。"""

import json

import pytest

from bearing_diag.cli import main as cli_main
from bearing_diag.diagnose import diagnose
from bearing_diag.loaders import save_signal_csv
from bearing_diag.report import build_json_payload, build_text_report, write_reports
from bearing_diag.simulate import simulate


@pytest.fixture(scope="module")
def bpfo_result():
    x, info = simulate("BPFO", fs=12000, duration=2.0, snr_db=8, seed=100)
    return diagnose(x, fs=12000, shaft_freq=info["shaft_freq"], bearing="SKF6205"), info


class TestReport:
    def test_text_report_content(self, bpfo_result):
        res, info = bpfo_result
        text = build_text_report(res, "unit-test")
        assert "诊断报告" in text and "BPFO" in text
        assert "置信度" in text and "峭度" in text
        assert str(int(info["fs"])) in text

    def test_json_payload_serializable(self, bpfo_result):
        res, _ = bpfo_result
        payload = build_json_payload(res, "unit-test")
        s = json.dumps(payload, ensure_ascii=False)
        assert "verdict" in json.loads(s)["match"]

    def test_write_reports_creates_files(self, bpfo_result, tmp_path):
        res, _ = bpfo_result
        paths = write_reports(res, tmp_path / "out")
        assert paths["png"].stat().st_size > 30_000
        assert paths["txt"].exists() and paths["json"].exists()


class TestCLI:
    def test_simulate_analyze_roundtrip(self, tmp_path, capsys, monkeypatch):
        sim_csv = tmp_path / "sim.csv"
        monkeypatch.chdir(tmp_path)
        rc = cli_main(["simulate", "--fault", "BPFO", "--snr", "10",
                       "--seed", "5", "--out", str(sim_csv)])
        assert rc == 0 and sim_csv.exists()
        out = capsys.readouterr().out
        assert "BPFO" in out and "107" in out  # 特征频率打印

        rc = cli_main(["analyze", str(sim_csv), "--fs", "12000",
                       "--rpm", "1797", "--bearing", "SKF6205",
                       "--out", str(tmp_path / "rep")])
        assert rc == 0
        captured = capsys.readouterr()
        assert "判定" in captured.out
        rep = tmp_path / "rep"
        assert (rep / "diagnosis_report.png").exists()
        assert (rep / "diagnosis_report.json").exists()
        verdict = json.loads((rep / "diagnosis_report.json").read_text())
        assert verdict["match"]["verdict"] in ("BPFO", "normal")

    def test_fault_freqs_direct(self, tmp_path, capsys):
        import numpy as np
        from bearing_diag.simulate import simulate as sim
        x, _ = sim("BPFO", fs=12000, duration=2.0, snr_db=10, seed=7)
        p = tmp_path / "s.npy"
        np.save(p, x)
        rc = cli_main(["analyze", str(p), "--fs", "12000", "--shaft-freq", "29.95",
                       "--fault-freqs", '{"BPFO":107.4,"BPFI":162.2,'
                                        '"BSF":70.6,"FTF":11.9}',
                       "--out", str(tmp_path / "r2")])
        assert rc == 0
        assert "诊断报告" in capsys.readouterr().out

    def test_bearings_listing(self, capsys):
        assert cli_main(["bearings", "--rpm", "1797"]) == 0
        out = capsys.readouterr().out
        assert "SKF6205" in out and "BPFO" in out

    def test_missing_input_error(self, capsys):
        rc = cli_main(["analyze", "/nonexistent.csv", "--fs", "1000", "--rpm", "1800"])
        assert rc == 2
        assert "错误" in capsys.readouterr().err

    def test_invalid_bearing_error(self, capsys, tmp_path):
        import numpy as np
        p = tmp_path / "s.npy"
        np.save(p, np.random.default_rng(0).standard_normal(5000))
        rc = cli_main(["analyze", str(p), "--fs", "1000", "--rpm", "1800",
                       "--bearing", "NOPE999"])
        assert rc == 2
        assert "未知轴承" in capsys.readouterr().err


class TestSimulateCSVFormat:
    def test_csv_header_comment(self, tmp_path):
        x, _ = simulate("normal", duration=0.3, seed=2)
        p = tmp_path / "c.csv"
        save_signal_csv(p, x, 12000)
        assert p.read_text().startswith("# fs=12000")

"""仿真器与匹配器单元测试 + 端到端诊断准确率验证。"""

import numpy as np
import pytest

from bearing_diag.diagnose import diagnose
from bearing_diag.fault_freqs import BUILTIN_BEARINGS, FAULT_NAMES_CN
from bearing_diag.matcher import match_fault_frequencies
from bearing_diag.simulate import simulate

BEARING = BUILTIN_BEARINGS["SKF6205"]
RPM = 1797.0


class TestSimulate:
    def test_reproducible_with_seed(self):
        a, _ = simulate("BPFO", seed=7)
        b, _ = simulate("BPFO", seed=7)
        assert np.array_equal(a, b)

    def test_normal_has_no_impulses(self):
        x, info = simulate("normal", snr_db=10, seed=3)
        from bearing_diag.indicators import time_domain_indicators
        ind = time_domain_indicators(x)
        assert ind["kurtosis"] < 4.5  # 无冲击 → 峭度接近 3

    def test_fault_impulsive(self):
        x, _ = simulate("BPFO", snr_db=10, seed=3)
        from bearing_diag.indicators import time_domain_indicators
        ind = time_domain_indicators(x)
        assert ind["kurtosis"] > 4.5

    def test_info_contains_truth(self):
        _, info = simulate("BSF", rpm=1500, seed=1)
        ff = BEARING.fault_frequencies(1500 / 60)
        assert info["true_fault_freq"] == pytest.approx(ff["BSF"])

    def test_invalid_fault(self):
        with pytest.raises(ValueError):
            simulate("BAD")


class TestMatcher:
    def _env_of(self, x, fs, max_f=1000):
        from bearing_diag.spectrum import envelope_spectrum
        return envelope_spectrum(x, fs, max_freq=max_f)

    def test_synthetic_bpfo_envelope_matched(self):
        x, info = simulate("BPFO", fs=12000, duration=2.0, snr_db=10, seed=11)
        f, a = self._env_of(x, 12000)
        ff = info["true_fault_freqs"]
        m = match_fault_frequencies(f, a, ff)
        assert m.verdict == "BPFO"
        assert m.confidence > 0.3

    def test_pure_noise_no_fault(self):
        rng = np.random.default_rng(5)
        x = rng.standard_normal(32768)
        f, a = self._env_of(x, 12000)
        ff = BEARING.fault_frequencies(1797 / 60)
        m = match_fault_frequencies(f, a, ff)
        assert m.verdict == "normal"

    def test_ftf_vs_bpfo_deconfliction(self):
        # 保持架故障：BPFO = 9×FTF，其谐波族会污染 BPFO 证据 → 应判 FTF
        x, info = simulate("FTF", fs=12000, duration=4.0, snr_db=8, seed=21)
        f, a = self._env_of(x, 12000, max_f=700)
        m = match_fault_frequencies(f, a, info["true_fault_freqs"])
        assert m.verdict == "FTF", [c.fault for c in m.candidates]

    def test_undetectable_low_freq_noted(self):
        f = np.linspace(1, 500, 500)
        a = np.abs(np.random.default_rng(0).standard_normal(500))
        m = match_fault_frequencies(f, a, {"FTF": 0.3})  # 低于分辨率下限
        assert m.verdict == "normal"
        assert any("分辨率" in n for n in m.notes)


class TestDiagnoseEndToEnd:
    """四种故障 × 多档 SNR 端到端判型验证。"""

    @pytest.mark.parametrize("fault", ["BPFO", "BPFI", "BSF", "FTF"])
    @pytest.mark.parametrize("snr", [15.0, 5.0])
    def test_correct_verdict(self, fault, snr):
        dur = 4.0 if fault == "FTF" else 2.0
        x, info = simulate(fault, fs=12000, duration=dur, snr_db=snr, seed=42)
        res = diagnose(x, fs=12000, shaft_freq=info["shaft_freq"],
                       bearing=BEARING)
        assert res.verdict == fault, (
            f"{fault}@{snr}dB 被误判为 {res.verdict}"
            f"（得分 {[(c.fault, round(c.score,2)) for c in res.match.candidates]}）"
        )

    def test_normal_signal_verdict(self):
        x, info = simulate("normal", fs=12000, duration=2.0, snr_db=10, seed=9)
        res = diagnose(x, fs=12000, shaft_freq=info["shaft_freq"],
                       bearing=BEARING)
        assert res.verdict == "normal"

    @pytest.mark.parametrize("fault", ["BPFO", "BPFI"])
    def test_low_snr_no_wrong_location(self, fault):
        # 低 SNR 允许判 normal（漏报），但不允许判错部位（误报）
        x, info = simulate(fault, fs=12000, duration=2.0, snr_db=-6, seed=13)
        res = diagnose(x, fs=12000, shaft_freq=info["shaft_freq"],
                       bearing=BEARING)
        assert res.verdict in (fault, "normal")

    def test_manual_band(self):
        x, info = simulate("BPFO", fs=12000, duration=2.0, snr_db=8, seed=4)
        res = diagnose(x, fs=12000, shaft_freq=info["shaft_freq"],
                       bearing=BEARING, manual_band=(1500, 4500))
        assert res.band_source == "manual"
        assert res.verdict == "BPFO"

    def test_explicit_fault_freqs(self):
        x, info = simulate("BPFO", fs=12000, duration=2.0, snr_db=10, seed=6)
        ff = BEARING.fault_frequencies(info["shaft_freq"])
        res = diagnose(x, fs=12000, shaft_freq=info["shaft_freq"],
                       fault_freqs=ff)
        assert res.verdict == "BPFO"

    def test_requires_fault_freq_source(self):
        with pytest.raises(ValueError, match="fault_freqs"):
            diagnose(np.zeros(1000), fs=1000, shaft_freq=20)

    def test_result_fields_complete(self):
        x, info = simulate("BPFI", fs=12000, duration=2.0, snr_db=8, seed=8)
        res = diagnose(x, fs=12000, shaft_freq=info["shaft_freq"], bearing=BEARING)
        assert set(res.indicators) >= {"rms", "kurtosis", "crest_factor"}
        assert FAULT_NAMES_CN["BPFI"] in res.verdict_cn or res.verdict == "BPFI"
        assert res.env_freqs.size > 100
        assert res.cepstrum_q is not None and res.cepstrum_q.size > 0
        assert res.match is not None and len(res.match.candidates) == 4

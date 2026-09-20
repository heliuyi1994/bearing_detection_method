"""仿真器测试：信号长度、SNR 定量、故障周期正确性、可复现性。"""

import numpy as np
import pytest

from bearing_fault import Bearing, BearingSimulator, envelope_analysis
from bearing_fault.spectrum import find_peaks_near

FS = 12000.0
RPM = 1797.0
FR = RPM / 60.0


class TestBasic:
    def test_length_and_type(self):
        sim = BearingSimulator(fs=FS, duration=2.0, rpm=RPM, seed=0)
        x = sim.signal("outer")
        assert x.shape == (int(FS * 2.0),)
        assert x.dtype == np.float64

    def test_seed_reproducible(self):
        sim1 = BearingSimulator(fs=FS, duration=1.0, rpm=RPM, seed=7)
        sim2 = BearingSimulator(fs=FS, duration=1.0, rpm=RPM, seed=7)
        np.testing.assert_array_equal(sim1.signal("inner"), sim2.signal("inner"))

    def test_unknown_fault_raises(self):
        sim = BearingSimulator(fs=FS, duration=1.0, rpm=RPM)
        with pytest.raises(ValueError):
            sim.signal("gearbox")

    def test_resonance_above_nyquist_raises(self):
        with pytest.raises(ValueError):
            BearingSimulator(fs=FS, duration=1.0, rpm=RPM, resonance_freq=FS)


class TestSNR:
    def test_snr_quantitative(self):
        """噪声功率应满足 P_noise = P_signal / 10^(snr/10)。"""
        sim = BearingSimulator(fs=FS, duration=3.0, rpm=RPM, seed=3)
        comp = sim.signal("outer", snr_db=-6.0, return_components=True)
        p_sig = np.mean(comp["clean"] ** 2)
        p_noise = np.mean(comp["noise"] ** 2)
        measured_db = 10.0 * np.log10(p_sig / p_noise)
        assert measured_db == pytest.approx(-6.0, abs=0.3)

    def test_no_noise(self):
        sim = BearingSimulator(fs=FS, duration=1.0, rpm=RPM, seed=3)
        comp = sim.signal("outer", snr_db=None, return_components=True)
        assert np.all(comp["noise"] == 0.0)
        np.testing.assert_array_equal(comp["signal"], comp["clean"])


class TestFaultPeriodicity:
    """高信噪比下，包络谱最强峰应落在理论故障频率 ±2% 内。"""

    @pytest.mark.parametrize("fault", ["inner", "outer", "ball", "cage"])
    def test_envelope_peak_matches_theory(self, fault):
        sim = BearingSimulator(fs=FS, duration=4.0, rpm=RPM, seed=5)
        x = sim.signal(fault, snr_db=10.0)
        result = envelope_analysis(x, FS)
        f_theory = Bearing.sk6205().characteristic_frequencies(FR)[fault]
        hits = find_peaks_near(
            result["env_freqs"], result["env_amps"], f_theory, harmonics=3, tol=0.02
        )
        assert hits, f"{fault}: 理论频率 {f_theory:.2f} Hz 附近未找到谱峰"
        h, f_peak, _ = max(hits, key=lambda item: item[2])
        assert abs(f_peak - h * f_theory) / (h * f_theory) < 0.02

    def test_impact_interval_stats(self):
        """冲击间隔均值应接近故障周期，滑差使标准差约为 1%。"""
        sim = BearingSimulator(fs=FS, duration=5.0, rpm=RPM, seed=2, slip=0.01)
        comp = sim.signal("outer", snr_db=None, return_components=True)
        intervals = np.diff(comp["impact_times"])
        f_theory = Bearing.sk6205().bpfo(FR)
        assert intervals.mean() == pytest.approx(1.0 / f_theory, rel=0.01)
        assert intervals.std() / intervals.mean() == pytest.approx(0.01, rel=0.5)


class TestVariableSpeed:
    def test_ramp_outputs_monotonic_phase(self):
        sim = BearingSimulator(fs=FS, duration=2.0, rpm=1200, seed=0)
        comp = sim.signal("outer", speed_ramp=(1200, 1800), return_components=True)
        assert np.all(np.diff(comp["theta"]) > 0)
        assert comp["fr_t"][0] == pytest.approx(20.0)
        assert comp["fr_t"][-1] == pytest.approx(30.0, rel=0.01)

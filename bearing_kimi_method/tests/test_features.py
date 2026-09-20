"""时域特征测试：正弦波与常值信号的解析值比对。"""

import numpy as np
import pytest

from bearing_fault import time_domain_features

FS = 4000.0


def _sine(amp=2.0, freq=50.0, duration=2.0):
    t = np.arange(int(FS * duration)) / FS
    return amp * np.sin(2 * np.pi * freq * t)


class TestSine:
    """幅值 A 的正弦波：RMS=A/√2，峰值=A，峰值因子=√2，峭度=1.5。"""

    feats = time_domain_features(_sine())

    def test_rms(self):
        assert self.feats["rms"] == pytest.approx(2.0 / np.sqrt(2), rel=1e-3)

    def test_peak(self):
        assert self.feats["peak"] == pytest.approx(2.0, rel=1e-3)

    def test_crest_factor(self):
        assert self.feats["crest_factor"] == pytest.approx(np.sqrt(2), rel=1e-3)

    def test_kurtosis(self):
        # 正弦波 Pearson 峭度解析值为 1.5
        assert self.feats["kurtosis"] == pytest.approx(1.5, rel=1e-2)

    def test_skewness_zero(self):
        assert abs(self.feats["skewness"]) < 1e-6

    def test_impulse_factor(self):
        # mean(|sin|) = 2/π → 脉冲因子 = A/(2A/π) = π/2
        assert self.feats["impulse_factor"] == pytest.approx(np.pi / 2, rel=1e-3)

    def test_shape_factor(self):
        # 波形因子 = (A/√2)/(2A/π) = π/(2√2)
        assert self.feats["shape_factor"] == pytest.approx(np.pi / (2 * np.sqrt(2)), rel=1e-3)


class TestConstant:
    def test_constant_signal(self):
        feats = time_domain_features(np.full(1000, 3.0))
        assert feats["rms"] == pytest.approx(3.0)
        assert feats["peak"] == pytest.approx(3.0)
        assert feats["crest_factor"] == pytest.approx(1.0)
        assert feats["kurtosis"] == 0.0  # 方差为 0 时约定返回 0
        assert feats["skewness"] == 0.0


class TestFaultVsHealthy:
    """故障信号的 RMS/峭度/峰值因子应显著高于健康信号。"""

    def test_ordering(self):
        from bearing_fault import BearingSimulator

        sim_h = BearingSimulator(fs=12000, duration=2.0, rpm=1797, seed=1)
        sim_f = BearingSimulator(fs=12000, duration=2.0, rpm=1797, seed=1)
        f_h = time_domain_features(sim_h.signal("healthy", snr_db=-6))
        f_f = time_domain_features(sim_f.signal("outer", snr_db=-6))
        assert f_f["kurtosis"] > f_h["kurtosis"]
        assert f_f["crest_factor"] > f_h["crest_factor"]

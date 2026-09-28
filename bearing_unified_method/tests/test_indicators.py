"""时域指标单元测试：对已知分布信号校验理论值。"""

import numpy as np
import pytest

from bearing_unified.indicators import detrend, health_hint, time_domain_indicators


class TestIndicators:
    def test_gaussian_kurtosis_about_3(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(200_000)
        ind = time_domain_indicators(x)
        assert ind["kurtosis"] == pytest.approx(3.0, abs=0.1)
        assert ind["skewness"] == pytest.approx(0.0, abs=0.1)
        assert ind["rms"] == pytest.approx(1.0, abs=0.05)

    def test_sinusoid_closed_form(self):
        t = np.arange(100_000) / 10_000.0
        x = 2.0 * np.sin(2 * np.pi * 100 * t)
        ind = time_domain_indicators(x)
        assert ind["rms"] == pytest.approx(2.0 / np.sqrt(2), rel=1e-3)
        assert ind["peak"] == pytest.approx(2.0, rel=1e-3)
        # 正弦峭度理论值 1.5
        assert ind["kurtosis"] == pytest.approx(1.5, abs=0.01)
        # 峰值因子 = √2
        assert ind["crest_factor"] == pytest.approx(np.sqrt(2), rel=1e-3)

    def test_impulsive_signal_high_kurtosis(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(100_000) * 0.01
        x[::5000] += 50.0  # 稀疏强冲击
        ind = time_domain_indicators(x)
        assert ind["kurtosis"] > 20
        assert ind["clearance_factor"] > ind["crest_factor"]  # 裕度因子对稀疏冲击更大

    def test_all_zero_signal_safe(self):
        ind = time_domain_indicators(np.zeros(1000))
        assert ind["rms"] == 0 and ind["crest_factor"] == 0

    def test_too_short_raises(self):
        with pytest.raises(ValueError):
            time_domain_indicators([1.0])

    def test_detrend_removes_linear_drift(self):
        t = np.arange(5000) / 1000.0
        x = 3.0 * t + 100.0 + 0.1 * np.sin(2 * np.pi * 30 * t)
        y = detrend(x)
        assert abs(np.mean(y)) < 1e-6
        assert abs(np.polyfit(t, y, 1)[0]) < 1e-3

    def test_health_hint_buckets(self):
        assert "正常" in health_hint({"kurtosis": 3.2})
        assert "早期" in health_hint({"kurtosis": 4.5})
        assert "明显" in health_hint({"kurtosis": 9.0})

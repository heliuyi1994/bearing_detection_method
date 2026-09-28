# -*- coding: utf-8 -*-
"""preprocessing 单元测试：去直流/去趋势/零相位带通/频带夹取。"""

import numpy as np
import pytest
from scipy.signal import butter, sosfilt

from bearing_unified.preprocessing import bandpass, clamp_band, detrend, remove_dc

FS = 16000.0


class TestRemoveDcDetrend:
    def test_remove_dc_zero_mean(self):
        x = np.random.default_rng(0).standard_normal(5000) + 3.7
        y = remove_dc(x)
        assert float(np.mean(y)) == pytest.approx(0.0, abs=1e-12)

    def test_detrend_removes_linear(self):
        t = np.arange(5000) / FS
        x = 0.5 * np.sin(2 * np.pi * 100 * t) + 8.0 * t
        y = detrend(x)
        # 去趋势后线性分量应基本消失：首尾均值差远小于原趋势
        assert abs(y[:500].mean() - y[-500:].mean()) < 0.05


class TestBandpassZeroPhase:
    def _impulse(self, n=4096):
        x = np.zeros(n)
        x[n // 2] = 1.0
        return x

    def test_zero_phase_keeps_impulse_position(self):
        """零相位滤波不移动冲击时刻；sosfilt（单向）有群延迟偏移。"""
        x = self._impulse()
        center = x.size // 2
        y_zp = bandpass(x, FS, 500.0, 4000.0)
        sos = butter(4, [500.0, 4000.0], btype="bandpass", fs=FS, output="sos")
        y_1p = sosfilt(sos, x)
        idx_zp = int(np.argmax(np.abs(y_zp)))
        idx_1p = int(np.argmax(np.abs(y_1p)))
        assert abs(idx_zp - center) <= 2
        assert abs(idx_zp - center) < abs(idx_1p - center)

    def test_band_selectivity(self):
        t = np.arange(int(FS)) / FS
        x_in = np.sin(2 * np.pi * 2000 * t)
        x_out = np.sin(2 * np.pi * 100 * t)
        y_in = bandpass(x_in, FS, 1000.0, 3000.0)
        y_out = bandpass(x_out, FS, 1000.0, 3000.0)
        # 带内保留大部分能量，带外强衰减
        assert np.std(y_in) > 0.5 * np.std(x_in)
        assert np.std(y_out) < 0.01 * np.std(x_out)

    def test_invalid_band_raises(self):
        with pytest.raises(ValueError):
            bandpass(np.zeros(100), FS, 4000.0, 1000.0)
        with pytest.raises(ValueError):
            bandpass(np.zeros(100), FS, 100.0, 9000.0)


class TestClampBand:
    def test_zero_low_clamped(self):
        lo, hi = clamp_band((0.0, 500.0), FS)
        assert lo >= 1.0 and hi == 500.0

    def test_nyquist_edge_clamped(self):
        lo, hi = clamp_band((7500.0, 8000.0), FS)
        assert hi < FS / 2

    def test_normal_band_unchanged(self):
        assert clamp_band((1000.0, 2000.0), FS) == (1000.0, 2000.0)

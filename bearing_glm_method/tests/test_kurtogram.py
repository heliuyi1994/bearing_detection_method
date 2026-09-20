"""Kurtogram 单元测试：冲击频带定位。"""

import numpy as np
import pytest

from bearing_diag.kurtogram import bandpass_signal, envelope_kurtosis, kurtogram


def _impulse_train(fs, n, rate, res_freq, damping=0.04, rng=None):
    """共振衰减冲击串（测试夹具）。"""
    rng = rng or np.random.default_rng(1)
    t = np.arange(n) / fs
    x = np.zeros(n)
    decay = 2 * np.pi * damping * res_freq
    t_next = 0.0
    while t_next < n / fs:
        dt = t - t_next
        m = (dt >= 0) & (dt < 5 / decay)
        x[m] += np.exp(-decay * dt[m]) * np.sin(2 * np.pi * res_freq * dt[m])
        t_next += 1.0 / rate * (1 + 0.01 * rng.standard_normal())
    return x


class TestKurtogram:
    def test_locates_impulse_band_under_strong_tone(self):
        # 强低频正弦 + 冲击集中在 2500~3500 Hz 共振带 → 应选中共振带附近
        fs, n = 12000.0, 65536
        t = np.arange(n) / fs
        tone = 8.0 * np.sin(2 * np.pi * 60 * t)  # 能量远大于冲击
        impulses = 0.2 * _impulse_train(fs, n, rate=110.0, res_freq=3000.0,
                                        damping=0.08)
        x = tone + impulses + 0.05 * np.random.default_rng(3).standard_normal(n)
        res = kurtogram(x, fs, max_level=4)
        assert res.best_kurtosis > 2.8  # 显著高于窄带噪声基线 ≈2.0
        assert abs(res.center - 3000.0) < 750.0  # 选中带中心接近共振
        assert res.best_band[1] > res.best_band[0] > 0

    def test_pure_noise_flat_kurtosis(self):
        rng = np.random.default_rng(7)
        x = rng.standard_normal(65536)
        res = kurtogram(x, 12000.0, max_level=3)
        assert res.best_kurtosis < 2.8  # 纯噪声各带包络峭度 ≈2（瑞利基线）

    def test_bandpass_isolates_band(self):
        fs = 4096.0
        t = np.arange(16384) / fs
        x = np.sin(2 * np.pi * 100 * t) + np.sin(2 * np.pi * 1500 * t)
        xb = bandpass_signal(x, fs, 1000, 2000)
        f, a = (lambda r: (np.fft.rfftfreq(xb.size, 1 / fs), np.abs(r)))(
            np.fft.rfft(xb))
        i1500 = np.argmin(abs(f - 1500))
        i100 = np.argmin(abs(f - 100))
        assert a[i1500] > 20 * a[i100]

    def test_envelope_kurtosis_broadband_noise_baseline(self):
        rng = np.random.default_rng(5)
        b = rng.standard_normal(32768)
        # 宽带噪声包络峭度基线 ≈2（未中心化 4 阶矩比）
        assert 1.5 < envelope_kurtosis(b) < 2.5

    def test_short_signal_raises(self):
        with pytest.raises(ValueError):
            kurtogram(np.random.default_rng(0).standard_normal(100), 1000.0)

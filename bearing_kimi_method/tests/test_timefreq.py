"""时频分析测试：STFT/CWT 输出形状与冲击时刻能量定位。"""

import numpy as np
import pytest

from bearing_fault import cwt_spectrogram, morlet_wavelet, stft_spectrogram

FS = 12000.0


class TestSTFT:
    def test_shapes(self):
        x = np.random.default_rng(0).standard_normal(int(FS * 1.0))
        t, f, z = stft_spectrogram(x, FS, nperseg=256)
        assert z.shape == (len(f), len(t))
        assert f[-1] == pytest.approx(FS / 2)

    def test_tone_localization(self):
        """500 Hz 正弦的 STFT 能量应集中在 500 Hz 行。"""
        t_ax = np.arange(int(FS * 1.0)) / FS
        x = np.sin(2 * np.pi * 500.0 * t_ax)
        t, f, z = stft_spectrogram(x, FS, nperseg=512)
        row_energy = z.mean(axis=1)
        assert f[np.argmax(row_energy)] == pytest.approx(500.0, abs=f[1] - f[0])


class TestCWT:
    def test_shapes(self):
        x = np.random.default_rng(0).standard_normal(int(FS * 0.5))
        freqs = np.array([100.0, 500.0, 1000.0, 4000.0])
        t, f_out, w = cwt_spectrogram(x, FS, freqs)
        assert w.shape == (4, x.size)
        assert np.all(f_out == freqs)
        assert t[-1] == pytest.approx((x.size - 1) / FS)

    def test_tone_ridge(self):
        """500 Hz 正弦的 CWT 脊线应在 500 Hz 刻度处。"""
        t_ax = np.arange(int(FS * 1.0)) / FS
        x = np.sin(2 * np.pi * 500.0 * t_ax)
        freqs = np.linspace(100, 1000, 46)
        _, _, w = cwt_spectrogram(x, FS, freqs)
        ridge = freqs[np.argmax(w.mean(axis=1))]
        assert abs(ridge - 500.0) / 500.0 < 0.05

    def test_impulse_time_localization(self):
        """冲击时刻的 CWT 高频能量应高于其他时刻。"""
        n = int(FS * 1.0)
        x = np.zeros(n)
        x[n // 2] = 1.0  # 单个冲击
        _, _, w = cwt_spectrogram(x, FS, np.array([4000.0]))
        energy = w[0] ** 2
        mid = n // 2
        assert energy[mid - 50 : mid + 50].mean() > 10 * energy[
            max(0, mid - 2000) : mid - 1000
        ].mean()

    def test_invalid_freqs_raise(self):
        with pytest.raises(ValueError):
            cwt_spectrogram(np.zeros(100), FS, np.array([0.0]))
        with pytest.raises(ValueError):
            cwt_spectrogram(np.zeros(100), FS, np.array([FS]))


class TestMorlet:
    def test_wavelet_properties(self):
        psi = morlet_wavelet(scale=10.0)
        assert np.iscomplexobj(psi)
        # 复 Morlet 近似零均值（截断支撑区引入的残差远小于峰值 π^{-1/4}≈0.75）
        assert abs(psi.mean()) < 1e-3

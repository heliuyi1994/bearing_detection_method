"""频谱分析单元测试：幅值谱标定、包络解调、倒频谱。"""

import numpy as np
import pytest

from bearing_diag.spectrum import (
    amplitude_spectrum,
    envelope,
    envelope_spectrum,
    find_spectrum_peaks,
    real_cepstrum,
)


class TestAmplitudeSpectrum:
    def test_single_tone_amplitude(self):
        fs, f0, a = 4096.0, 100.0, 3.0
        t = np.arange(8192) / fs
        x = a * np.sin(2 * np.pi * f0 * t)
        freqs, amps = amplitude_spectrum(x, fs)
        i = np.argmax(amps)
        assert freqs[i] == pytest.approx(f0, abs=1.0)
        assert amps[i] == pytest.approx(a, rel=0.05)  # 汉宁窗幅值校正

    def test_dc_not_required(self):
        # 幅值谱对直流分量不敏感（去均值由调用方负责，这里只验证有限性）
        fs = 1000.0
        t = np.arange(4096) / fs
        x = np.sin(2 * np.pi * 50 * t) + 10.0
        freqs, amps = amplitude_spectrum(x, fs)
        assert np.all(np.isfinite(amps)) and freqs[0] == 0.0


class TestEnvelope:
    def test_envelope_of_am_signal(self):
        fs = 8000.0
        t = np.arange(16000) / fs
        carrier = np.sin(2 * np.pi * 2000 * t)
        mod = 1.0 + 0.5 * np.cos(2 * np.pi * 30 * t)
        env = envelope(mod * carrier)
        assert np.max(env) == pytest.approx(1.5, abs=0.05)
        assert np.min(env) == pytest.approx(0.5, abs=0.05)

    def test_envelope_spectrum_reveals_modulation(self):
        fs = 12000.0
        t = np.arange(32768) / fs
        carrier = np.sin(2 * np.pi * 3000 * t)
        mod_freq = 107.0
        x = (1 + 0.8 * np.cos(2 * np.pi * mod_freq * t)) * carrier
        f, a = envelope_spectrum(x, fs, max_freq=1000)
        peaks = find_spectrum_peaks(f, a, n=3, min_prominence_ratio=0.3)
        assert peaks and peaks[0][0] == pytest.approx(mod_freq, abs=2.0)


class TestRealCepstrum:
    def test_echo_detection(self):
        # 倒频谱检测边带族：两频率间隔 20 Hz → 倒频峰在 1/20 = 0.05 s。
        # 谱需有连续底噪，否则双线谱的 log 出现 -inf 退化。
        fs = 4096.0
        t = np.arange(16384) / fs
        rng = np.random.default_rng(2)
        x = (np.sin(2 * np.pi * 500 * t) + 0.6 * np.sin(2 * np.pi * 520 * t)
             + 0.3 * rng.standard_normal(t.size))
        q, c = real_cepstrum(x, fs, max_quefrency=0.1)
        # 排除音调自身周期（1/500 s 级）的低倒频区，找边带族周期
        m = q > 0.02
        i = int(np.argmax(c[m]))
        assert q[m][i] == pytest.approx(0.05, abs=0.006)


class TestFindPeaks:
    def test_top_peaks_sorted(self):
        fs = 2048.0
        t = np.arange(8192) / fs
        x = 1.0 * np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)
        f, a = amplitude_spectrum(x, fs)
        peaks = find_spectrum_peaks(f, a, n=2, min_prominence_ratio=0.1)
        assert len(peaks) == 2
        assert {round(p[0]) for p in peaks} == {50, 120}
        assert peaks[0][1] >= peaks[1][1]

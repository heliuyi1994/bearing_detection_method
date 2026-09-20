"""包络分析测试：AM 信号解调、谱峰拾取、幅值谱幅值精度。"""

import numpy as np
import pytest

from bearing_fault import (
    amplitude_spectrum,
    analytic_envelope,
    envelope_analysis,
    find_peaks_near,
)

FS = 12000.0


class TestAmplitudeSpectrum:
    def test_sine_amplitude(self):
        """幅值谱应还原正弦分量的真实幅值（Hann 窗相干增益修正）。"""
        t = np.arange(int(FS * 2.0)) / FS
        x = 1.5 * np.sin(2 * np.pi * 1000.0 * t)
        freqs, amps = amplitude_spectrum(x, FS)
        i = int(np.argmin(np.abs(freqs - 1000.0)))
        assert amps[i] == pytest.approx(1.5, rel=0.01)
        # 其余谱线应远低于主峰
        assert amps.max() == pytest.approx(amps[i], rel=1e-9)

    def test_frequency_axis(self):
        freqs, amps = amplitude_spectrum(np.zeros(1000), FS)
        assert freqs[-1] == pytest.approx(FS / 2)
        assert len(freqs) == len(amps) == 501


class TestHilbertEnvelope:
    def test_am_signal_envelope(self):
        """调幅信号的包络应还原调制波形。"""
        t = np.arange(int(FS * 1.0)) / FS
        x = (1.0 + 0.8 * np.cos(2 * np.pi * 100.0 * t)) * np.sin(2 * np.pi * 4000.0 * t)
        env = analytic_envelope(x)
        # 去掉端部边缘效应后比对
        mid = slice(200, -200)
        expected = 1.0 + 0.8 * np.cos(2 * np.pi * 100.0 * t)
        assert np.corrcoef(env[mid], expected[mid])[0, 1] > 0.99

    def test_envelope_spectrum_peak_at_modulation(self):
        """AM 信号包络谱主峰应在调制频率 100 Hz 处。"""
        t = np.arange(int(FS * 2.0)) / FS
        x = (1.0 + 0.8 * np.cos(2 * np.pi * 100.0 * t)) * np.sin(2 * np.pi * 4000.0 * t)
        result = envelope_analysis(x, FS)
        band = (result["env_freqs"] > 20) & (result["env_freqs"] < 500)
        f_peak = result["env_freqs"][band][np.argmax(result["env_amps"][band])]
        assert abs(f_peak - 100.0) < 2.0


class TestFindPeaksNear:
    def test_hits_and_tolerance(self):
        freqs = np.arange(0, 1000, 0.5).astype(float)
        amps = np.random.default_rng(0).random(freqs.size) * 0.01
        amps[int(100.4 / 0.5)] = 1.0   # 100.4 Hz 处放主峰（目标 100 Hz，偏移 0.4%）
        amps[int(200.6 / 0.5)] = 0.6   # 二次谐波
        hits = find_peaks_near(freqs, amps, 100.0, harmonics=3, tol=0.02)
        harmonics_found = {h for h, _, _ in hits}
        assert 1 in harmonics_found and 2 in harmonics_found
        f1 = dict((h, f) for h, f, _ in hits)[1]
        assert f1 == pytest.approx(100.4, abs=0.5)

    def test_no_hit_outside_tolerance(self):
        freqs = np.arange(0, 500, 0.5).astype(float)
        amps = np.random.default_rng(1).random(freqs.size) * 0.01
        amps[int(110.0 / 0.5)] = 1.0  # 偏离目标 10%，超出 ±2% 容差
        hits = find_peaks_near(freqs, amps, 100.0, harmonics=1, tol=0.02)
        assert hits[0][2] < 0.05  # 窗口内只有噪声底

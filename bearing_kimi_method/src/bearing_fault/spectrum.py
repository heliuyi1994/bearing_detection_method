"""频谱分析与谱峰拾取。

幅值谱采用相干增益修正：加窗 w(n) 后，正弦分量的谱峰幅值为
|Σ x(n)w(n)e^{-jωn}| · 2 / Σ w(n)，从而谱图纵轴可直接读作物理幅值。
默认使用 Hann 窗抑制滑差导致的准周期性泄漏。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def amplitude_spectrum(
    x: NDArray[np.float64],
    fs: float,
    window: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """单边幅值谱，返回 (频率轴 Hz, 幅值)。"""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if window:
        w = np.hanning(n)
        gain = w.sum()
    else:
        w = np.ones(n)
        gain = float(n)
    spec = np.abs(np.fft.rfft(x * w)) * 2.0 / gain
    spec[0] /= 2.0  # 直流分量不翻倍
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    return freqs, spec


def power_spectrum(
    x: NDArray[np.float64],
    fs: float,
    window: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """单边功率谱（幅值谱平方的一半，保持正弦功率守恒）。"""
    freqs, amp = amplitude_spectrum(x, fs, window=window)
    return freqs, amp**2 / 2.0


def envelope_spectrum(
    envelope: NDArray[np.float64],
    fs: float,
    window: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """包络信号的幅值谱（先去直流，突出故障重复频率及其谐波）。"""
    env = np.asarray(envelope, dtype=np.float64) - np.mean(envelope)
    return amplitude_spectrum(env, fs, window=window)


def find_peaks_near(
    freqs: NDArray[np.float64],
    amps: NDArray[np.float64],
    target: float,
    harmonics: int = 3,
    tol: float = 0.015,
) -> list[tuple[int, float, float]]:
    """在目标频率的 1..harmonics 次谐波附近 ±tol 相对容差内拾取最强谱峰。

    返回 [(谐波次数, 峰值频率, 峰值幅值), ...]，找不到谐波的跳过。
    容差用于吸收滚动体滑差（1–2%）造成的实际频率偏移。
    """
    freqs = np.asarray(freqs)
    amps = np.asarray(amps)
    found: list[tuple[int, float, float]] = []
    for h in range(1, harmonics + 1):
        f = h * target
        if f > freqs[-1]:
            break
        mask = (freqs >= f * (1.0 - tol)) & (freqs <= f * (1.0 + tol))
        if not np.any(mask):
            continue
        idx_local = int(np.argmax(amps[mask]))
        f_peak = float(freqs[mask][idx_local])
        a_peak = float(amps[mask][idx_local])
        found.append((h, f_peak, a_peak))
    return found

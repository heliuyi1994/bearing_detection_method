"""时频分析：短时傅里叶变换（STFT）与连续小波变换（CWT）。

故障冲击是瞬态非平稳成分，单张全局频谱无法体现"何时出现"。
STFT 以固定窗滑动提供均匀时频网格；CWT 用尺度伸缩的 Morlet 小波，
低频长时间窗、高频短时间窗，更适合定位瞬态冲击。

新版 SciPy（≥1.15）已移除 scipy.signal.cwt，此处基于 FFT 快速卷积
自行实现复 Morlet CWT：

    ψ(t) = π^{−1/4} · exp(jω0·t) · exp(−t²/2)
    W(f, t) = x ⋆ conj(ψ_s(−·)) / √s,   s = ω0·fs/(2πf)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import signal as sps


def stft_spectrogram(
    x: NDArray[np.float64],
    fs: float,
    nperseg: int = 256,
    noverlap: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """STFT 时频谱，返回 (时间轴 s, 频率轴 Hz, 幅值 |STFT|)。"""
    if noverlap is None:
        noverlap = nperseg // 2
    f, t, z = sps.stft(x, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap)
    return t, f, np.abs(z)


def morlet_wavelet(scale: float, w0: float = 6.0) -> NDArray[np.complex128]:
    """复 Morlet 小波，scale 以采样点为单位，支撑区取 ±4·scale。"""
    half = max(4, int(4.0 * scale))
    t = np.arange(-half, half + 1) / scale
    return np.pi ** -0.25 * np.exp(1j * w0 * t) * np.exp(-0.5 * t**2)


def cwt_spectrogram(
    x: NDArray[np.float64],
    fs: float,
    freqs: NDArray[np.float64],
    w0: float = 6.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """复 Morlet 连续小波变换（FFT 卷积实现）。

    参数
    ----
    x : 输入信号。
    fs : 采样频率 (Hz)。
    freqs : 关注的频率刻度数组 (Hz)，对应尺度 s = ω0·fs/(2πf)。
    w0 : Morlet 小波中心角频率，ω0=6 是时频分辨率的常用折中。

    返回 (时间轴 s, 频率轴 Hz, 幅值 |W|)。
    """
    x = np.asarray(x, dtype=np.float64)
    freqs = np.asarray(freqs, dtype=np.float64)
    if np.any(freqs <= 0) or np.any(freqs >= fs / 2):
        raise ValueError("freqs 需满足 0 < f < fs/2")
    out = np.empty((freqs.size, x.size), dtype=np.complex128)
    for i, f in enumerate(freqs):
        scale = w0 * fs / (2.0 * np.pi * f)
        psi = morlet_wavelet(scale, w0)
        out[i] = sps.fftconvolve(x, np.conj(psi[::-1]), mode="same") / np.sqrt(scale)
    t = np.arange(x.size) / fs
    return t, freqs, np.abs(out)

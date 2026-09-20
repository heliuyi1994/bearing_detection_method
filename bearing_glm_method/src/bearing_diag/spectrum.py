"""频谱分析：FFT 幅值谱、Hilbert 包络谱、实倒频谱。

原理见 docs/principles.md 第 5、6、8 章。
"""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert


def amplitude_spectrum(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """单边幅值谱。

    返回 (freqs, amps)：频率轴 (Hz) 与各频率分量幅值（与信号同量纲）。
    加汉宁窗抑制泄漏，幅值按窗增益 2/sum(window) 校正。
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n < 2:
        raise ValueError("信号过短")
    win = np.hanning(n)
    spec = np.fft.rfft(x * win)
    amps = np.abs(spec) * (2.0 / win.sum())
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    return freqs, amps


def envelope(x: np.ndarray) -> np.ndarray:
    """Hilbert 包络：|analytic(x)|。调用方应先做带通滤波。"""
    x = np.asarray(x, dtype=float).ravel()
    return np.abs(hilbert(x))


def envelope_spectrum(
    x: np.ndarray,
    fs: float,
    max_freq: float | None = None,
    lowpass: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """包络解调谱（共振解调法）。

    流程：Hilbert 取包络 → 去均值 →（可选）低通去高频噪声 → FFT 幅值谱。
    包络谱的峰对应调制频率，即轴承故障特征频率及其谐波。

    max_freq: 只返回该频率以下的谱（默认 1000 Hz，特征频率远低于共振频带）。
    """
    env = envelope(x)
    env = env - np.mean(env)
    freqs, amps = amplitude_spectrum(env, fs)
    if max_freq is None:
        max_freq = 1000.0
    mask = freqs <= max_freq
    return freqs[mask], amps[mask]


def real_cepstrum(x: np.ndarray, fs: float, max_quefrency: float | None = None):
    """实倒频谱 c(q) = |IFFT(log(|FFT(x)|))|。

    倒频谱把频谱上间隔为 f_c 的边带族压缩为 quefrency = 1/f_c 处的单峰，
    常用于确认转频/故障频率调制的边带族（文档第 8 章）。

    返回 (quefrency, cepstrum)。max_quefrency 单位秒，默认 0.2 s（即检测 5 Hz 以上周期）。
    """
    x = np.asarray(x, dtype=float).ravel()
    spec = np.abs(np.fft.fft(x))
    spec = np.where(spec > 0, spec, np.finfo(float).eps)
    cep = np.abs(np.fft.ifft(np.log(spec)))
    q = np.arange(x.size) / fs
    if max_quefrency is None:
        max_quefrency = 0.2
    mask = (q > 0) & (q <= max_quefrency)
    return q[mask], cep[mask]


def find_spectrum_peaks(
    freqs: np.ndarray,
    amps: np.ndarray,
    n: int | None = None,
    min_prominence_ratio: float = 0.1,
) -> list[tuple[float, float]]:
    """找谱峰，返回 [(频率, 幅值)]，按幅值降序。

    min_prominence_ratio: 峰的最小突出度，按谱最大幅值的比例计。
    """
    from scipy.signal import find_peaks

    if freqs.size < 3:
        return []
    prom = max(min_prominence_ratio * np.max(amps), 1e-12)
    idx, _ = find_peaks(amps, prominence=prom)
    peaks = sorted(
        ((float(freqs[i]), float(amps[i])) for i in idx),
        key=lambda t: t[1],
        reverse=True,
    )
    if n is not None:
        peaks = peaks[:n]
    return peaks

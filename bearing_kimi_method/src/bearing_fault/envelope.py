"""包络分析（共振解调 / 包络解调）——轴承局部故障诊断的核心方法。

流程
----
1. 带通滤波：锁定结构共振频带，滤除低频的转频及其谐波、宽带噪声；
2. Hilbert 变换构造解析信号 z(t) = x(t) + j·H[x(t)]；
3. 取模得包络 a(t) = |z(t)|，即共振响应的幅值调制（冲击重复）信息；
4. 包络去直流后做 FFT 得包络谱，在谱上寻找故障特征频率。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import signal as sps

from .preprocessing import bandpass
from .spectrum import envelope_spectrum


def analytic_envelope(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Hilbert 变换求包络：解析信号 z(t) = x + j·H[x]，包络 = |z|。"""
    return np.abs(sps.hilbert(x))


def auto_resonance_band(fs: float) -> tuple[float, float]:
    """缺省共振解调频带：(fs/6, 0.45·fs)。

    仿真信号共振频率默认 4 kHz（fs=12 kHz 时恰在带内）；真实数据应结合
    谱峭度或敲击试验选择频带，此处给出对仿真信号稳健的工程缺省值。
    """
    return fs / 6.0, 0.45 * fs


def envelope_analysis(
    x: NDArray[np.float64],
    fs: float,
    band: tuple[float, float] | None = None,
    filter_order: int = 4,
) -> dict[str, NDArray[np.float64]]:
    """完整共振解调流程。

    返回字典：filtered（带通后信号）、envelope（包络）、
    env_freqs / env_amps（包络谱频率轴与幅值）。
    """
    if band is None:
        band = auto_resonance_band(fs)
    filtered = bandpass(x, fs, band[0], band[1], order=filter_order)
    env = analytic_envelope(filtered)
    env_freqs, env_amps = envelope_spectrum(env, fs)
    return {
        "filtered": filtered,
        "envelope": env,
        "env_freqs": env_freqs,
        "env_amps": env_amps,
    }

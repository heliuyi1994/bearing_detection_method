# -*- coding: utf-8 -*-
"""信号预处理：去均值、去趋势、零相位带通（移植 kimi preprocessing）。

零相位前向-反向滤波（sosfiltfilt）相位零畸变、冲击时刻不偏移——这对
包络分析的时域定位与梳检的峰位一致性至关重要；统一包用它替代 glm 原生
的非零相位 sosfilt 解调带通（合并方案移植项 2）。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import signal as sps


def remove_dc(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """去除直流分量（零均值化）。"""
    return x - np.mean(x)


def detrend(x: NDArray[np.float64], type: str = "linear") -> NDArray[np.float64]:
    """去除趋势项，``type="linear"`` 去线性趋势，``"constant"`` 去均值。"""
    return sps.detrend(x, type=type)


def bandpass(
    x: NDArray[np.float64],
    fs: float,
    low: float,
    high: float,
    order: int = 4,
) -> NDArray[np.float64]:
    """Butterworth 零相位带通滤波。

    使用 sosfiltfilt 前向-反向滤波，相位零畸变，冲击时刻不偏移——
    这对后续包络分析的时域定位至关重要。幅值响应为 |H|²，通带内近似无损。
    """
    nyq = fs / 2.0
    if not (0.0 < low < high < nyq):
        raise ValueError(f"需要 0 < low < high < fs/2，得到 low={low}, high={high}, fs/2={nyq}")
    sos = sps.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return sps.sosfiltfilt(sos, x)


def clamp_band(band: tuple[float, float], fs: float) -> tuple[float, float]:
    """把任意候选频带夹取到 bandpass 的合法域 0 < low < high < fs/2。

    Kurtogram 可能选到贴奈奎斯特或贴 0 的频带（如 (0,500)、(7500,8000)@16k），
    统一夹取后再做零相位带通，避免滤波器设计非法。
    """
    nyq = fs / 2.0
    hi = min(float(band[1]), nyq - max(10.0, 0.001 * fs))
    lo = min(float(band[0]), hi * 0.9)
    lo = max(lo, 1.0)
    return (lo, hi)

"""谱峭度与 Kurtogram：自适应选择包络解调的带通频带。

原理（docs/principles.md 第 7 章）：谱峭度 SK(f) 度量每个频率分量幅值的非高斯性，
受瞬态冲击激发的共振频带 SK 高，平稳噪声频带 SK≈常量。Kurtogram 在二进制/三分之一
倍频程树状频带划分中搜索"带内包络峭度"最大的频带，作为包络解调前的带通中心。

实现说明：采用 Antoni 快速 Kurtogram 的树状频带结构（每层含 2^level 个二进制频带
与错位 1/3 倍频程频带），但以 butter 带通 + Hilbert 包络峭度直接计算各频带指标
（等价于该带的积分谱峭度），实现简洁且数值稳定（sos 级联型滤波器）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, sosfilt

_MIN_BAND_BINS = 4  # 频带至少要覆盖若干谱线，否则滤波无意义


@dataclass
class KurtogramResult:
    """Kurtogram 搜索结果。"""

    best_band: tuple[float, float]     # (low, high) Hz，峭度最大的解调频带
    best_kurtosis: float               # 该频带的包络峭度
    fs: float
    levels: list[list[tuple[float, float, float]]] = field(default_factory=list)
    # levels[j] = 第 j 层各频带 [(low, high, envelope_kurtosis), ...]，供报告绘图

    @property
    def center(self) -> float:
        return 0.5 * (self.best_band[0] + self.best_band[1])


def envelope_kurtosis(x_band: np.ndarray) -> float:
    """带通信号的 Hilbert 包络峭度（未中心化 4 阶矩比；窄带高斯噪声基线≈2）。

    hilbert 为 FFT 圆卷积实现，信号两端存在回绕伪影（末端包络虚高），
    故裁掉两端各 1% 样本再统计。
    """
    from .spectrum import envelope

    xb = np.asarray(x_band, dtype=float).ravel()
    n = xb.size
    cut = max(1, n // 100)
    env = envelope(xb[cut : n - cut])
    m2 = np.mean(env**2)
    if m2 <= 0 or not np.isfinite(m2):
        return 0.0
    k = float(np.mean(env**4) / m2**2)
    return k if np.isfinite(k) else 0.0


def _bandpass(x: np.ndarray, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
    """数值稳定带通（sos 级联实现）。low/high 单位 Hz，自动夹到 (0, fs/2) 内。"""
    nyq = fs / 2.0
    lo = min(max(low, 1e-3), nyq * 0.999)
    hi = min(max(high, lo + 1e-3), nyq * 0.999)
    sos = butter(order, (lo, hi), btype="bandpass", output="sos", fs=fs)
    return sosfilt(sos, x)


def kurtogram(x: np.ndarray, fs: float, max_level: int = 4) -> KurtogramResult:
    """层级树搜索峭度最大频带。

    max_level: 分解层数。level j 将 [0, fs/2] 分为 2^j 个二进制频带，
    并附加同宽 2/3 的错位 1/3 倍频程频带。默认 4 层，最窄频带 fs/32。
    """
    x = np.asarray(x, dtype=float).ravel()
    if x.size < 256:
        raise ValueError("信号过短，无法做 Kurtogram 分析（建议 ≥ 256 点）")
    nyq = fs / 2.0
    freq_res = fs / x.size  # 频率分辨率
    min_width = _MIN_BAND_BINS * freq_res

    best = (None, -np.inf)
    levels: list[list[tuple[float, float, float]]] = []
    for level in range(1, max_level + 1):
        width = nyq / 2**level
        bands: list[tuple[float, float]] = []
        # 二进制频带
        for i in range(2**level):
            bands.append((i * width, (i + 1) * width))
        # 错位 1/3 倍频程频带（宽度 2/3·width，起点步进 width/3）
        w3 = width * 2.0 / 3.0
        start = 0.0
        while start + w3 <= nyq + 1e-9:
            bands.append((start, min(start + w3, nyq)))
            start += width / 3.0

        row: list[tuple[float, float, float]] = []
        for low, high in bands:
            if high - low < min_width:
                continue  # 频带窄于 4 根谱线，跳过（通常是最深层低频带）
            xb = _bandpass(x, fs, low, high)
            # 剔除滤波器起始瞬态（时间常数 ~1/带宽），其振铃会虚增峭度
            n_trim = min(xb.size // 4, int(8.0 * fs / (high - low)))
            k = envelope_kurtosis(xb[n_trim:])
            row.append((low, high, k))
            if k > best[1]:
                best = ((low, high), k)
        levels.append(row)

    if best[0] is None:
        # 极端情况（信号太短），回退为全带
        best = ((0.0, nyq), envelope_kurtosis(x))
    return KurtogramResult(
        best_band=best[0], best_kurtosis=best[1], fs=fs, levels=levels
    )


def bandpass_signal(x: np.ndarray, fs: float, low: float, high: float) -> np.ndarray:
    """按指定频带带通滤波（供诊断流水线在选定解调频带后使用）。"""
    return _bandpass(np.asarray(x, dtype=float), fs, low, high)

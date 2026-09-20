"""阶次分析：变转速工况下的角域重采样与阶次谱。

恒速时故障冲击近似等时间间隔，FFT 频谱上呈现清晰谱线；变速时冲击
等**角度**出现，等时间采样信号的频谱发生"频率涂抹"。阶次分析按轴相位
θ(t) 把信号重采样到等角度域，再做 FFT 得到阶次谱（横轴 = 每转次数），
故障阶次 = 特征频率/转频，是与转速无关的常数。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def angular_resample(
    x: NDArray[np.float64],
    theta: NDArray[np.float64],
    samples_per_rev: int = 512,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """把等时间采样信号按轴相位重采样为等角度信号。

    参数
    ----
    x : 等时间采样信号。
    theta : 与 x 等长的轴相位（弧度，单调递增，可由转速积分获得）。
    samples_per_rev : 每转采样点数，决定阶次奈奎斯特上限（其一半）。

    返回 (等角度轴 θ_u, 重采样信号 x_u)。
    """
    x = np.asarray(x, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    if np.any(np.diff(theta) <= 0):
        raise ValueError("轴相位 theta 必须严格单调递增")
    n_rev = (theta[-1] - theta[0]) / (2.0 * np.pi)
    n_out = int(n_rev * samples_per_rev)
    if n_out < 8:
        raise ValueError("信号圈数太少，无法重采样")
    theta_u = theta[0] + np.arange(n_out) * (2.0 * np.pi / samples_per_rev)
    x_u = np.interp(theta_u, theta, x)
    return theta_u, x_u


def order_spectrum(
    x: NDArray[np.float64],
    theta: NDArray[np.float64],
    samples_per_rev: int = 512,
    window: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """阶次幅值谱：角域重采样后做 FFT，横轴单位为阶（每转次数）。"""
    _, x_u = angular_resample(x, theta, samples_per_rev)
    x_u = x_u - np.mean(x_u)
    n = x_u.size
    if window:
        w = np.hanning(n)
        gain = w.sum()
    else:
        w = np.ones(n)
        gain = float(n)
    spec = np.abs(np.fft.rfft(x_u * w)) * 2.0 / gain
    spec[0] /= 2.0
    # 采样"频率"为 samples_per_rev 点/转，频率轴即阶次轴
    orders = np.fft.rfftfreq(n, d=1.0 / samples_per_rev)
    return orders, spec

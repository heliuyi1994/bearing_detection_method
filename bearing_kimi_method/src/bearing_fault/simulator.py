"""轴承故障振动信号仿真器（Randall 冲击响应模型）。

物理模型
--------
滚动轴承局部缺陷产生的振动信号可建模为周期性准周期冲击串激发结构
共振后的响应叠加噪声：

    x(t) = Σ_k A_k · h(t − t_k) + n(t)

- t_k：第 k 次冲击发生时刻，名义间隔为故障特征周期的倒数，逐冲击叠加
  小幅随机抖动模拟滚动体滑差（slip，典型 1–2%）；
- h(t)：结构共振的单自由度衰减振荡响应
  h(t) = exp(−2πζf_n t)·sin(2πf_n t)，f_n 为共振频率，ζ 为阻尼比；
- A_k：冲击幅值。内圈故障随轴转频调制（缺陷随轴旋转进出载荷区），
  滚动体故障随保持架频率（FTF）调制，外圈故障位置固定、无调制；
- n(t)：高斯白噪声，按目标信噪比定量加入。

变速工况下冲击在**等角度**上出现而非等时间，因此仿真器以轴相位 θ(t)
为主变量生成冲击，同时输出 θ(t) 供阶次分析做角域重采样。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .bearing import Bearing

FAULT_TYPES = ("healthy", "inner", "outer", "ball", "cage")


class BearingSimulator:
    """轴承故障振动信号仿真器。

    参数
    ----
    bearing : Bearing, optional
        轴承几何参数，默认 SKF 6205（CWRU 数据集轴承）。
    fs : float
        采样频率 (Hz)。
    duration : float
        信号时长 (s)。
    rpm : float
        轴转速 (rpm)。``speed_ramp`` 生效时作为起点之外的参考。
    resonance_freq : float
        结构共振频率 f_n (Hz)。
    damping : float
        共振阻尼比 ζ。
    slip : float
        滚动体滑差，冲击间隔的相对随机抖动标准差（0.01 ≈ 1%）。
    modulation_depth : float
        幅值调制深度（0–1）。
    seed : int, optional
        随机种子，固定后可复现同一信号。
    """

    def __init__(
        self,
        bearing: Bearing | None = None,
        fs: float = 12000.0,
        duration: float = 5.0,
        rpm: float = 1797.0,
        resonance_freq: float = 4000.0,
        damping: float = 0.05,
        slip: float = 0.01,
        modulation_depth: float = 0.6,
        seed: int | None = None,
    ) -> None:
        if resonance_freq >= fs / 2:
            raise ValueError("resonance_freq 必须小于奈奎斯特频率 fs/2")
        self.bearing = bearing or Bearing.sk6205()
        self.fs = float(fs)
        self.duration = float(duration)
        self.rpm = float(rpm)
        self.resonance_freq = float(resonance_freq)
        self.damping = float(damping)
        self.slip = float(slip)
        self.modulation_depth = float(modulation_depth)
        self.seed = seed

    def signal(
        self,
        fault: str = "healthy",
        snr_db: float | None = -6.0,
        speed_ramp: tuple[float, float] | None = None,
        return_components: bool = False,
    ) -> NDArray[np.float64] | dict[str, NDArray[np.float64]]:
        """生成一段振动信号。

        参数
        ----
        fault : str
            故障类型：``"healthy" / "inner" / "outer" / "ball" / "cage"``。
        snr_db : float or None
            信噪比 (dB)，定义为 10·log10(P_signal/P_noise)；None 表示不加噪声。
            健康信号无故障分量，退化为标准差 0.1 的低幅值白噪声。
        speed_ramp : (float, float), optional
            (起始 rpm, 结束 rpm)，给出时模拟匀变速工况；否则恒速 ``self.rpm``。
        return_components : bool
            True 时返回字典，含 signal/clean/noise/t/theta/fr_t/impact_times，
            便于教学展示与阶次分析；False 时只返回合成信号数组。
        """
        fault = fault.lower()
        if fault not in FAULT_TYPES:
            raise ValueError(f"未知故障类型 {fault!r}，可选 {FAULT_TYPES}")

        rng = np.random.default_rng(self.seed)
        n = int(round(self.fs * self.duration))
        t = np.arange(n) / self.fs

        # 轴瞬时转频 fr(t) 与轴相位 θ(t)（θ 单调递增，是可逆的）
        if speed_ramp is None:
            fr_t = np.full(n, self.rpm / 60.0)
            theta = 2.0 * np.pi * fr_t[0] * t
        else:
            fr0, fr1 = speed_ramp[0] / 60.0, speed_ramp[1] / 60.0
            rate = (fr1 - fr0) / self.duration
            fr_t = fr0 + rate * t
            theta = 2.0 * np.pi * (fr0 * t + 0.5 * rate * t**2)

        clean = np.zeros(n)
        impact_times = np.empty(0)
        if fault != "healthy":
            impact_times, impact_amps = self._impacts(fault, t, theta, fr_t, rng)
            spikes = np.zeros(n)
            idx = np.clip((impact_times * self.fs).astype(int), 0, n - 1)
            np.add.at(spikes, idx, impact_amps)
            clean = np.convolve(spikes, self._impulse_response())[:n]

        noise = self._make_noise(clean, snr_db, n, rng)
        x = clean + noise

        if not return_components:
            return x
        return {
            "signal": x,
            "clean": clean,
            "noise": noise,
            "t": t,
            "theta": theta,
            "fr_t": fr_t,
            "impact_times": impact_times,
        }

    # ------------------------------------------------------------------
    def _impacts(
        self,
        fault: str,
        t: NDArray[np.float64],
        theta: NDArray[np.float64],
        fr_t: NDArray[np.float64],
        rng: np.random.Generator,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """在轴相位域生成冲击时刻与幅值。

        冲击等角度间隔 Δθ = 2π/order，order 为故障特征频率与转频之比
        （常数，与转速无关），因此同一套逻辑同时覆盖恒速与变速。
        """
        fr0 = fr_t[0]
        fault_freq = self.bearing.characteristic_frequencies(fr0)[fault]
        order = fault_freq / fr0
        d_theta = 2.0 * np.pi / order

        count = int((theta[-1] - theta[0]) / d_theta) + 2
        k = np.arange(count)
        phase0 = rng.uniform(0.0, d_theta)
        jitter = rng.normal(0.0, self.slip, size=count) * d_theta
        imp_theta = theta[0] + phase0 + k * d_theta + jitter
        imp_theta = imp_theta[(imp_theta >= theta[0]) & (imp_theta <= theta[-1])]

        imp_t = np.interp(imp_theta, theta, t)

        # 幅值调制：内圈随轴每转调制一次，滚动体随保持架旋转调制
        if fault == "inner":
            amps = 1.0 + self.modulation_depth * np.cos(imp_theta)
        elif fault == "ball":
            ftf_order = 0.5 * (1.0 - self.bearing.geometry_ratio)
            amps = 1.0 + self.modulation_depth * np.cos(imp_theta * ftf_order)
        else:
            amps = np.ones_like(imp_t)
        return imp_t, amps

    def _impulse_response(self) -> NDArray[np.float64]:
        """单自由度欠阻尼结构的单位冲击响应，长度取约 6 个时间常数。"""
        tau = 1.0 / (2.0 * np.pi * self.damping * self.resonance_freq)
        n_h = max(8, int(6.0 * tau * self.fs))
        th = np.arange(n_h) / self.fs
        return np.exp(-th / tau) * np.sin(2.0 * np.pi * self.resonance_freq * th)

    @staticmethod
    def _make_noise(
        clean: NDArray[np.float64],
        snr_db: float | None,
        n: int,
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        if snr_db is None:
            return np.zeros(n)
        p_sig = float(np.mean(clean**2))
        if p_sig == 0.0:
            # 健康信号：无故障分量，只保留低幅值背景噪声
            return 0.1 * rng.standard_normal(n)
        p_noise = p_sig / (10.0 ** (snr_db / 10.0))
        return np.sqrt(p_noise) * rng.standard_normal(n)

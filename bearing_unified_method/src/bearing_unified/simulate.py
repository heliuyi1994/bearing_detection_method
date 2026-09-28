"""轴承故障振动信号仿真：共振衰减冲击模型。

模型（docs/principles.md 第 2、3 章）：

    局部缺陷使滚动体每次经过缺陷处产生一次冲击，激发轴承座/传感器的
    固有共振 f_r 并按阻尼指数衰减：

        s_i(t) = A_i · exp(-2πζf_r (t - t_i)) · sin(2π f_r (t - t_i)),  t ≥ t_i

    冲击时刻 t_i 以故障特征频率 f_c 到达，并带 1~2% 随机滑移。
    不同故障部位的调制差异：

    - BPFO 外圈：外圈相对载荷区固定 → 冲击等幅
    - BPFI 内圈：缺陷随轴旋转进出载荷区 → A_i 按 1×fr（轴频）调制
    - BSF  滚动体：缺陷随保持架公转 → A_i 按 FTF 调制
    - FTF  保持架：保持架碰擦 → 低幅、低频冲击，滑移略大

    背景叠加轻度不平衡（1×fr 正弦）与高斯噪声（按 SNR 定标）。
"""

from __future__ import annotations

import numpy as np

from .fault_freqs import Bearing, BUILTIN_BEARINGS

FAULT_CHOICES = ("normal", "BPFO", "BPFI", "BSF", "FTF")


def simulate(
    fault: str = "BPFO",
    fs: float = 12000.0,
    duration: float = 1.0,
    rpm: float = 1797.0,
    bearing: Bearing | str = "SKF6205",
    snr_db: float = 5.0,
    jitter: float = 0.01,
    resonance_hz: float = 3000.0,
    damping: float = 0.08,
    seed: int | None = None,
) -> tuple[np.ndarray, dict]:
    """生成仿真振动信号。

    返回 (signal, info)。info 含真实特征频率、轴频等，供测试与演示核对。
    snr_db 为冲击成分对噪声的信噪比；jitter 为冲击间隔随机滑移比例。
    """
    if fault not in FAULT_CHOICES:
        raise ValueError(f"fault 须为 {FAULT_CHOICES} 之一，得到 {fault!r}")
    if isinstance(bearing, str):
        from .fault_freqs import get_bearing

        bearing = get_bearing(bearing)

    rng = np.random.default_rng(seed)
    n = int(round(fs * duration))
    t = np.arange(n) / fs
    fr = rpm / 60.0
    freqs = bearing.fault_frequencies(fr)
    ftf = freqs["FTF"]

    fault_power = 0.0
    sig = np.zeros(n)

    if fault == "normal":
        fault_freq = 0.0
    else:
        fault_freq = freqs[fault]

    if fault_freq > 0:
        # 冲击时刻：名义间隔 1/f_fault + 随机滑移（保持架滑移更大）
        j = jitter * (1.5 if fault == "FTF" else 1.0)
        impacts: list[float] = []
        t_next = rng.uniform(0, 1.0 / fault_freq)
        while t_next < duration:
            impacts.append(t_next)
            t_next += (1.0 / fault_freq) * (1.0 + j * rng.standard_normal())
        impacts_t = np.asarray(impacts)

        # 幅值调制（按故障部位；先建立等幅数组再按部位叠加调制）
        amp = np.ones_like(impacts_t)
        if fault == "BPFI":
            amp = amp * (1.0 + 0.7 * np.cos(2 * np.pi * fr * impacts_t))
        elif fault == "BSF":
            amp = amp * (1.0 + 0.7 * np.cos(2 * np.pi * ftf * impacts_t))
        elif fault == "FTF":
            amp = amp * 0.4  # 保持架碰擦冲击弱

        # 共振衰减冲击核
        decay = 2 * np.pi * damping * resonance_hz
        for t_i, a_i in zip(impacts_t, amp):
            dt = t - t_i
            m = dt >= 0
            # 只保留冲击后 5 倍时间常数内的样本，避免无效计算
            keep = m & (dt < 5.0 / decay)
            if not np.any(keep):
                continue
            d = dt[keep]
            sig[keep] += a_i * np.exp(-decay * d) * np.sin(2 * np.pi * resonance_hz * d)

        fault_power = float(np.mean(sig**2))

    # 背景一阶不平衡分量（约冲击能量的 -20 dB，正常信号则为主要成分）
    imbalance = 0.1 * np.sqrt(fault_power) * np.sin(2 * np.pi * fr * t) if fault_power > 0 \
        else 0.05 * np.sin(2 * np.pi * fr * t)
    sig = sig + imbalance

    # 高斯噪声按 SNR 定标（相对冲击成分；normal 信号则给出固定单位噪声）
    if fault_power > 0:
        noise_pow = fault_power / (10 ** (snr_db / 10.0))
    else:
        noise_pow = 1.0
    sig = sig + rng.standard_normal(n) * np.sqrt(noise_pow)

    info = {
        "fault": fault,
        "fs": fs,
        "duration": duration,
        "rpm": rpm,
        "shaft_freq": fr,
        "bearing": bearing.name,
        "snr_db": snr_db,
        "true_fault_freq": fault_freq,
        "true_fault_freqs": freqs,
        "resonance_hz": resonance_hz,
        "seed": seed,
    }
    return sig, info

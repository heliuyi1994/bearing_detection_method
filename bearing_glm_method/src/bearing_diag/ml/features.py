"""特征工程：一段振动信号 → 固定长度特征向量。

五组特征（docs/principles.md 第 11 章）：

    T1 时域指标      9 维   复用 indicators（有量纲项 log 变换）
    T2 物理判别特征  20 维  四候选特征频率 × 5 次谐波的包络谱局部突出度
                           （与 matcher 同源；依赖转速+轴承参数，判别力核心）
    T3 谱峭度特征    7 维   Kurtogram 最佳峭度/频带位置宽度 + 各层峭度统计
    T4 频谱形态      11 维  8 分频带能量占比、谱重心、谱熵、谱平坦度
    T5 包络谱形态    2 维   低频段谱熵、主峰局部突出度（均不依赖轴承参数）

设计要点：
- T2/T3 与传统诊断方法同源——模型学到的判据可被物理解释与复核；
- 突出度、能量占比、熵等均为**尺度不变**特征，模型对增益/载荷变化稳健；
- use_physical=False 时剔除 T2（消融路线 B：纯统计盲特征，无需转速信息）。
"""

from __future__ import annotations

import numpy as np

from ..fault_freqs import Bearing, FAULT_TYPES
from ..indicators import detrend, time_domain_indicators
from ..kurtogram import kurtogram
from ..spectrum import amplitude_spectrum, envelope_spectrum

N_HARMONICS_FEAT = 5
N_SPEC_BANDS = 8
KURTOGRAM_LEVEL = 3  # 特征用途 3 层足够（最窄带 fs/16），比诊断用 4 层快一倍

_DIMENSIONAL_KEYS = ("rms", "peak", "peak_to_peak")  # 有量纲 → log 变换


def _harmonic_prominences(
    env_freqs: np.ndarray,
    env_amps: np.ndarray,
    fault_freqs: dict[str, float],
) -> list[float]:
    """四候选 × 5 次谐波的局部突出度（与 matcher 相同的窗口/基线规则，取值不判命中）。"""
    freqs, amps = env_freqs, env_amps
    df = float(freqs[1] - freqs[0])
    vals: list[float] = []
    for fault in FAULT_TYPES:
        fc = fault_freqs.get(fault)
        if fc is None:
            vals.extend([0.0] * N_HARMONICS_FEAT)
            continue
        for k in range(1, N_HARMONICS_FEAT + 1):
            target = k * fc
            half_win = max(0.02 * target, 1.5 * df)
            m = (freqs >= target - half_win) & (freqs <= target + half_win)
            half_local = max(0.16 * target, 10.0 * df)
            ml = (freqs >= target - half_local) & (freqs <= target + half_local)
            if np.any(m) and np.count_nonzero(ml) >= 5:
                amp = float(np.max(amps[m]))
                local_med = max(float(np.median(amps[ml])), np.finfo(float).eps)
                vals.append(min(amp / local_med, 100.0))  # 封顶防极端值
            else:
                vals.append(0.0)
    return vals


def _spectrum_shape_feats(freqs: np.ndarray, amps: np.ndarray) -> list[float]:
    """T4：频带能量占比 / 谱重心 / 谱熵 / 谱平坦度。"""
    nyq = float(freqs[-1]) if freqs[-1] > 0 else 1.0
    p = amps**2
    total = float(np.sum(p))
    out: list[float] = []
    if total <= 0:
        return [0.0] * (N_SPEC_BANDS + 3)
    # 分频带能量占比
    edges = np.linspace(0, nyq, N_SPEC_BANDS + 1)
    for i in range(N_SPEC_BANDS):
        m = (freqs >= edges[i]) & (freqs < edges[i + 1])
        out.append(float(np.sum(p[m]) / total))
    # 谱重心（归一化）
    centroid = float(np.sum(freqs * p) / total / nyq)
    # 谱熵（幅值归一化后按离散分布计，再除以 log(n) 归一）
    pn = p / total
    pn = pn[pn > 0]
    entropy = float(-np.sum(pn * np.log(pn)) / np.log(pn.size)) if pn.size > 1 else 0.0
    # 谱平坦度（几何均值/算术均值，越接近 1 越平坦）
    a = np.maximum(amps, np.finfo(float).eps)
    flatness = float(np.exp(np.mean(np.log(a))) / np.mean(a))
    out.extend([centroid, entropy, flatness])
    return out


def _envelope_shape_feats(env_freqs: np.ndarray, env_amps: np.ndarray) -> list[float]:
    """T5：包络谱低频段熵 + 主峰局部突出度（不依赖轴承参数）。"""
    amps = env_amps
    p = amps**2
    total = float(np.sum(p))
    if total <= 0:
        return [0.0, 0.0]
    pn = p / total
    pn = pn[pn > 0]
    entropy = float(-np.sum(pn * np.log(pn)) / np.log(pn.size)) if pn.size > 1 else 0.0
    i_peak = int(np.argmax(amps))
    df = float(env_freqs[1] - env_freqs[0])
    half_local = max(0.16 * float(env_freqs[i_peak]), 10.0 * df)
    ml = (env_freqs >= env_freqs[i_peak] - half_local) & (
        env_freqs <= env_freqs[i_peak] + half_local)
    local_med = max(float(np.median(amps[ml])) if np.count_nonzero(ml) >= 5
                    else float(np.median(amps)), np.finfo(float).eps)
    prominence = min(float(amps[i_peak] / local_med), 100.0)
    return [entropy, prominence]


def feature_names(use_physical: bool = True) -> list[str]:
    """特征名列表（与 extract_features 输出顺序一致）。"""
    names: list[str] = []
    for k in ("rms", "peak", "peak_to_peak", "kurtosis", "skewness",
              "crest_factor", "impulse_factor", "shape_factor", "clearance_factor"):
        names.append(f"t1_{k}" + ("_log" if k in _DIMENSIONAL_KEYS else ""))
    if use_physical:
        for fault in FAULT_TYPES:
            names.extend(f"t2_{fault}_k{k}" for k in range(1, N_HARMONICS_FEAT + 1))
    names.extend(["t3_best_kurtosis", "t3_band_center", "t3_band_width"])
    names.extend(f"t3_level{j}_max_kurt" for j in range(1, KURTOGRAM_LEVEL + 1))
    names.extend(f"t4_band{i}_energy" for i in range(N_SPEC_BANDS))
    names.extend(["t4_centroid", "t4_entropy", "t4_flatness"])
    names.extend(["t5_env_entropy", "t5_env_peak_prom"])
    return names


def extract_features(
    x: np.ndarray,
    fs: float,
    shaft_freq: float | None = None,
    bearing: Bearing | str | None = None,
    use_physical: bool = True,
    kurtogram_level: int = KURTOGRAM_LEVEL,
) -> np.ndarray:
    """提取特征向量。返回一维 float 数组（顺序同 feature_names）。

    use_physical=True 时必须提供 shaft_freq 与 bearing（T2 需要特征频率）。
    """
    x = np.asarray(x, dtype=float).ravel()
    if x.size < 512:
        raise ValueError("信号过短（特征提取需 ≥512 点）")

    # T1 时域指标（有量纲项 log 变换，使分布近似对称且尺度压缩）
    xd = detrend(x)
    ind = time_domain_indicators(xd)
    vec: list[float] = []
    for k, v in ind.items():
        vec.append(float(np.log1p(abs(v))) if k in _DIMENSIONAL_KEYS else float(v))

    # 公共中间量：kurtogram 选带 → 包络谱、幅值谱
    kg = kurtogram(xd, fs, max_level=kurtogram_level)
    lo, hi = kg.best_band
    from ..kurtogram import bandpass_signal

    xb = bandpass_signal(xd, fs, lo, hi)
    fault_freqs = None
    if use_physical:
        if shaft_freq is None or bearing is None:
            raise ValueError("use_physical=True 需要 shaft_freq 与 bearing")
        if isinstance(bearing, str):
            from ..fault_freqs import get_bearing

            bearing = get_bearing(bearing)
        fault_freqs = bearing.fault_frequencies(shaft_freq)
        fc_max = max(fault_freqs.values())
        env_max = min(6.0 * fc_max, fs / 2.0)
    else:
        env_max = min(1000.0, fs / 2.0)
    env_f, env_a = envelope_spectrum(xb, fs, max_freq=env_max)

    # T2 物理判别特征
    if use_physical:
        vec.extend(_harmonic_prominences(env_f, env_a, fault_freqs))

    # T3 谱峭度特征
    nyq = fs / 2.0
    vec.append(kg.best_kurtosis)
    vec.append((0.5 * (lo + hi)) / nyq)
    vec.append((hi - lo) / nyq)
    for row in kg.levels[:kurtogram_level]:
        vec.append(max((k for (_, _, k) in row), default=0.0))

    # T4 频谱形态
    spec_f, spec_a = amplitude_spectrum(xd, fs)
    vec.extend(_spectrum_shape_feats(spec_f, spec_a))

    # T5 包络谱形态
    vec.extend(_envelope_shape_feats(env_f, env_a))

    arr = np.nan_to_num(np.asarray(vec, dtype=float), nan=0.0, posinf=100.0, neginf=0.0)
    return arr

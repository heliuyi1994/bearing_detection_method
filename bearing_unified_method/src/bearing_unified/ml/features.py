# -*- coding: utf-8 -*-
"""特征工程：一段振动信号 → 固定长度特征向量。

六组特征（glm 五组迁移适配 + 统一包新增 T6；原理见 glm docs/principles.md
第 11 章与合并建议稿 §6.5）：

    T1 时域指标      9 维   复用 indicators（有量纲项 log 变换）
    T2 物理判别特征  15 维  三候选特征频率（BPFO/BPFI/BSF）× 5 次谐波的
                           包络谱局部突出度（glm 四候选 20 维 → 统一包三候选
                           15 维；与 matcher 同源；依赖转速+轴承参数）
    T3 谱峭度特征    6 维   Kurtogram 最佳峭度/频带位置宽度 + 各层峭度统计
    T4 频谱形态      11 维  8 分频带能量占比、谱重心、谱熵、谱平坦度
    T5 包络谱形态    2 维   低频段谱熵、主峰局部突出度（均不依赖轴承参数）
    T6 盲梳检结构化  8 维   净梳数、最强梳得分、是否落带、干扰标记数、
                           3 类定位置信度、未定位标记（combdet/localize 的
                           结构化输出；不依赖转速——频带由标称转速窗×几何
                           现算，是本组与 T2 的关键差异）

维度合计：默认（T2+T6 全开）9+15+6+11+2+8 = **51**；--no-comb 43；
--no-physical（去 T2）36；两者皆去 28。

设计要点：
- T2/T3 与传统诊断方法同源、T6 与 v2 盲梳检同源——模型学到的判据可被
  物理解释与复核（双轨+结构化双保险）；
- 突出度、能量占比、熵、置信度等均为**尺度不变**特征，模型对增益/载荷
  变化稳健（测试有逐项核对）；
- use_physical=False 剔除 T2（消融：纯统计盲特征，无需转速信息）；
  use_comb=False 剔除 T6（消融：检验结构化梳特征的增量价值，评估报告
  给出有/无对比后再决定保留与否）；
- T6 不依赖转速：定位频带由 rpm_window（标称窗）×轴承几何现算，与 v2
  DSP 流水线同规则；use_comb=True 需要 bearing（几何）。
"""

from __future__ import annotations

import numpy as np

from ..combdet import cepstrum_confirm, comb_scan, mark_interference
from ..fault_freqs import Bearing
from ..indicators import detrend, time_domain_indicators
from ..kurtogram import kurtogram
from ..localize import RPM_WINDOW_DEFAULT, band_of, class_bands, fr_window, localize_combs
from ..preprocessing import bandpass as zbandpass, clamp_band
from ..spectrum import amplitude_spectrum, envelope_spectrum, real_cepstrum

N_HARMONICS_FEAT = 5
N_SPEC_BANDS = 8
KURTOGRAM_LEVEL = 3  # 特征用途 3 层足够（最窄带 fs/16），比诊断用 4 层快一倍

#: 判别三类（统一包无保持架判别）
FAULT_TYPES_3 = ("BPFO", "BPFI", "BSF")

_DIMENSIONAL_KEYS = ("rms", "peak", "peak_to_peak")  # 有量纲 → log 变换

#: T6 盲梳检结构化特征名（独立开关 use_comb 控制；默认启用）
COMB_FEATURE_NAMES = (
    "t6_n_clean_combs",      # 净梳数（确认且未被干扰排除）
    "t6_best_comb_score",    # 最强净梳的梳得分（无梳 0）
    "t6_best_in_band",       # 最强净梳是否落入定位频带（0/1）
    "t6_n_interference",     # 被干扰防护标记的梳数
    "t6_conf_outer",         # 外圈定位置信度
    "t6_conf_inner",         # 内圈定位置信度
    "t6_conf_ball",          # 滚珠定位置信度
    "t6_unlocalized",        # 有净梳但全部无法定位（0/1）
)


def _harmonic_prominences(
    env_freqs: np.ndarray,
    env_amps: np.ndarray,
    fault_freqs: dict[str, float],
) -> list[float]:
    """三候选 × 5 次谐波的局部突出度（与 matcher 相同的窗口/基线规则，取值不判命中）。"""
    freqs, amps = env_freqs, env_amps
    df = float(freqs[1] - freqs[0])
    vals: list[float] = []
    for fault in FAULT_TYPES_3:
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


def _comb_feats(
    xb: np.ndarray,
    env_f: np.ndarray,
    env_a: np.ndarray,
    fs: float,
    bearing: Bearing,
    rpm_window: tuple[float, float],
    n_harmonics: int = 5,
) -> list[float]:
    """T6：盲梳检结构化特征（与 v2 DSP 同路径的轻量版）。

    在解调包络谱上跑 comb_scan → 倒谱互证 → 干扰防护 → 定位合成，
    输出结构化计数/得分/置信度（不做判决——判决留给 ML 模型）。
    """
    flo, _fhi = fr_window(rpm_window)
    fr_nominal = 0.5 * (flo + _fhi)
    df = float(env_f[1] - env_f[0])
    combs_a = comb_scan(env_f, env_a, n_harmonics=n_harmonics, dive_floor=flo - 2.0)
    cep_q, cep_v = real_cepstrum(xb, fs, max_quefrency=0.2)
    n_cep = n_interf = 0
    clean_f0s: list[float] = []
    best_score = 0.0
    best_in_band = 0.0
    bands = class_bands(bearing, rpm_window)
    for c in combs_a:
        ok, _hits = cepstrum_confirm(cep_q, cep_v, c["f0"])
        if not ok:
            continue
        n_cep += 1
        if c["low_f0"] or mark_interference(c["f0"], df, fr_nominal=fr_nominal):
            n_interf += 1
            continue
        clean_f0s.append(c["f0"])
        if c["score"] > best_score:
            best_score = float(c["score"])
            best_in_band = 1.0 if band_of(c["f0"], bands) else 0.0
    if not clean_f0s:
        # 无净梳：计数类特征如实给出（含干扰标记数），其余置零
        out = [0.0] * len(COMB_FEATURE_NAMES)
        out[3] = float(n_interf)
        return out
    loc = localize_combs(env_f, env_a, clean_f0s, bearing, rpm_window,
                         n_harmonics=n_harmonics)
    return [
        float(len(clean_f0s)),
        best_score,
        best_in_band,
        float(n_interf),
        float(loc["conf_outer"]),
        float(loc["conf_inner"]),
        float(loc["conf_ball"]),
        1.0 if loc["unlocalized"] else 0.0,
    ]


def feature_names(use_physical: bool = True, use_comb: bool = True) -> list[str]:
    """特征名列表（与 extract_features 输出顺序一致）。"""
    names: list[str] = []
    for k in ("rms", "peak", "peak_to_peak", "kurtosis", "skewness",
              "crest_factor", "impulse_factor", "shape_factor", "clearance_factor"):
        names.append(f"t1_{k}" + ("_log" if k in _DIMENSIONAL_KEYS else ""))
    if use_physical:
        for fault in FAULT_TYPES_3:
            names.extend(f"t2_{fault}_k{k}" for k in range(1, N_HARMONICS_FEAT + 1))
    names.extend(["t3_best_kurtosis", "t3_band_center", "t3_band_width"])
    names.extend(f"t3_level{j}_max_kurt" for j in range(1, KURTOGRAM_LEVEL + 1))
    names.extend(f"t4_band{i}_energy" for i in range(N_SPEC_BANDS))
    names.extend(["t4_centroid", "t4_entropy", "t4_flatness"])
    names.extend(["t5_env_entropy", "t5_env_peak_prom"])
    if use_comb:
        names.extend(COMB_FEATURE_NAMES)
    return names


def extract_features(
    x: np.ndarray,
    fs: float,
    shaft_freq: float | None = None,
    bearing: Bearing | str | None = None,
    use_physical: bool = True,
    use_comb: bool = True,
    rpm_window: tuple[float, float] = RPM_WINDOW_DEFAULT,
    kurtogram_level: int = KURTOGRAM_LEVEL,
) -> np.ndarray:
    """提取特征向量。返回一维 float 数组（顺序同 feature_names）。

    use_physical=True 时必须提供 shaft_freq 与 bearing（T2 需要特征频率）；
    use_comb=True 时必须提供 bearing（T6 需要几何计算定位频带；T6 不依赖
    shaft_freq——频带由标称转速窗×几何现算）。
    """
    x = np.asarray(x, dtype=float).ravel()
    if x.size < 512:
        raise ValueError("信号过短（特征提取需 ≥512 点）")
    if (use_physical or use_comb) and bearing is None:
        raise ValueError("use_physical 或 use_comb 为 True 时需要 bearing")
    if use_physical and shaft_freq is None:
        raise ValueError("use_physical=True 需要 shaft_freq")
    if isinstance(bearing, str):
        from ..fault_freqs import get_bearing

        bearing = get_bearing(bearing)

    # T1 时域指标（有量纲项 log 变换，使分布近似对称且尺度压缩）
    xd = detrend(x)
    ind = time_domain_indicators(xd)
    vec: list[float] = []
    for k, v in ind.items():
        vec.append(float(np.log1p(abs(v))) if k in _DIMENSIONAL_KEYS else float(v))

    # 公共中间量：kurtogram 选带 → 零相位带通 → 包络谱、幅值谱
    # （与 v2 DSP 同路径：Kurtogram → 夹取 → sosfiltfilt 零相位解调）
    kg = kurtogram(xd, fs, max_level=kurtogram_level)
    lo, hi = clamp_band(kg.best_band, fs)
    xb = zbandpass(xd, fs, lo, hi)
    fault_freqs = None
    env_max = min(1000.0, fs / 2.0)
    if use_physical:
        fault_freqs = bearing.fault_frequencies(shaft_freq)
        fc_max = max(fault_freqs.values())
        env_max = min(6.0 * fc_max, fs / 2.0)
    if use_comb:
        from ..combdet import COMB_ENV_MAX_HZ

        env_max = max(env_max, min(COMB_ENV_MAX_HZ, fs / 2.0))
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

    # T6 盲梳检结构化特征
    if use_comb:
        vec.extend(_comb_feats(xb, env_f, env_a, fs, bearing, rpm_window))

    arr = np.nan_to_num(np.asarray(vec, dtype=float), nan=0.0, posinf=100.0, neginf=0.0)
    return arr

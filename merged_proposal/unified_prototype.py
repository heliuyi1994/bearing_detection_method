# -*- coding: utf-8 -*-
"""统一合并方案原型（级联门控架构）。

对应 merged_proposal/双方案优缺点汇总与合并建议.md §6.2 的评审结论：
"glm 出骨架（选带→匹配），kimi 出门控与兜底（健康基线触发级、零相位预处理）"。
本模块作为第三方同时 import 两个姊妹包的公共 API 在外层组合
（glm: ``bearing_diag`` / kimi: ``bearing_fault``），两个包之间没有交叉导入。

流水线
------

.. code-block:: text

    原始信号（16 kHz 单声道 WAV）
      → 预处理（kimi preprocessing：去直流 + 去趋势 + 零相位带通 sosfiltfilt）
      → 转速自估（预处理信号的包络谱 1×fr 峰，搜索窗 [17.5, 20.5] Hz，
          prominence 检查失败回退 1150 rpm）
      →【触发级】马氏距离健康基线（kimi HealthBaseline，域内标定）
      │      └─ 未触发 → 判"正常"
      →【复核级】glm Kurtogram 自适应选带 → kimi 零相位带通 → glm 包络谱
         → glm matcher 谐波匹配（仅 BPFO/BPFI/BSF 三类，不做保持架 FTF；
            滚珠主峰 2×BSF 由 BSF 的 2 次谐波检索天然覆盖，与前次评测口径一致）
      → max(三类置信度) ≥ 阈值预设 → 判"轴承"（输出外圈/内圈/滚珠三个独立置信度）
         否则判"摩擦"；触发但两级证据皆弱 → low_conf 标记

关键取舍（决策记录）
--------------------

1. **触发级特征**：kimi ``DETECTION_FEATURE_NAMES`` 共 4 维，其中
   ``env_max_ratio`` 依赖故障特征频率（转速与轴承几何强先验）与解调选带，
   按任务约定回退为纯时域冲击性子集 ``(kurtosis, crest_factor,
   clearance_factor)``——三个无量纲指标对健康信号稳定、对冲击敏感，
   且与转速估计误差、选带误差解耦。
2. **预处理带通 (500, 7500) Hz @16 kHz**：本台架信号能量集中在 1.5–3 kHz；
   下沿 500 Hz 去除低频轰隆与工频（仍远高于最高特征频率 BPFI≈88 Hz），
   上沿 7500 Hz 给奈奎斯特留余量。该带通只作用于转速自估与触发级特征。
3. **复核级的输入**：Kurtogram 需要在所有候选频带上比较真实内容的包络
   峭度，若先宽带带通，带外只剩滤波器残差，其峭度是尺度不变的数值尘埃、
   可能虚假胜出，故复核级输入只做去直流/去趋势；选定频带后再用 kimi 的
   零相位 ``sosfiltfilt`` 带通解调，替代 glm 原生的非零相位
   ``sosfilt``——这正是移植项 2"零相位预处理"的落点。
4. **置信度口径**：三类置信度取 glm matcher 各候选的 score（[0,1]，已含
   局部邻域中位数基线与整数倍混淆消歧），不做 softmax 归一，逐文件落盘，
   三档预设（保守 0.3 / 平衡 0.2 / 灵敏 0.1）可由落盘置信度事后重算。

v2 判别规则（2026-09-21 评审裁定，``rules="v2"``）
---------------------------------------------------

判别从 v1 的"理论频率锚定匹配"改为"先盲检轴承故障特征存在性、再用理论
频率定位"，v1 路径完整保留（``rules="v1"``，默认）：

.. code-block:: text

    预处理(不变) → Kurtogram 选带 + 零相位解调 → 包络谱
      →【盲梳检】不依赖任何理论频率：
         A. 观测峰驱动梳扫：局部邻域中位数显著度拾取显著峰（复用 glm matcher
            显著度定义 峰/局部中位 ≥ 5），每个显著峰为候选基频 f0 ∈ [20,150] Hz，
            验证 k·f0 (k=2..5) 谐波族（容差 ±2% 或 1.5 谱线），w_k=1/k 加权得分
            （直接复用 match_fault_frequencies 单候选调用保证口径一致）；
            下探偏好"能解释最多显著峰的最小基频"，同族峰只保留一个梳
         B. 倒频谱互证：glm real_cepstrum，q≈m/f0 (m=1..3) 处显著 quefrency 峰
         → A、B 都确认 = 轴承故障特征存在（可多个梳）
      → 有梳 → 判「异常-轴承」→【定位】频带落入法（fr∈[18.33,20] Hz 使四类
         特征频带互不重叠：BSF 32.2–35.2 / BPFO 47.3–51.6 / 2×BSF 64.5–70.3 /
         BPFI 81.1–88.4 Hz）+ 理论复核（fr 取窗内最接近者，容差 ±4%）合成
         三类置信度；空档梳走 fr_hyp=f0/(k·阶次) 补救；仍无解判「轴承-未定位」
      → 无梳 且 马氏距离触发（v1 触发级降级为辅助判据）→「异常-非轴承(摩擦)」
      → 皆无 → 正常；证据皆弱 → 低置信单列
    干扰防护：f0 与 k×19.167 Hz（标称轴频谐波）或 50.0 Hz（工频，恰好落在
    外圈频带内！）在 1.5 谱线内吻合的梳打 suspected_interference 标记并从
    轴承判决中排除（仍记录）。
    v2 匹配不依赖转速估计（rpm_est 仅记录）——这是 v2 的关键收益：v1 的
    转速自估回退率高达 92.6%，理论锚定对边缘转速结构性不可达。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import signal as sps
from scipy.io import wavfile

from bearing_diag.fault_freqs import Bearing as GlmBearing
from bearing_diag.kurtogram import kurtogram
from bearing_diag.matcher import (
    LOCAL_MIN_BINS,
    LOCAL_TOL_REL,
    PROMINENCE_MIN,
    TOL_REL,
    match_fault_frequencies,
)
from bearing_diag.spectrum import envelope_spectrum, real_cepstrum
from bearing_fault.diagnosis import HealthBaseline
from bearing_fault.features import time_domain_features
from bearing_fault.preprocessing import bandpass, detrend, remove_dc

FS = 16000.0  # 音频资产库统一采样率

# 轴承几何（labeled_dataset_4class/geometry.json 的机器可读副本；同一台架）
GEOMETRY = {"n": 7, "d_mm": 3.969, "D_mm": 15.0, "alpha_deg": 5.6}
#: 设计阶次（×fr），用于与 glm 轴承对象自检
ORDERS = {"BPFO": 2.578320, "BPFI": 4.421680, "BSF": 1.758605, "FTF": 0.368331}

#: 预处理零相位带通频带（见模块 docstring 取舍 2）
PREPROCESS_BAND = (500.0, 7500.0)

RPM_FALLBACK = 1150.0            # 转速自估失败时的回退值（标称窗 1100–1200 的中点）
RPM_SEARCH_HZ = (17.5, 20.5)     # 1×fr 搜索窗（1100–1200 rpm ↔ 18.33–20 Hz，留余量）
RPM_BASELINE_HZ = (10.0, 25.0)   # 包络谱噪声地板估计窗
RPM_ENV_MAX_HZ = 120.0           # 转速自估只需低频段包络谱
RPM_PROM_MIN = 5.0               # 1×fr 峰所需最小突出度（与 matcher 的 PROMINENCE_MIN 一致）

#: 触发级特征（kimi 检测级冲击性特征的纯时域子集，见模块 docstring 取舍 1）
FEATURE_NAMES = ("kurtosis", "crest_factor", "clearance_factor")

#: 复核级"轴承证据"置信度预设三档
PRESETS = {"保守": 0.3, "平衡": 0.2, "灵敏": 0.1}
DEFAULT_PRESET = "平衡"

#: 触发级阈值取正常集折外马氏距离的分位数；默认 99，95/99.7 供敏感性
MAHA_QUANTILES = (95.0, 99.0, 99.7)
DEFAULT_MAHA_QUANTILE = 99.0

#: 复核级判别类别（保持架 FTF 按方案出 scope）
REVIEW_CLASSES = ("BPFO", "BPFI", "BSF")
CLASS_CN = {"BPFO": "外圈", "BPFI": "内圈", "BSF": "滚珠"}
#: decide()/diagnose_file() 内部置信度字典键 → 中文类名
CONF_CN = {"outer": "外圈", "inner": "内圈", "ball": "滚珠"}

# ---------------------------------------------------------------- v2 常量

#: v2 盲梳检候选基频范围：覆盖四类特征频带（32–88 Hz）及其低次谐波归属
COMB_F0_RANGE = (20.0, 150.0)
#: v2 包络谱分析上限（需覆盖 150 Hz 基频的 5 次谐波）
COMB_ENV_MAX_HZ = 800.0
COMB_MIN_HITS = 2        # 梳确认的最少谐波命中数（沿用 glm matcher MIN_HITS）
COMB_MIN_SCORE = 0.15    # 梳确认的最小得分（沿用 glm matcher MIN_SCORE）
CEP_PROM_MIN = 5.0       # 倒频谱互证 quefrency 峰显著度阈值（与谱峰显著度同口径）
CEP_RAHMONICS = 3        # 倒频谱互证检查的 quefrency 谐波（rahmonics）个数
FR_WINDOW = (1100.0 / 60.0, 1200.0 / 60.0)   # 标称轴频窗 (18.333, 20.0) Hz
FR_NOMINAL = 1150.0 / 60.0                   # 19.167 Hz，轴频谐波梳防护基准
MAINS_HZ = 50.0          # 工频干扰（恰落在外圈频带内，必须防护）
INTERFERENCE_TOL_BINS = 1.5   # 干扰吻合容差（谱线数，与 matcher 的 1.5 谱线一致）
LOCALIZE_TOL_REL = 0.04  # 定位理论复核的频率容差（任务规定放宽 ±4%）


def make_bearing() -> GlmBearing:
    """按台架几何构造 glm 轴承对象。"""
    return GlmBearing(
        name="audio_assets_custom",
        n_balls=GEOMETRY["n"],
        pitch_diameter=GEOMETRY["D_mm"],
        ball_diameter=GEOMETRY["d_mm"],
        contact_angle_deg=GEOMETRY["alpha_deg"],
    )


def verify_orders(tol: float = 2e-4) -> dict[str, float]:
    """自检：glm 轴承在 fr=1 Hz 的阶次与设计值逐项一致，且 BPFO+BPFI=n·fr。"""
    ff = make_bearing().fault_frequencies(1.0)
    for name, v in ORDERS.items():
        assert abs(ff[name] - v) < tol, f"阶次不一致 {name}: {ff[name]} != {v}"
    assert abs(ff["BPFO"] + ff["BPFI"] - GEOMETRY["n"]) < 1e-9
    return ff


def review_fault_freqs(rpm: float) -> dict[str, float]:
    """三类复核候选的特征频率（Hz），fr = rpm/60；不含 FTF。"""
    ff = make_bearing().fault_frequencies(rpm / 60.0)
    return {k: ff[k] for k in REVIEW_CLASSES}


def read_wav(path: str | Path, expect_fs: float = FS) -> tuple[np.ndarray, float, float, str]:
    """读 WAV → (满幅归一信号, 采样率, 削波比例, 备注)。

    采样率不为 expect_fs 时用 polyphase 有理数重采样转换（自带抗混叠），
    并记入备注——音频资产库中有个别 48 kHz 录音（如 id=20101）。
    """
    from fractions import Fraction

    sr, raw = wavfile.read(str(path))
    notes: list[str] = []
    if raw.ndim > 1:  # 防御：清单声称为单声道，若遇立体声取第一声道
        raw = raw[:, 0]
        notes.append("立体声取第1声道")
    clip_ratio = float(np.mean(np.abs(raw) >= 32767))
    x = raw.astype(np.float64) / 32768.0
    if sr != int(expect_fs):
        frac = Fraction(int(expect_fs), int(sr))  # 自动约分（48000→16000 即 1/3）
        x = sps.resample_poly(x, frac.numerator, frac.denominator)
        notes.append(f"重采样{sr}→{int(expect_fs)}Hz(resample_poly {frac.numerator}/{frac.denominator})")
        sr = int(expect_fs)
    return x, float(sr), clip_ratio, "; ".join(notes)


def preprocess(x: np.ndarray, fs: float = FS, band: tuple[float, float] = PREPROCESS_BAND) -> np.ndarray:
    """去直流 + 去线性趋势 + 零相位带通（kimi preprocessing，sosfiltfilt）。"""
    return bandpass(detrend(remove_dc(np.asarray(x, dtype=float))), fs, band[0], band[1])


def preprocess_light(x: np.ndarray) -> np.ndarray:
    """仅去直流 + 去趋势——复核级 Kurtogram 的输入（见模块 docstring 取舍 3）。"""
    return detrend(remove_dc(np.asarray(x, dtype=float)))


def estimate_rpm(x_pre: np.ndarray, fs: float = FS) -> dict:
    """包络谱 1×fr 峰自估转速；prominence 不足或无局部峰时回退 1150 rpm。

    在预处理信号的包络谱上，以 [10, 25] Hz 区段的中位数为噪声地板，
    在 [17.5, 20.5] Hz 窗内找局部峰；最强峰的 幅值/地板 ≥ RPM_PROM_MIN
    才接受，峰值频率经抛物线插值细化（3.8 s 信号 Δf≈0.26 Hz）。
    """
    env_f, env_a = envelope_spectrum(x_pre, fs, max_freq=RPM_ENV_MAX_HZ)
    info = {"rpm": RPM_FALLBACK, "fr_hz": RPM_FALLBACK / 60.0, "rpm_fallback": True,
            "rpm_prom_ratio": 0.0, "rpm_n_peaks": 0}
    region = (env_f >= RPM_BASELINE_HZ[0]) & (env_f <= RPM_BASELINE_HZ[1])
    if np.count_nonzero(region) < 8:
        return info
    floor = float(np.median(env_a[region]))
    floor = max(floor, np.finfo(float).eps)
    ridx = np.flatnonzero(region)
    pk, _ = sps.find_peaks(env_a[region])
    cands = [int(i) for i in pk
             if RPM_SEARCH_HZ[0] <= float(env_f[ridx[i]]) <= RPM_SEARCH_HZ[1]]
    info["rpm_n_peaks"] = len(cands)
    if not cands:
        return info
    best = max(cands, key=lambda i: float(env_a[ridx[i]]))
    j = int(ridx[best])
    ratio = float(env_a[j]) / floor
    info["rpm_prom_ratio"] = ratio
    if ratio < RPM_PROM_MIN:
        return info
    # 抛物线插值细化峰频（边界或退化时保持原频点）
    fr = float(env_f[j])
    if 0 < j < env_a.size - 1:
        y0, y1, y2 = float(env_a[j - 1]), float(env_a[j]), float(env_a[j + 1])
        denom = y0 - 2.0 * y1 + y2
        if denom > 0:
            df = float(env_f[1] - env_f[0])
            fr += 0.5 * (y0 - y2) / denom * df
    info.update(rpm=60.0 * fr, fr_hz=fr, rpm_fallback=False)
    return info


def extract_trigger_features(x_pre: np.ndarray) -> dict[str, float]:
    """触发级特征：纯时域冲击性 3 维（kimi time_domain_features 的子集）。"""
    feats = time_domain_features(np.asarray(x_pre, dtype=float))
    out = {k: float(feats[k]) for k in FEATURE_NAMES}
    # 防御非有限值（常值信号等极端情形）
    return {k: (v if np.isfinite(v) else 0.0) for k, v in out.items()}


def fit_baseline(feature_dicts: list[dict[str, float]], reg: float = 1e-6) -> HealthBaseline:
    """用健康样本特征字典列表拟合马氏距离基线（kimi HealthBaseline）。"""
    return HealthBaseline(list(FEATURE_NAMES), reg=reg).fit(feature_dicts)


def maha_distance(baseline: HealthBaseline, features: dict[str, float]) -> float:
    """单个样本相对健康基线的马氏距离。"""
    return float(baseline.distance(features))


def calibrate_oof(
    feature_dicts: list[dict[str, float]],
    n_splits: int = 5,
    seed: int = 42,
    quantiles: tuple[float, ...] = MAHA_QUANTILES,
) -> dict:
    """触发级防泄漏标定：n_splits 折交叉，折外距离的分位数作阈值。

    返回 dict：oof（与输入顺序一致的折外距离）、thresholds（各分位数阈值）、
    baseline（全部正常样本拟合的最终基线，用于给异常集计分）、fold_sizes。
    """
    n = len(feature_dicts)
    if n < 2 * n_splits:
        raise ValueError(f"标定样本过少：{n} < 2×{n_splits}")
    perm = np.random.default_rng(seed).permutation(n)
    folds = np.array_split(perm, n_splits)
    oof = np.empty(n, dtype=float)
    for k, te in enumerate(folds):
        tr = np.setdiff1d(perm, te)
        bl = fit_baseline([feature_dicts[int(i)] for i in tr])
        for i in te:
            oof[int(i)] = maha_distance(bl, feature_dicts[int(i)])
    thresholds = {f"p{q:g}": float(np.percentile(oof, q)) for q in quantiles}
    return {
        "oof": oof,
        "thresholds": thresholds,
        "baseline": fit_baseline(feature_dicts),
        "fold_sizes": [int(t.size) for t in folds],
        "n": n,
        "n_splits": n_splits,
        "seed": seed,
    }


def _clamp_band(band: tuple[float, float], fs: float) -> tuple[float, float]:
    """把 kurtogram 选带夹取到 kimi 零相位带通的合法域 0 < low < high < fs/2。"""
    nyq = fs / 2.0
    hi = min(float(band[1]), nyq - max(10.0, 0.001 * fs))
    lo = min(float(band[0]), hi * 0.9)
    lo = max(lo, 1.0)
    return (lo, hi)


def review_stage(x_dt: np.ndarray, fs: float, rpm: float, n_harmonics: int = 5) -> dict:
    """复核级：glm Kurtogram 选带 → kimi 零相位带通 → glm 包络谱 → glm matcher。

    返回外圈/内圈/滚珠三个独立置信度（候选 score，[0,1]，不归一）与中间量。
    """
    ff3 = review_fault_freqs(rpm)
    kg = kurtogram(x_dt, fs, max_level=4)
    band = _clamp_band(kg.best_band, fs)
    xb = bandpass(x_dt, fs, band[0], band[1])  # 零相位解调带通（移植项 2）
    env_max = min(6.0 * max(ff3.values()), fs / 2.0)  # 与 glm diagnose 同规则
    env_f, env_a = envelope_spectrum(xb, fs, max_freq=env_max)
    m = match_fault_frequencies(env_f, env_a, ff3, n_harmonics=n_harmonics)
    score = {c.fault: float(min(max(c.score, 0.0), 1.0)) for c in m.candidates}
    return {
        "conf_outer": score.get("BPFO", 0.0),
        "conf_inner": score.get("BPFI", 0.0),
        "conf_ball": score.get("BSF", 0.0),
        "band_low": float(band[0]),
        "band_high": float(band[1]),
        "best_kurtosis": float(kg.best_kurtosis),
        "match_verdict": m.verdict,          # BPFO/BPFI/BSF/normal（FTF 不参与）
        "match_confidence": float(m.confidence),  # 含次高惩罚的总置信度（参考）
    }


def decide(maha_dist: float, maha_threshold: float, conf3: dict[str, float],
           conf_threshold: float) -> dict:
    """级联判决：触发（距离 ≥ 阈值）→ 复核（max 置信度 ≥ 预设档）。

    低置信规则：触发 且 马氏距离 < 2×阈值 且 max 置信度 < 预设档。
    """
    mx = float(max(conf3.values()))
    argmax = max(conf3, key=lambda k: conf3[k])
    trigger = bool(maha_dist >= maha_threshold)
    if not trigger:
        pred = "正常"
        low_conf = False
    else:
        pred = "轴承" if mx >= conf_threshold else "摩擦"
        low_conf = bool(maha_dist < 2.0 * maha_threshold and mx < conf_threshold)
    return {
        "trigger": trigger,
        "pred_normal_abnormal": "异常" if trigger else "正常",
        "pred_fault_type": pred,
        "low_conf": low_conf,
        "max_conf": mx,
        "argmax": argmax,
    }


# ---------------------------------------------------------------- v2：盲梳检 + 定位

def _review_spectrum_v2(x_dt: np.ndarray, fs: float) -> dict:
    """v2 共用前端：Kurtogram 选带 → kimi 零相位解调 → 包络谱 + 带通信号（供倒谱）。"""
    kg = kurtogram(x_dt, fs, max_level=4)
    band = _clamp_band(kg.best_band, fs)
    xb = bandpass(x_dt, fs, band[0], band[1])
    env_f, env_a = envelope_spectrum(xb, fs, max_freq=min(COMB_ENV_MAX_HZ, fs / 2.0))
    q, cep = real_cepstrum(xb, fs, max_quefrency=0.2)  # 覆盖 f0≥20 Hz 的 3 次 rahmonics
    return {"band": band, "best_kurtosis": float(kg.best_kurtosis),
            "env_f": env_f, "env_a": env_a, "cep_q": q, "cep_v": cep}


def pick_salient_peaks(
    env_f: np.ndarray,
    env_a: np.ndarray,
    f0_range: tuple[float, float] = COMB_F0_RANGE,
    prom_min: float = PROMINENCE_MIN,
) -> list[tuple[float, float, float]]:
    """观测峰驱动：在 [f0_range] 内拾取显著峰（glm matcher 的显著度定义）。

    显著度 = 峰幅值 / 局部邻域（±max(16%, 10 谱线)）中位数 ≥ prom_min。
    返回 [(freq, amp, prom)]，按频率升序。
    """
    env_f = np.asarray(env_f, dtype=float)
    env_a = np.asarray(env_a, dtype=float)
    df = float(env_f[1] - env_f[0])
    idx, _ = sps.find_peaks(env_a)
    peaks: list[tuple[float, float, float]] = []
    for j in idx:
        f = float(env_f[j])
        if not (f0_range[0] <= f <= f0_range[1]):
            continue
        half_local = max(LOCAL_TOL_REL * f, LOCAL_MIN_BINS * df)
        ml = (env_f >= f - half_local) & (env_f <= f + half_local)
        local_med = (float(np.median(env_a[ml])) if np.count_nonzero(ml) >= 5
                     else float(np.median(env_a)))
        prom = float(env_a[j]) / max(local_med, np.finfo(float).eps)
        if prom >= prom_min:
            peaks.append((f, float(env_a[j]), prom))
    return peaks


def _comb_match(
    env_f: np.ndarray,
    env_a: np.ndarray,
    f0: float,
    n_harmonics: int = 5,
) -> tuple[float, int, list[float]]:
    """单候选基频的谐波族验证——直接调 glm match_fault_frequencies（单候选），
    保证 ±2%/1.5 谱线容差、局部中位显著度、w_k=1/k 加权与 v1 口径一致。

    返回 (score, hits, 命中谐波的实测峰位列表)。
    """
    m = match_fault_frequencies(env_f, env_a, {"comb": float(f0)},
                                n_harmonics=n_harmonics)
    cs = m.candidates[0]
    covered = [float(h["found_hz"]) for h in cs.harmonics if h["hit"]]
    return float(cs.score), int(cs.hits), covered


def comb_scan(
    env_f: np.ndarray,
    env_a: np.ndarray,
    f0_range: tuple[float, float] = COMB_F0_RANGE,
    n_harmonics: int = 5,
    min_hits: int = COMB_MIN_HITS,
    min_score: float = COMB_MIN_SCORE,
    dive_floor: float = FR_WINDOW[0] - 2.0,
) -> list[dict]:
    """盲梳扫（v2-A）：显著峰 → 候选基频 → 下探最小基频 → 谐波族验证 → 去重。

    - 下探：f0/2 的谐波族命中数不少于 f0 时取更小基频（应对基频弱、谐波强，
      如滚珠主峰在 2×BSF）。候选峰来自 [20,150] Hz，但下探允许到 dive_floor
      （≈16.3 Hz）：若最小基频落到 20 Hz 以下，该梳是轴频族等低频周期性
      （如实测中 17.1 Hz 轴频谐波族的子梳 34.2/51.3 Hz 会误入故障频带），
      打 low_f0 标记并按干扰排除——这是"偏好能解释最多显著峰的最小基频"
      规则的直接推论；
    - 去重：已接受梳的命中峰不再作为新梳候选（同一族峰只保留一个梳）；
    - 确认条件：hits ≥ min_hits 且 score ≥ min_score。

    返回 [{f0, score, hits, covered, low_f0}]，按基频升序。
    """
    peaks = pick_salient_peaks(env_f, env_a, f0_range)
    if not peaks:
        return []
    df = float(env_f[1] - env_f[0])
    peak_freqs = [p[0] for p in peaks]
    combs: list[dict] = []
    explained: list[float] = []  # 已接受梳的命中峰位 + 其全谐波覆盖的显著峰
    for f0, _amp, _prom in peaks:
        if any(abs(f0 - e) <= max(TOL_REL * f0, 1.5 * df) for e in explained):
            continue  # 该显著峰已被已接受的梳解释
        f = f0
        for _ in range(4):  # 下探最小基频（f0/2、/4、/8、/16，不早于 dive_floor）
            fh = f / 2.0
            if fh < dive_floor:
                break
            _, h_hits, _ = _comb_match(env_f, env_a, fh, n_harmonics)
            _, f_hits, _ = _comb_match(env_f, env_a, f, n_harmonics)
            if h_hits >= f_hits and h_hits >= min_hits:
                f = fh
            else:
                break
        if any(abs(f - e) <= max(TOL_REL * f, 1.5 * df) for e in explained):
            continue  # 下探结果与已接受梳同族（如 102.6→51.3 已被 17.1 梳覆盖）
        score, hits, covered = _comb_match(env_f, env_a, f, n_harmonics)
        if hits >= min_hits and score >= min_score:
            combs.append({"f0": float(f), "score": float(score), "hits": int(hits),
                          "covered": covered, "low_f0": bool(f < f0_range[0])})
            explained.extend(covered)
            # 吸收该梳全部谐波上的显著峰（含超出 n_harmonics 验证范围的高次，
            # 如 17.1 Hz 轴频族的第 7 次谐波 119.5 Hz——防止同族残余峰再立梳）
            k_max = int(peak_freqs[-1] / f)
            for pf in peak_freqs:
                k = int(round(pf / f))
                if 1 <= k <= k_max and abs(pf - k * f) <= max(TOL_REL * pf, 1.5 * df):
                    explained.append(pf)
    return combs


def cepstrum_confirm(
    q: np.ndarray,
    cep: np.ndarray,
    f0: float,
    n_rahmonics: int = CEP_RAHMONICS,
    prom_min: float = CEP_PROM_MIN,
) -> tuple[bool, int]:
    """倒频谱互证（v2-B）：q ≈ m/f0 (m=1..n_rahmonics) 处的显著 quefrency 峰。

    显著度定义与谱峰一致（局部邻域中位数基线）。任一 rahmonic 显著即互证成立。
    返回 (是否互证, 显著 rahmonic 数)。
    """
    q = np.asarray(q, dtype=float)
    cep = np.asarray(cep, dtype=float)
    if q.size < 8:
        return False, 0
    dq = float(q[1] - q[0])
    hits = 0
    for m in range(1, n_rahmonics + 1):
        q0 = m / f0
        if q0 > q[-1]:
            break
        half_win = max(TOL_REL * q0, 1.5 * dq)
        msk = (q >= q0 - half_win) & (q <= q0 + half_win)
        if not np.any(msk):
            continue
        idx = int(np.flatnonzero(msk)[int(np.argmax(cep[msk]))])
        half_local = max(LOCAL_TOL_REL * q0, LOCAL_MIN_BINS * dq)
        ml = (q >= q0 - half_local) & (q <= q0 + half_local)
        local_med = (float(np.median(cep[ml])) if np.count_nonzero(ml) >= 5
                     else float(np.median(cep)))
        prom = float(cep[idx]) / max(local_med, np.finfo(float).eps)
        if prom >= prom_min:
            hits += 1
    return hits >= 1, hits


def mark_interference(
    f0: float,
    df: float,
    fr_nominal: float = FR_NOMINAL,
    mains: float = MAINS_HZ,
    tol_bins: float = INTERFERENCE_TOL_BINS,
) -> bool:
    """干扰防护：f0 与 k×fr_nominal（轴频谐波梳）或工频 50 Hz 在 tol_bins
    谱线内吻合 → 标记（工频 50 Hz 恰落在外圈频带 47.3–51.6 Hz 内）。"""
    tol = tol_bins * df
    if abs(f0 - mains) <= tol:
        return True
    k = int(round(f0 / fr_nominal))
    return k >= 1 and abs(f0 - k * fr_nominal) <= tol


def _theory_scores(
    env_f: np.ndarray,
    env_a: np.ndarray,
    fr: float,
    n_harmonics: int = 5,
) -> dict[str, float]:
    """三类理论频率在轴频 fr 下的谐波族得分（glm matcher 一次三候选）。

    三类阶次间无整数倍关系（2.578/4.422/1.759），不触发混淆消歧，各类独立。
    """
    ff = {"BPFO": ORDERS["BPFO"] * fr, "BPFI": ORDERS["BPFI"] * fr,
          "BSF": ORDERS["BSF"] * fr}
    m = match_fault_frequencies(env_f, env_a, ff, n_harmonics=n_harmonics)
    return {c.fault: float(min(max(c.score, 0.0), 1.0)) for c in m.candidates}


def _class_bands() -> list[tuple[str, int, float, float, float]]:
    """频带落入法的频带表：由阶次与标称轴频窗动态计算（注释给 Hz 数值）。

    返回 [(类, k, 阶次, 带低, 带高)]：滚珠 32.2–35.2（BSF 基频）、
    外圈 47.3–51.6（BPFO）、滚珠 64.5–70.3（2×BSF 主峰）、内圈 81.1–88.4（BPFI）。
    """
    lo, hi = FR_WINDOW
    return [
        ("ball", 1, ORDERS["BSF"], ORDERS["BSF"] * lo, ORDERS["BSF"] * hi),
        ("outer", 1, ORDERS["BPFO"], ORDERS["BPFO"] * lo, ORDERS["BPFO"] * hi),
        ("ball", 2, ORDERS["BSF"], 2 * ORDERS["BSF"] * lo, 2 * ORDERS["BSF"] * hi),
        ("inner", 1, ORDERS["BPFI"], ORDERS["BPFI"] * lo, ORDERS["BPFI"] * hi),
    ]


def localize_comb(
    env_f: np.ndarray,
    env_a: np.ndarray,
    f0: float,
    tol_rel: float = LOCALIZE_TOL_REL,
    min_score: float = COMB_MIN_SCORE,
    min_hits: int = COMB_MIN_HITS,
) -> dict:
    """定位单个确认梳（频带落入法 + 理论复核）。

    假设空间 (类, k)：fr_hyp = f0/(k·阶次) 夹取到标称轴频窗后，理论频率
    F = k·阶次·fr_hyp 与 f0 的偏差 ≤ ±4% 才有效——频带内命中（fr_hyp 在窗内，
    偏差≈0）与空档补救（k≥2 使 fr_hyp 回窗）被统一覆盖。有效假设上跑三类理论
    复核得分，取该梳指定类得分最高者定类；全部假设无效或得分不足 → 无法定位。

    返回 dict：localized、cls（outer/inner/ball/None）、confs（三类得分）、
    band（落入频带标签或 ""）、fr_hyp、note。
    """
    lo, hi = FR_WINDOW
    band_of = {"ball1": "滚珠(BSF基频带)", "outer1": "外圈(BPFO带)",
               "ball2": "滚珠(2×BSF主峰带)", "inner1": "内圈(BPFI带)"}
    bands = _class_bands()
    band_label = ""
    for cls, k, _order, blo, bhi in bands:
        if blo - 1e-9 <= f0 <= bhi + 1e-9:
            band_label = band_of[f"{cls}{k}"]
            break

    glm_of = {"outer": "BPFO", "inner": "BPFI", "ball": "BSF"}
    confs = {"outer": 0.0, "inner": 0.0, "ball": 0.0}
    best_cls, best_score, best_fr = None, 0.0, None
    for cls in ("outer", "inner", "ball"):
        for k in range(1, 6):
            fr = f0 / (k * ORDERS[glm_of[cls]])
            fr_clip = min(max(fr, lo), hi)
            f_theo = k * ORDERS[glm_of[cls]] * fr_clip
            if abs(f_theo - f0) > tol_rel * f0:
                continue  # 窗内无法解释该梳（超 ±4% 容差）
            scores = _theory_scores(env_f, env_a, fr_clip)
            for g, key in (("BPFO", "outer"), ("BPFI", "inner"), ("BSF", "ball")):
                confs[key] = max(confs[key], scores[g])
            if scores[glm_of[cls]] > best_score:
                best_cls, best_score, best_fr = cls, scores[glm_of[cls]], fr_clip
    localized = best_cls is not None and best_score >= min_score
    return {
        "localized": bool(localized),
        "cls": best_cls if localized else None,
        "confs": confs,
        "band": band_label,
        "fr_hyp": float(best_fr) if best_fr else None,
        "best_score": float(best_score),
    }


def review_stage_v2(x_dt: np.ndarray, fs: float) -> dict:
    """v2 复核主路径：盲梳检（A）→ 倒谱互证（B）→ 干扰防护 → 定位。

    返回全部中间量；判决由 decide_v2 在拿到马氏距离后完成。
    """
    spec = _review_spectrum_v2(x_dt, fs)
    env_f, env_a = spec["env_f"], spec["env_a"]
    df = float(env_f[1] - env_f[0])
    combs_a = comb_scan(env_f, env_a)
    combs: list[dict] = []
    n_cep = 0
    n_interf = 0
    for c in combs_a:
        ok, cep_hits = cepstrum_confirm(spec["cep_q"], spec["cep_v"], c["f0"])
        c["cep_confirmed"] = bool(ok)
        c["cep_hits"] = int(cep_hits)
        if not ok:
            continue
        n_cep += 1
        # 干扰防护：低频梳（f0<20，轴频族）或 k×fr_nominal / 工频 50 Hz 吻合
        if c["low_f0"]:
            c["suspected_interference"] = True
            c["guard_reason"] = "低频梳(f0<20Hz,疑轴频族)"
        elif mark_interference(c["f0"], df):
            k = int(round(c["f0"] / FR_NOMINAL))
            c["suspected_interference"] = True
            c["guard_reason"] = ("工频50Hz" if abs(c["f0"] - MAINS_HZ) <= INTERFERENCE_TOL_BINS * df
                              else f"轴频谐波{k}×{FR_NOMINAL:.3f}Hz")
        else:
            c["suspected_interference"] = False
            c["guard_reason"] = ""
        if c["suspected_interference"]:
            n_interf += 1
        combs.append(c)
    # 定位：对确认且未标记干扰的梳逐个定位，三类置信度逐类取 max
    clean = [c for c in combs if not c["suspected_interference"]]
    confs = {"outer": 0.0, "inner": 0.0, "ball": 0.0}
    n_localized = 0
    for c in clean:
        loc = localize_comb(env_f, env_a, c["f0"])
        c["loc"] = loc
        if loc["localized"]:
            n_localized += 1
        for key in confs:
            confs[key] = max(confs[key], loc["confs"][key])
    return {
        "band_low": float(spec["band"][0]),
        "band_high": float(spec["band"][1]),
        "best_kurtosis": spec["best_kurtosis"],
        "combs": combs,                       # A∩B 确认的梳（含被标记干扰的）
        "comb_confirmed": len(combs_a),       # A 确认的梳数
        "cepstrum_confirmed": n_cep,          # 其中 B 互证通过的梳数
        "suspected_interference": n_interf,   # 被干扰防护标记的梳数
        "n_clean_combs": len(clean),
        "n_localized": n_localized,
        "bearing_unlocalized": bool(clean and not n_localized),
        "conf_outer": confs["outer"],
        "conf_inner": confs["inner"],
        "conf_ball": confs["ball"],
    }


def decide_v2(
    maha_dist: float,
    maha_threshold: float,
    rev2: dict,
    conf_threshold: float,
) -> dict:
    """v2 判决：有确认且未被标记干扰的梳 → 异常-轴承（定位/未定位）；
    无梳且马氏触发（v1 触发级降级为辅助判据）→ 异常-非轴承(摩擦)；皆无 → 正常。

    低置信（沿用 v1 精神）：判摩擦但距离 < 2×阈值；或判轴承但 max 置信度 < 预设档。
    「轴承-未定位」走 bearing_unlocalized 单列，不并入低置信。
    """
    conf3 = {"outer": rev2["conf_outer"], "inner": rev2["conf_inner"],
             "ball": rev2["conf_ball"]}
    mx = float(max(conf3.values()))
    argmax = max(conf3, key=lambda k: conf3[k])
    trigger = bool(maha_dist >= maha_threshold)
    if rev2["n_clean_combs"] > 0:
        if rev2["n_localized"] > 0:
            pred = "轴承"
            low_conf = bool(mx < conf_threshold)
        else:
            pred = "轴承-未定位"
            low_conf = False
    elif trigger:
        pred = "摩擦"
        low_conf = bool(maha_dist < 2.0 * maha_threshold)
    else:
        pred = "正常"
        low_conf = False
    return {
        "trigger": trigger,
        "pred_normal_abnormal": "异常" if pred != "正常" else "正常",
        "pred_fault_type": pred,
        "low_conf": low_conf,
        "max_conf": mx,
        "argmax": argmax,
    }


def diagnose_file(
    path: str | Path,
    fs: float = FS,
    calibrator: HealthBaseline | None = None,
    maha_threshold: float | None = None,
    preset: str = DEFAULT_PRESET,
    conf_threshold: float | None = None,
    rules: str = "v1",
) -> dict:
    """对单个 WAV 文件执行完整诊断，返回全部中间量与判决。

    rules="v1"（默认）：理论锚定级联（触发级马氏距离 → 复核级理论谐波匹配）。
    rules="v2"：盲梳检存在性 + 频带落入定位（见模块 docstring）；rpm 仅记录，
    匹配不依赖转速。

    calibrator / maha_threshold 缺省时只计算特征与置信度（评测脚本先收集
    全部特征再做防泄漏标定，随后用 decide()/decide_v2() 事后判决）。
    """
    x, fs_actual, clip_ratio, note = read_wav(path, expect_fs=fs)
    x_pre = preprocess(x, fs_actual)
    rpm_info = estimate_rpm(x_pre, fs_actual)  # v2 匹配不依赖，仅记录
    feats = extract_trigger_features(x_pre)
    x_dt = preprocess_light(x)

    out: dict = {
        "rpm_est": float(rpm_info["rpm"]),
        "rpm_fallback": bool(rpm_info["rpm_fallback"]),
        "rpm_fr_hz": float(rpm_info["fr_hz"]),
        "rpm_prom_ratio": float(rpm_info["rpm_prom_ratio"]),
        "feat_kurtosis": feats["kurtosis"],
        "feat_crest_factor": feats["crest_factor"],
        "feat_clearance_factor": feats["clearance_factor"],
        "clip_ratio": clip_ratio,
        "n_samples": int(x.size),
        "duration_s": float(x.size / fs_actual),
        "note": note,
        "maha_dist": float("nan"),
        "trigger": False,
        "pred_normal_abnormal": "",
        "pred_fault_type": "",
        "low_conf": False,
    }

    if rules == "v1":
        rev = review_stage(x_dt, fs_actual, rpm_info["rpm"])
        conf3 = {"outer": rev["conf_outer"], "inner": rev["conf_inner"],
                 "ball": rev["conf_ball"]}
        out.update({
            "conf_outer": rev["conf_outer"],
            "conf_inner": rev["conf_inner"],
            "conf_ball": rev["conf_ball"],
            "max_conf": float(max(conf3.values())),
            "argmax_class": CONF_CN[max(conf3, key=lambda k: conf3[k])],
            "band_low": rev["band_low"],
            "band_high": rev["band_high"],
            "best_kurtosis": rev["best_kurtosis"],
            "match_verdict": rev["match_verdict"],
            "match_confidence": rev["match_confidence"],
        })
    elif rules == "v2":
        rev2 = review_stage_v2(x_dt, fs_actual)
        conf3 = {"outer": rev2["conf_outer"], "inner": rev2["conf_inner"],
                 "ball": rev2["conf_ball"]}
        out.update({
            "conf_outer": rev2["conf_outer"],
            "conf_inner": rev2["conf_inner"],
            "conf_ball": rev2["conf_ball"],
            "max_conf": float(max(conf3.values())),
            "argmax_class": CONF_CN[max(conf3, key=lambda k: conf3[k])],
            "band_low": rev2["band_low"],
            "band_high": rev2["band_high"],
            "best_kurtosis": rev2["best_kurtosis"],
            # 梳检明细（interference 标记的梳在 comb_freqs 中以 ! 后缀标注）
            "comb_freqs": ";".join(
                f"{c['f0']:.2f}{'!' if c['suspected_interference'] else ''}"
                for c in rev2["combs"]),
            "comb_confirmed": rev2["comb_confirmed"],
            "cepstrum_confirmed": rev2["cepstrum_confirmed"],
            "suspected_interference": rev2["suspected_interference"],
            "bearing_unlocalized": int(rev2["bearing_unlocalized"]),
            "n_clean_combs": rev2["n_clean_combs"],
            "n_localized": rev2["n_localized"],
            "comb_detail": json.dumps([
                {"f0": round(c["f0"], 3), "score": round(c["score"], 4),
                 "hits": c["hits"], "cep_hits": c["cep_hits"],
                 "interference": c["suspected_interference"],
                 "guard": c.get("guard_reason", ""),
                 "band": c.get("loc", {}).get("band", ""),
                 "cls": c.get("loc", {}).get("cls"),
                 "fr_hyp": c.get("loc", {}).get("fr_hyp")}
                for c in rev2["combs"]], ensure_ascii=False),
        })
    else:
        raise ValueError(f"未知规则版本: {rules}")

    if calibrator is not None:
        out["maha_dist"] = maha_distance(calibrator, feats)
        if maha_threshold is not None:
            thr = conf_threshold if conf_threshold is not None else PRESETS[preset]
            if rules == "v1":
                out.update(decide(out["maha_dist"], maha_threshold, conf3, thr))
            else:
                out.update(decide_v2(out["maha_dist"], maha_threshold, rev2, thr))
            out["argmax_class"] = CONF_CN[out.pop("argmax")]
    return out

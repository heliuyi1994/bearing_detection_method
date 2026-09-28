# -*- coding: utf-8 -*-
"""盲梳检（v2-A/B）：不依赖任何理论频率的轴承故障特征存在性确认。

A. 观测峰驱动梳扫：局部邻域中位数显著度拾取显著峰（复用 matcher 的显著度
   定义：峰/局部中位 ≥ 5），以每个显著峰为候选基频 f0（默认 [20,150] Hz），
   验证 k·f0（k=2..5）谐波族（容差 ±2% 或 1.5 谱线，w_k=1/k 加权——直接
   调 matcher 单候选保证口径一致）；下探偏好"能解释最多显著峰的最小基频"
   （应对基频弱、谐波强，如滚珠主峰在 2×BSF），同族峰只保留一个梳。
B. 倒频谱互证：real_cepstrum 上检查 q≈m/f0（m=1..3）处显著 quefrency 峰。

A、B 都确认 = "轴承故障特征存在"（可多个梳）。

干扰防护：f0 与 k×fr_nominal（轴频谐波族）或工频 50 Hz（恰落在外圈频带）
在 1.5 谱线内吻合的梳、以及下探后基频 <20 Hz 的轴频族梳，打
suspected_interference 标记并从轴承判决中排除（仍记录）。
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sps

from .matcher import (
    LOCAL_MIN_BINS,
    LOCAL_TOL_REL,
    PROMINENCE_MIN,
    TOL_REL,
    match_fault_frequencies,
)

#: 盲梳检候选基频范围：覆盖四类特征频带（32–88 Hz）及其低次谐波归属
COMB_F0_RANGE = (20.0, 150.0)
#: 包络谱分析上限（需覆盖 150 Hz 基频的 5 次谐波）
COMB_ENV_MAX_HZ = 800.0
COMB_MIN_HITS = 2        # 梳确认的最少谐波命中数（沿用 matcher MIN_HITS）
COMB_MIN_SCORE = 0.15    # 梳确认的最小得分（沿用 matcher MIN_SCORE）
CEP_PROM_MIN = 5.0       # 倒频谱互证 quefrency 峰显著度阈值（与谱峰同口径）
CEP_RAHMONICS = 3        # 倒频谱互证检查的 quefrency 谐波（rahmonics）个数
MAINS_HZ = 50.0          # 工频干扰（恰落在外圈频带内，必须防护）
#: 工频防护的绝对频率容差上限（Hz）。工频是精确频率（漂移 <0.1 Hz），容差
#: 只用于覆盖谱分辨率：取 min(1.5 谱线, 0.4 Hz)。不能用裸 1.5 谱线——短窗
#: （如 1–2 s 的 ML 特征窗，df=0.5–1.0 Hz）下 ±0.75–1.5 Hz 的工频区会误吞
#: 真外圈梳（BPFO 49.42 Hz 距 50 Hz 仅 0.58 Hz）。0.4 Hz 上限下 49.42 安全、
#: 50.0±0.4 的梳仍被排除（频带内真外圈 50.2 Hz 处会被误标——固有的工频/外圈
#: 歧义，按建议稿取保守）。
MAINS_TOL_MAX_HZ = 0.4
INTERFERENCE_TOL_BINS = 1.5   # 干扰吻合容差（谱线数，与 matcher 一致）
#: 同族吸收的相对容差。matcher 的 ±2% 搜索窗含滑差裕量，直接用于"同族吸收"
#: 会把相邻 ~2% 的**异族**强峰误并（实测：伪梳 33.08 Hz 把真梳 162.3 Hz 当
#: 5 次谐波吸收，真故障漏检）。同族判定改严到 ±1%（≈滑差量级）：真族分裂
#: 的代价仅是产生一个同类的重复梳（良性），异族误并的代价是真梳被漏检
#: （有害）——取前者。与合并原型的 ±2% 吸收存在刻意行为差异（改进）。
ABSORB_TOL_REL = 0.01


def pick_salient_peaks(
    env_f: np.ndarray,
    env_a: np.ndarray,
    f0_range: tuple[float, float] = COMB_F0_RANGE,
    prom_min: float = PROMINENCE_MIN,
) -> list[tuple[float, float, float]]:
    """观测峰驱动：在 [f0_range] 内拾取显著峰（matcher 的显著度定义）。

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


def comb_match(
    env_f: np.ndarray,
    env_a: np.ndarray,
    f0: float,
    n_harmonics: int = 5,
    tol_rel: float = TOL_REL,
) -> tuple[float, int, list[float], list[int]]:
    """单候选基频的谐波族验证——调 matcher 单候选，保证 ±2%/1.5 谱线容差、
    局部中位显著度、w_k=1/k 加权口径一致。

    返回 (score, hits, 命中谐波的实测峰位列表, 命中谐波次数 k 列表)。
    """
    m = match_fault_frequencies(env_f, env_a, {"comb": float(f0)},
                                n_harmonics=n_harmonics, tol_rel=tol_rel)
    cs = m.candidates[0]
    hit_ks = [int(h["k"]) for h in cs.harmonics if h["hit"]]
    covered = [float(h["found_hz"]) for h in cs.harmonics if h["hit"]]
    return float(cs.score), int(cs.hits), covered, hit_ks


def comb_scan(
    env_f: np.ndarray,
    env_a: np.ndarray,
    f0_range: tuple[float, float] = COMB_F0_RANGE,
    n_harmonics: int = 5,
    min_hits: int = COMB_MIN_HITS,
    min_score: float = COMB_MIN_SCORE,
    dive_floor: float = 16.33,
) -> list[dict]:
    """盲梳扫（v2-A）：显著峰 → 候选基频 → 下探最小基频 → 谐波族验证 → 去重。

    - 下探：f0/2 的谐波族命中数不少于 f0 时取更小基频。候选峰来自
      f0_range，但下探允许到 dive_floor（默认 ≈16.3 Hz，即标称轴频窗下沿
      减 2 Hz）：若最小基频落到 f0_range 下沿（20 Hz）以下，该梳是轴频族
      等低频周期性（实测：17.1 Hz 轴频谐波族的子梳 34.2/51.3 Hz 会误入
      故障频带），打 low_f0 标记并按干扰排除；
    - 去重：已接受梳的命中峰与其全部谐波上的显著峰不再作为新梳候选
      （同一族峰只保留一个梳）；下探结果与已接受梳同族也跳过；
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
            _, h_hits, _, _ = comb_match(env_f, env_a, fh, n_harmonics)
            _, f_hits, _, _ = comb_match(env_f, env_a, f, n_harmonics)
            if h_hits >= f_hits and h_hits >= min_hits:
                f = fh
            else:
                break
        if any(abs(f - e) <= max(TOL_REL * f, 1.5 * df) for e in explained):
            continue  # 下探结果与已接受梳同族（如 102.6→51.3 已被 17.1 梳覆盖）
        score, hits, covered, hit_ks = comb_match(env_f, env_a, f, n_harmonics)
        # 防"高次窗扫峰"伪梳：错误基频的高次谐波窗（±2% 相对窗随 k 线性展宽）
        # 会扫到不相干强峰拼成假族——要求最低命中谐波 ≤ 2（基频或 2 次）。
        # 真族天然满足（基频或 2 次谐波几乎总在），伪梳多为 k≥3 的拼凑。
        family_ok = hits >= min_hits and score >= min_score and hit_ks and min(hit_ks) <= 2
        if family_ok:
            combs.append({"f0": float(f), "score": float(score), "hits": int(hits),
                          "covered": covered, "low_f0": bool(f < f0_range[0])})
            explained.extend(covered)
            # 吸收该梳全部谐波上的显著峰（含超出 n_harmonics 验证范围的高次，
            # 如 17.1 Hz 轴频族的第 7 次谐波 119.5 Hz——防止同族残余峰再立梳）。
            # 容差用 ABSORB_TOL_REL（±1%）而非 matcher 的 ±2%：见常量注释。
            k_max = int(peak_freqs[-1] / f)
            for pf in peak_freqs:
                k = int(round(pf / f))
                if 1 <= k <= k_max and abs(pf - k * f) <= max(ABSORB_TOL_REL * pf, 1.5 * df):
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
    fr_nominal: float,
    mains: float = MAINS_HZ,
    mains_tol_max: float = MAINS_TOL_MAX_HZ,
    tol_bins: float = INTERFERENCE_TOL_BINS,
) -> str:
    """干扰防护：f0 与 k×fr_nominal（轴频谐波族）或工频 50 Hz 吻合 → 原因串。

    - 工频：|f0 − mains| ≤ min(tol_bins×df, mains_tol_max)——工频是精确频率，
      容差只覆盖谱分辨率并设 0.4 Hz 硬上限（见 MAINS_TOL_MAX_HZ 注释）；
    - 轴频谐波族：f0 ≈ k×fr_nominal（k≥1）在 tol_bins 谱线内——fr_nominal
      一般取转速窗中点（1150 rpm → 19.167 Hz），只覆盖标称附近，窗边缘的
      轴频族由 low_f0 下探规则兜底；
    不吻合返回 ""。
    """
    if abs(f0 - mains) <= min(tol_bins * df, mains_tol_max):
        return f"工频{mains:g}Hz"
    tol = tol_bins * df
    k = int(round(f0 / fr_nominal))
    if k >= 1 and abs(f0 - k * fr_nominal) <= tol:
        return f"轴频谐波{k}×{fr_nominal:.3f}Hz"
    return ""

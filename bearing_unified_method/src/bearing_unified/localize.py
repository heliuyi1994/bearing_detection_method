# -*- coding: utf-8 -*-
"""v2 定位：频带落入法 + 理论复核，输出外圈/内圈/滚珠 3 类置信度。

频带表**函数化**：由转速窗（默认 1100–1200 rpm）与轴承几何的阶次现算
四个互不重叠的频带——滚珠 BSF 基频带、外圈 BPFO 带、滚珠 2×BSF 主峰带、
内圈 BPFI 带（FIELD7 @1100–1200 rpm 时为 32.2–35.2 / 47.3–51.6 /
64.5–70.3 / 81.1–88.4 Hz，与合并建议稿 §6.2 表一致）；转速窗或几何可变，
频带随之重算（class_bands 测试有数值核对）。

定位规则：
- 梳基频 f0 落入某频带 → 粗定类；理论复核：fr_hyp = f0/(k·阶次) 夹取到
  轴频窗后，理论频率与 f0 偏差 ≤ ±4%（准入容差）才有效，有效假设上跑
  三类谐波复核得分（matcher，容差放宽 ±4% 参数化），取指定类得分最高者
  定类；
- f0 落入频带间空档 → 补救路径：fr_hyp = f0/(k·阶次)（k=1..5）在窗内且
  理论谐波族吻合者定类；
- 仍无解 → 「轴承特征存在但无法定位」（localized=False），单列语义，
  不强行归类。

判别仅 3 类（外圈 BPFO / 内圈 BPFI / 滚珠 BSF；滚珠 2×BSF 主峰带归入滚珠），
保持架 FTF 不参与定位。
"""

from __future__ import annotations

import numpy as np

from .fault_freqs import Bearing
from .matcher import match_fault_frequencies

#: 判别三类（glm 键名 → localize 内部键名）
LOCALIZE_CLASSES = {"outer": "BPFO", "inner": "BPFI", "ball": "BSF"}
CLASS_CN = {"outer": "外圈", "inner": "内圈", "ball": "滚珠"}

#: 定位准入容差（任务规定 ±4%）：|k·阶次·fr_hyp − f0| ≤ 4%·f0
LOCALIZE_TOL_REL = 0.04
#: 理论复核的谐波搜索窗（任务规定 matcher 容差在 v2 理论复核时放宽 ±4%）
THEORY_MATCH_TOL_REL = 0.04
#: 理论复核采纳的最低得分（沿用 matcher MIN_SCORE）
LOCALIZE_MIN_SCORE = 0.15
#: 标称转速窗（rpm）→ 轴频窗 (Hz)，与 labeled_dataset_4class 标称一致
RPM_WINDOW_DEFAULT = (1100.0, 1200.0)


def fr_window(rpm_window: tuple[float, float] = RPM_WINDOW_DEFAULT) -> tuple[float, float]:
    """转速窗 (rpm) → 轴频窗 fr (Hz)。"""
    return (rpm_window[0] / 60.0, rpm_window[1] / 60.0)


def bearing_orders(bearing: Bearing) -> dict[str, float]:
    """轴承在 fr=1 Hz 的三类判别阶次（FTF 不参与定位）。"""
    ff = bearing.fault_frequencies(1.0)
    return {k: ff[v] for k, v in LOCALIZE_CLASSES.items()}


def class_bands(
    bearing: Bearing,
    rpm_window: tuple[float, float] = RPM_WINDOW_DEFAULT,
) -> list[dict]:
    """由转速窗×几何阶次计算四个定位频带（函数化，不硬编码）。

    返回 [{cls, k, order, lo, hi, label}]：滚珠 k=1 BSF 基频带、外圈 k=1 BPFO
    带、滚珠 k=2 2×BSF 主峰带、内圈 k=1 BPFI 带。转速窗/几何改变时频带
    随之重算；同一几何与默认转速窗下数值与合并建议稿 §6.2 表一致。
    """
    lo, hi = fr_window(rpm_window)
    orders = bearing_orders(bearing)
    spec = [("ball", 1, "滚珠(BSF基频带)"), ("outer", 1, "外圈(BPFO带)"),
            ("ball", 2, "滚珠(2×BSF主峰带)"), ("inner", 1, "内圈(BPFI带)")]
    return [
        {"cls": cls, "k": k, "order": orders[cls],
         "lo": k * orders[cls] * lo, "hi": k * orders[cls] * hi, "label": label}
        for cls, k, label in spec
    ]


def band_of(f0: float, bands: list[dict]) -> str:
    """f0 落入的频带标签（无则空串）。"""
    for b in bands:
        if b["lo"] - 1e-9 <= f0 <= b["hi"] + 1e-9:
            return b["label"]
    return ""


def _theory_scores(
    env_f: np.ndarray,
    env_a: np.ndarray,
    bearing: Bearing,
    fr: float,
    n_harmonics: int = 5,
    tol_rel: float = THEORY_MATCH_TOL_REL,
) -> dict[str, float]:
    """三类理论频率在轴频 fr 下的谐波族得分（matcher 一次三候选）。

    三类阶次间一般无整数倍关系（FIELD7: 2.578/4.422/1.759），不触发混淆
    消歧，各类独立；容差按 v2 理论复核规定放宽 ±4%（参数化）。
    """
    ff = bearing.fault_frequencies(fr)
    ff3 = {v: ff[v] for v in LOCALIZE_CLASSES.values()}
    m = match_fault_frequencies(env_f, env_a, ff3, n_harmonics=n_harmonics,
                                tol_rel=tol_rel)
    return {c.fault: float(min(max(c.score, 0.0), 1.0)) for c in m.candidates}


def localize_comb(
    env_f: np.ndarray,
    env_a: np.ndarray,
    f0: float,
    bearing: Bearing,
    rpm_window: tuple[float, float] = RPM_WINDOW_DEFAULT,
    tol_rel: float = LOCALIZE_TOL_REL,
    min_score: float = LOCALIZE_MIN_SCORE,
    n_harmonics: int = 5,
) -> dict:
    """定位单个确认梳（频带落入法 + 理论复核）。

    假设空间 (类, k)，k=1..5：fr_hyp = f0/(k·阶次) 夹取到轴频窗后，理论
    频率 F = k·阶次·fr_hyp 与 f0 的偏差 ≤ tol_rel（±4%）才有效——频带内
    命中（fr_hyp 在窗内，偏差≈0）与空档补救（k≥2 使 fr_hyp 回窗）被统一
    覆盖。有效假设上跑三类理论复核得分，取该梳指定类得分最高者定类；
    全部假设无效或得分不足 → 无法定位。

    返回 dict：localized、cls（outer/inner/ball/None）、confs（三类得分）、
    band（落入频带标签或 ""）、fr_hyp、best_score。
    """
    lo, hi = fr_window(rpm_window)
    orders = bearing_orders(bearing)
    bands = class_bands(bearing, rpm_window)
    band_label = band_of(f0, bands)

    confs = {"outer": 0.0, "inner": 0.0, "ball": 0.0}
    best_cls, best_score, best_fr = None, 0.0, None
    for cls, glm_key in LOCALIZE_CLASSES.items():
        for k in range(1, 6):
            fr = f0 / (k * orders[cls])
            fr_clip = min(max(fr, lo), hi)
            f_theo = k * orders[cls] * fr_clip
            if abs(f_theo - f0) > tol_rel * f0:
                continue  # 窗内无法解释该梳（超 ±4% 准入容差）
            scores = _theory_scores(env_f, env_a, bearing, fr_clip,
                                    n_harmonics=n_harmonics)
            for g, key in (("BPFO", "outer"), ("BPFI", "inner"), ("BSF", "ball")):
                confs[key] = max(confs[key], scores[g])
            if scores[glm_key] > best_score:
                best_cls, best_score, best_fr = cls, scores[glm_key], fr_clip
    localized = best_cls is not None and best_score >= min_score
    return {
        "localized": bool(localized),
        "cls": best_cls if localized else None,
        "confs": confs,
        "band": band_label,
        "fr_hyp": float(best_fr) if best_fr else None,
        "best_score": float(best_score),
    }


def localize_combs(
    env_f: np.ndarray,
    env_a: np.ndarray,
    f0s: list[float],
    bearing: Bearing,
    rpm_window: tuple[float, float] = RPM_WINDOW_DEFAULT,
    **kwargs,
) -> dict:
    """多个确认梳的批量定位与 3 类置信度合成（逐类取 max）。

    返回 dict：per_comb（各梳 localize_comb 结果）、n_localized、
    unlocalized（有梳但全部无法定位）、conf_outer/inner/ball。
    """
    per_comb = [localize_comb(env_f, env_a, f0, bearing, rpm_window, **kwargs)
                for f0 in f0s]
    confs = {"outer": 0.0, "inner": 0.0, "ball": 0.0}
    n_localized = 0
    for loc in per_comb:
        if loc["localized"]:
            n_localized += 1
        for key in confs:
            confs[key] = max(confs[key], loc["confs"][key])
    return {
        "per_comb": per_comb,
        "n_localized": n_localized,
        "unlocalized": bool(f0s and not n_localized),
        "conf_outer": confs["outer"],
        "conf_inner": confs["inner"],
        "conf_ball": confs["ball"],
    }

# -*- coding: utf-8 -*-
"""马氏距离健康基线（移植 kimi HealthBaseline）+ 域内标定流程。

思路：用健康样本的特征向量估计基线分布 N(μ, Σ)，待测样本的马氏距离
D = sqrt((x−μ)ᵀ Σ⁻¹ (x−μ)) 反映其偏离健康状态的程度。相对化、可域内
重标定——这是 kimi 侧吸取"固定阈值跨域失效"教训后的设计（合并方案
移植项 1），在统一包的 v2 流水线中承担**辅助判据**角色：
盲梳检无梳时，用触发级（D ≥ 域内标定阈值）区分「异常-非轴承(摩擦)」
与「正常」。

特征口径（与合并原型一致）：kimi 检测级冲击性特征中的纯时域子集
(kurtosis, crest_factor, clearance_factor)——env_max_ratio 依赖故障
特征频率（转速/几何强先验）与选带，v2 的辅助判据刻意与之解耦。

重尾塌陷教训（2026-09 音频资产库评测实测，见
analysis/reports/audio_assets/音频资产库评测报告.md §4/§9）：
当健康总体的冲击性特征重尾（正常录音中约 3.7% 峭度 >5、最大 902）时，
协方差被尾部样本撑大，白化后典型折外距离被压缩（中位仅 0.36），p99
阈值被推到 8.89，轻中度异常落在「不算远」的位置——马氏距离的椭球
假设在该总体上失效，触发级灵敏度塌陷。可选缓解（未内置，留阶段二评估）：
特征 log/秩变换、robust 协方差（EllipticEnvelope 路径）、分位数触发。
因此本模块的定位是**辅助判据**而非主检路径：v2 主检是盲梳检
（combdet），马氏触发只在无梳时兜底。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

#: 触发级特征（纯时域冲击性 3 维，见模块 docstring 口径说明）
FEATURE_NAMES = ("kurtosis", "crest_factor", "clearance_factor")

#: 触发阈值默认取折外距离的分位数（95/99.7 供敏感性）
MAHA_QUANTILES = (95.0, 99.0, 99.7)
DEFAULT_MAHA_QUANTILE = 99.0


def feature_vector(features: dict[str, float], names: list[str]) -> NDArray[np.float64]:
    """按给定名字顺序把特征字典排成向量，供距离判别使用。"""
    return np.array([features[name] for name in names], dtype=np.float64)


class HealthBaseline:
    """健康状态基线：由健康样本特征向量估计马氏距离参考分布。

    参数
    ----
    feature_names : 构成特征向量的特征名顺序。
    reg : 协方差矩阵正则化系数，防止小样本下奇异。
    """

    def __init__(self, feature_names: list[str] | tuple[str, ...] = FEATURE_NAMES,
                 reg: float = 1e-6) -> None:
        self.feature_names = list(feature_names)
        self.reg = float(reg)
        self._mean: NDArray[np.float64] | None = None
        self._cov_inv: NDArray[np.float64] | None = None

    def fit(self, samples: list[dict[str, float]]) -> "HealthBaseline":
        """用若干健康样本的特征字典拟合均值与协方差逆矩阵。"""
        if len(samples) < 2:
            raise ValueError("至少需要 2 个健康样本估计基线分布")
        mat = np.array([feature_vector(s, self.feature_names) for s in samples])
        self._mean = mat.mean(axis=0)
        cov = np.cov(mat, rowvar=False)
        cov += self.reg * np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0])
        self._cov_inv = np.linalg.inv(cov)
        return self

    def distance(self, features: dict[str, float]) -> float:
        """待测样本特征向量相对健康基线的马氏距离。"""
        if self._mean is None or self._cov_inv is None:
            raise RuntimeError("请先调用 fit() 拟合基线")
        diff = feature_vector(features, self.feature_names) - self._mean
        return float(np.sqrt(diff @ self._cov_inv @ diff))


def calibrate_oof(
    feature_dicts: list[dict[str, float]],
    n_splits: int = 5,
    seed: int = 42,
    quantiles: tuple[float, ...] = MAHA_QUANTILES,
    feature_names: list[str] | tuple[str, ...] = FEATURE_NAMES,
) -> dict:
    """触发级防泄漏标定：n_splits 折交叉，折外距离的分位数作阈值。

    每折用其余折拟合基线、给留出折算距离，得 N 个折外距离；
    阈值 = 折外距离的各分位数。异常集应用时用全部健康样本拟合的
    最终基线计距离。返回 dict：oof（与输入顺序一致）、thresholds、
    baseline、fold_sizes。
    """
    n = len(feature_dicts)
    if n < 2 * n_splits:
        raise ValueError(f"标定样本过少：{n} < 2×{n_splits}")
    perm = np.random.default_rng(seed).permutation(n)
    folds = np.array_split(perm, n_splits)
    oof = np.empty(n, dtype=float)
    for te in folds:
        tr = np.setdiff1d(perm, te)
        bl = HealthBaseline(feature_names).fit([feature_dicts[int(i)] for i in tr])
        for i in te:
            oof[int(i)] = bl.distance(feature_dicts[int(i)])
    thresholds = {f"p{q:g}": float(np.percentile(oof, q)) for q in quantiles}
    return {
        "oof": oof,
        "thresholds": thresholds,
        "baseline": HealthBaseline(feature_names).fit(feature_dicts),
        "fold_sizes": [int(t.size) for t in folds],
        "n": n,
        "n_splits": n_splits,
        "seed": seed,
    }

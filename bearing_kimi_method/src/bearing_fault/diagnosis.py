"""故障判别：谱峰匹配定性与马氏距离定量判别。

两条互补的判别路径：

1. **谱峰匹配**（定位故障部位）：在包络谱上分别检索四类故障特征频率的
   1–3 次谐波，命中峰幅值与谱噪声地板（幅值中位数）之比即显著性，
   各类故障取其谐波平均倍数作为得分，取得分最高者为故障类型；
   全部得分低于阈值则判健康。
2. **马氏距离**（判断是否异常）：用若干健康样本的特征向量估计基线分布
   N(μ, Σ)，待测样本的马氏距离 D = sqrt((x−μ)ᵀ Σ⁻¹ (x−μ)) 反映其偏离
   健康状态的程度，距离越大异常越显著。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .features import feature_vector
from .spectrum import find_peaks_near

#: 谱峰匹配平均峰地比低于该值时判为健康。
#: 实测：健康信号噪声峰约 5–12 倍地板，故障谱线通常 >80 倍地板。
DEFAULT_HEALTH_THRESHOLD = 15.0


def fault_scores(
    env_freqs: NDArray[np.float64],
    env_amps: NDArray[np.float64],
    char_freqs: dict[str, float],
    harmonics: int = 3,
    tol: float = 0.02,
) -> dict[str, float]:
    """对每类故障计算谱峰匹配得分。

    得分 = 该类故障 1–harmonics 次谐波命中峰的"峰值/噪声地板中位数"
    之比的平均值。以幅值中位数作为噪声地板是稳健的：包络谱中谱线稀疏，
    中位数几乎不受谱线影响。
    """
    env_freqs = np.asarray(env_freqs)
    env_amps = np.asarray(env_amps)
    floor = float(np.median(env_amps))
    if floor <= 0:
        floor = float(env_amps.mean()) or 1.0
    scores: dict[str, float] = {}
    for name, f0 in char_freqs.items():
        hits = find_peaks_near(env_freqs, env_amps, f0, harmonics=harmonics, tol=tol)
        if hits:
            scores[name] = float(np.mean([a / floor for _, _, a in hits]))
        else:
            scores[name] = 0.0
    return scores


def diagnose(
    env_freqs: NDArray[np.float64],
    env_amps: NDArray[np.float64],
    char_freqs: dict[str, float],
    harmonics: int = 3,
    tol: float = 0.02,
    health_threshold: float = DEFAULT_HEALTH_THRESHOLD,
) -> tuple[str, dict[str, float]]:
    """谱峰匹配判别，返回 (结论, 各类故障得分)。

    结论为得分最高的故障类型；若最高得分低于 health_threshold 则返回
    ``"healthy"``。
    """
    scores = fault_scores(env_freqs, env_amps, char_freqs, harmonics=harmonics, tol=tol)
    best = max(scores, key=scores.get)
    if scores[best] < health_threshold:
        return "healthy", scores
    return best, scores


def detected_peaks(
    env_freqs: NDArray[np.float64],
    env_amps: NDArray[np.float64],
    char_freqs: dict[str, float],
    harmonics: int = 3,
    tol: float = 0.02,
) -> dict[str, list[tuple[int, float, float]]]:
    """返回每类故障在包络谱上命中的谐波峰明细，供报告与绘图使用。"""
    return {
        name: find_peaks_near(env_freqs, env_amps, f0, harmonics=harmonics, tol=tol)
        for name, f0 in char_freqs.items()
    }


class HealthBaseline:
    """健康状态基线：由健康样本特征向量估计马氏距离参考分布。

    参数
    ----
    feature_names : 构成特征向量的特征名顺序。
    reg : 协方差矩阵正则化系数，防止小样本下奇异。
    """

    def __init__(self, feature_names: list[str], reg: float = 1e-6) -> None:
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

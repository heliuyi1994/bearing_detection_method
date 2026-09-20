"""时域统计特征：传统状态监测中最常用的一组无量纲/有量纲指标。

- 有量纲指标（RMS、峰值等）反映振动总能量，对故障发展阶段敏感；
- 无量纲指标（峭度、峰值因子、脉冲因子、裕度因子等）与信号概率分布
  形状有关，对早期局部冲击敏感，且原则上不随转速载荷线性变化。

峭度采用 Pearson 定义（正态分布 = 3）。对常值信号（方差为 0）返回 0，
避免 0/0 未定义。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats


def time_domain_features(x: NDArray[np.float64]) -> dict[str, float]:
    """计算一组时域统计特征，返回 {特征名: 数值} 字典。"""
    x = np.asarray(x, dtype=np.float64)
    abs_x = np.abs(x)
    rms = float(np.sqrt(np.mean(x**2)))
    peak = float(abs_x.max())
    mean_abs = float(abs_x.mean())
    mean_sqrt_abs_sq = float(np.mean(np.sqrt(abs_x)) ** 2)

    if float(np.std(x)) == 0.0:
        kurtosis = 0.0
        skewness = 0.0
    else:
        kurtosis = float(stats.kurtosis(x, fisher=False))
        skewness = float(stats.skew(x))

    return {
        "rms": rms,
        "peak": peak,
        "peak_to_peak": float(np.ptp(x)),
        "std": float(np.std(x)),
        "kurtosis": kurtosis,
        "skewness": skewness,
        "crest_factor": peak / rms if rms > 0 else 0.0,
        "impulse_factor": peak / mean_abs if mean_abs > 0 else 0.0,
        "shape_factor": rms / mean_abs if mean_abs > 0 else 0.0,
        "clearance_factor": peak / mean_sqrt_abs_sq if mean_sqrt_abs_sq > 0 else 0.0,
    }


def feature_vector(features: dict[str, float], names: list[str]) -> NDArray[np.float64]:
    """按给定名字顺序把特征字典排成向量，供距离判别使用。"""
    return np.array([features[name] for name in names], dtype=np.float64)

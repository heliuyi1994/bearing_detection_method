"""时域统计指标。

对零均值化后的振动信号 x（长度 N），指标定义（见 docs/principles.md 第 4 章）：

    RMS          均方根值       sqrt(mean(x^2))            反映振动总体能量
    Peak         峰值           max|x|                     反映冲击幅度
    PeakToPeak   峰峰值         max(x) - min(x)
    Kurtosis     峭度           E[x^4]/E[x^2]^2            对冲击敏感，正态=3，早期能力最强
    Skewness     偏度           E[x^3]/E[x^2]^1.5          分布不对称性
    CrestFactor  峰值因子       Peak / RMS                 冲击性与磨损早期指示
    ImpulseFactor 脉冲因子      Peak / mean|x|             对冲击比峰值因子更敏感
    ShapeFactor  波形因子       RMS / mean|x|              波形形状
    ClearanceFactor 裕度因子    Peak / (mean(sqrt|x|))^2   对早期点蚀冲击最敏感的指标族

峭度/偏度采用有偏（总体）估计 g1/g2，与信号处理文献一致（正态噪声峭度≈3）。
"""

from __future__ import annotations

import numpy as np

INDICATOR_KEYS = (
    "rms", "peak", "peak_to_peak", "kurtosis", "skewness",
    "crest_factor", "impulse_factor", "shape_factor", "clearance_factor",
)


def time_domain_indicators(x: np.ndarray) -> dict[str, float]:
    """计算 9 项时域统计指标。输入一维数组，返回指标字典。"""
    x = np.asarray(x, dtype=float).ravel()
    if x.size < 2:
        raise ValueError("信号长度至少为 2")

    absx = np.abs(x)
    mean_abs = np.mean(absx)
    mean_sqrt_abs = np.mean(np.sqrt(absx)) ** 2
    m2 = np.mean(x**2)
    rms = float(np.sqrt(m2))
    peak = float(np.max(absx))

    # 防零除（全零信号）
    def _safe_div(a: float, b: float) -> float:
        return float(a / b) if b > 0 else 0.0

    return {
        "rms": rms,
        "peak": peak,
        "peak_to_peak": float(np.max(x) - np.min(x)),
        "kurtosis": _safe_div(float(np.mean(x**4)), m2**2),
        "skewness": _safe_div(float(np.mean(x**3)), m2**1.5),
        "crest_factor": _safe_div(peak, rms),
        "impulse_factor": _safe_div(peak, mean_abs),
        "shape_factor": _safe_div(rms, mean_abs),
        "clearance_factor": _safe_div(peak, mean_sqrt_abs),
    }


def remove_dc(x: np.ndarray) -> np.ndarray:
    """去均值。"""
    return np.asarray(x, dtype=float) - np.mean(x)


def detrend(x: np.ndarray) -> np.ndarray:
    """去线性趋势（含去均值），避免低频漂移污染频谱。"""
    from scipy.signal import detrend as _detrend

    return np.asarray(_detrend(np.asarray(x, dtype=float), type="linear"), dtype=float)


def health_hint(indicators: dict[str, float]) -> str:
    """基于峭度的粗略健康提示（经验阈值，详见文档第 4 章）。"""
    k = indicators.get("kurtosis", 3.0)
    if k < 4.0:
        return "峭度≈3~4：无明显冲击成分，倾向正常（需结合包络谱确认）"
    if k < 6.0:
        return "峭度 4~6：存在轻度冲击成分，疑似早期故障，建议检查包络谱特征频率"
    return "峭度>6：存在显著冲击成分，故障特征明显，重点核对包络谱谐波匹配结果"

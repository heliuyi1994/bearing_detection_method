"""机器学习特征提取：把一段振动信号变成固定 26 维特征向量。

特征分三组（与各 DSP 模块一一对应，原理见 docs/03、04 章）：

- 时域 10 维：``time_domain_features`` 的 RMS、峭度、峰值因子等；
- 包络谱 13 维：四类故障特征频率 1–3 次谐波的峰地比
  （谱峰幅值 / 包络谱幅值中位数）及 12 项中的最大值——这是携带故障
  部位信息最强的一组特征；
- 频域 3 维：谱质心、谱峭度、共振带能量占比。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from ..bearing import Bearing
from ..envelope import auto_resonance_band, envelope_analysis
from ..features import time_domain_features
from ..spectrum import amplitude_spectrum, find_peaks_near

#: 四类故障在特征向量中的固定顺序
FAULT_ORDER = ("inner", "outer", "ball", "cage")

_HARMONICS = 3
_TOL = 0.02

_TIME_NAMES = (
    "rms", "peak", "peak_to_peak", "std", "kurtosis", "skewness",
    "crest_factor", "impulse_factor", "shape_factor", "clearance_factor",
)
_ENVELOPE_NAMES = tuple(
    f"env_{fault}_h{h}" for fault in FAULT_ORDER for h in range(1, _HARMONICS + 1)
) + ("env_max_ratio",)
_FREQ_NAMES = ("spec_centroid", "spec_kurtosis", "band_energy_ratio")

#: 26 维特征的固定名称顺序，训练与推理必须共用
FEATURE_NAMES: tuple[str, ...] = _TIME_NAMES + _ENVELOPE_NAMES + _FREQ_NAMES

#: 检测级（健康/异常）使用的特征子集。全 26 维下健康样本有近零方差特征，
#: 经标准化放大后 IsolationForest 高维稀疏过拟合、误报率高；这 4 维
#: "冲击性"指标对健康稳定、对故障极度敏感（实测故障 env_max_ratio 80–300
#: 倍地板，健康仅 5–12 倍）。
DETECTION_FEATURE_NAMES: tuple[str, ...] = (
    "kurtosis", "crest_factor", "clearance_factor", "env_max_ratio",
)


class FeatureExtractor:
    """信号 → 26 维特征字典。

    参数
    ----
    fs : 采样频率 (Hz)。
    band : 共振解调频带，None 时自动取 (fs/6, 0.45·fs)。
    bearing : 轴承几何参数（默认 SKF 6205）。
    """

    def __init__(
        self,
        fs: float = 12000.0,
        band: tuple[float, float] | None = None,
        bearing: Bearing | None = None,
    ) -> None:
        self.fs = float(fs)
        self.band = band
        self.bearing = bearing or Bearing.sk6205()

    def extract(self, x: NDArray[np.float64], rpm: float) -> dict[str, float]:
        """提取特征，返回键与 FEATURE_NAMES 一致的字典。"""
        x = np.asarray(x, dtype=np.float64)
        feats: dict[str, float] = {}
        feats.update(time_domain_features(x))
        feats.update(self._envelope_features(x, rpm))
        feats.update(self._spectrum_features(x))
        return feats

    def extract_vector(self, x: NDArray[np.float64], rpm: float) -> NDArray[np.float64]:
        """按 FEATURE_NAMES 顺序返回特征向量。"""
        feats = self.extract(x, rpm)
        return np.array([feats[name] for name in FEATURE_NAMES])

    # ------------------------------------------------------------------
    def _envelope_features(self, x: NDArray[np.float64], rpm: float) -> dict[str, float]:
        result = envelope_analysis(x, self.fs, band=self.band)
        env_freqs, env_amps = result["env_freqs"], result["env_amps"]
        floor = float(np.median(env_amps))
        if floor <= 0:
            floor = float(env_amps.mean()) or 1.0

        char_freqs = self.bearing.characteristic_frequencies(rpm / 60.0)
        feats: dict[str, float] = {}
        best = 0.0
        for fault in FAULT_ORDER:
            hits = find_peaks_near(
                env_freqs, env_amps, char_freqs[fault],
                harmonics=_HARMONICS, tol=_TOL,
            )
            by_harmonic = {h: a for h, _, a in hits}
            for h in range(1, _HARMONICS + 1):
                ratio = by_harmonic.get(h, 0.0) / floor if h in by_harmonic else 0.0
                feats[f"env_{fault}_h{h}"] = ratio
                best = max(best, ratio)
        feats["env_max_ratio"] = best
        return feats

    def _spectrum_features(self, x: NDArray[np.float64]) -> dict[str, float]:
        freqs, amps = amplitude_spectrum(x, self.fs)
        total = float(amps.sum())
        centroid = float((freqs * amps).sum() / total) if total > 0 else 0.0
        spec_kurt = float(stats.kurtosis(amps, fisher=False)) if amps.std() > 0 else 0.0
        band = self.band or auto_resonance_band(self.fs)
        mask = (freqs >= band[0]) & (freqs <= band[1])
        band_ratio = float((amps[mask] ** 2).sum() / (amps**2).sum()) if total > 0 else 0.0
        return {
            "spec_centroid": centroid,
            "spec_kurtosis": spec_kurt,
            "band_energy_ratio": band_ratio,
        }

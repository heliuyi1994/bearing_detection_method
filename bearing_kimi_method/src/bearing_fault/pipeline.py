"""端到端诊断流程编排：信号 → 包络分析 → 谱峰匹配 → 诊断报告。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .bearing import Bearing
from .diagnosis import DEFAULT_HEALTH_THRESHOLD, detected_peaks, diagnose
from .envelope import envelope_analysis
from .features import time_domain_features


@dataclass
class DiagnosisReport:
    """一次诊断的完整结果。"""

    verdict: str
    scores: dict[str, float]
    char_freqs: dict[str, float]
    peaks: dict[str, list[tuple[int, float, float]]]
    features: dict[str, float]
    env_freqs: NDArray[np.float64] = field(repr=False)
    env_amps: NDArray[np.float64] = field(repr=False)
    envelope: NDArray[np.float64] = field(repr=False)
    filtered: NDArray[np.float64] = field(repr=False)
    fs: float = 0.0
    rpm: float = 0.0
    band: tuple[float, float] = (0.0, 0.0)

    def peak_error(self, fault: str | None = None) -> float | None:
        """最强命中峰相对理论频率的相对误差；无命中返回 None。"""
        fault = fault or self.verdict
        hits = self.peaks.get(fault) or []
        if not hits:
            return None
        _, f_peak, _ = max(hits, key=lambda item: item[2])
        h = max(hits, key=lambda item: item[2])[0]
        return abs(f_peak - h * self.char_freqs[fault]) / (h * self.char_freqs[fault])

    def summary(self) -> dict:
        """可 JSON 序列化的诊断摘要。"""
        err = self.peak_error()
        return {
            "verdict": self.verdict,
            "scores": self.scores,
            "char_freqs_hz": self.char_freqs,
            "detected_peaks": {
                k: [{"harmonic": h, "freq_hz": f, "amp": a} for h, f, a in v]
                for k, v in self.peaks.items()
            },
            "peak_relative_error": err,
            "features": self.features,
            "fs": self.fs,
            "rpm": self.rpm,
            "filter_band_hz": list(self.band),
        }


class DiagnosisPipeline:
    """轴承故障诊断流水线。

    参数
    ----
    bearing : 轴承几何参数（默认 SKF 6205）。
    fs : 采样频率 (Hz)。
    band : 共振解调带通频带，None 时自动取 (fs/6, 0.45·fs)。
    harmonics / tol : 谱峰匹配的谐波数与相对容差。
    """

    def __init__(
        self,
        bearing: Bearing | None = None,
        fs: float = 12000.0,
        band: tuple[float, float] | None = None,
        harmonics: int = 3,
        tol: float = 0.02,
        health_threshold: float = DEFAULT_HEALTH_THRESHOLD,
    ) -> None:
        self.bearing = bearing or Bearing.sk6205()
        self.fs = float(fs)
        self.band = band
        self.harmonics = harmonics
        self.tol = tol
        self.health_threshold = health_threshold

    def run(self, x: NDArray[np.float64], rpm: float) -> DiagnosisReport:
        """对一段振动信号执行完整诊断流程。"""
        result = envelope_analysis(x, self.fs, band=self.band)
        shaft_freq = rpm / 60.0
        char_freqs = self.bearing.characteristic_frequencies(shaft_freq)
        verdict, scores = diagnose(
            result["env_freqs"],
            result["env_amps"],
            char_freqs,
            harmonics=self.harmonics,
            tol=self.tol,
            health_threshold=self.health_threshold,
        )
        peaks = detected_peaks(
            result["env_freqs"],
            result["env_amps"],
            char_freqs,
            harmonics=self.harmonics,
            tol=self.tol,
        )
        from .envelope import auto_resonance_band

        return DiagnosisReport(
            verdict=verdict,
            scores=scores,
            char_freqs=char_freqs,
            peaks=peaks,
            features=time_domain_features(x),
            env_freqs=result["env_freqs"],
            env_amps=result["env_amps"],
            envelope=result["envelope"],
            filtered=result["filtered"],
            fs=self.fs,
            rpm=float(rpm),
            band=self.band or auto_resonance_band(self.fs),
        )

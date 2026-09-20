"""诊断流水线编排：预处理 → 时域指标 → Kurtogram 选带 → 包络谱 → 谐波匹配 → 结论。

对应 docs/principles.md 第 10 章的完整工作流。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .fault_freqs import Bearing, bearing_to_dict
from .indicators import detrend, health_hint, time_domain_indicators
from .kurtogram import KurtogramResult, bandpass_signal, kurtogram
from .matcher import MatchResult, match_fault_frequencies
from .spectrum import amplitude_spectrum, envelope_spectrum, real_cepstrum


@dataclass
class DiagnosisResult:
    """一次完整诊断的全部中间量与结论（供报告与二次分析使用）。"""

    fs: float
    shaft_freq: float
    signal: np.ndarray                     # 预处理（去趋势）后的信号
    indicators: dict[str, float]
    health_hint: str
    spec_freqs: np.ndarray                 # FFT 幅值谱
    spec_amps: np.ndarray
    kurtogram_result: KurtogramResult | None
    demod_band: tuple[float, float]        # 实际使用的解调频带 (low, high)
    band_source: str                       # 'kurtogram' | 'manual' | 'full'
    env_freqs: np.ndarray                  # 包络谱
    env_amps: np.ndarray
    cepstrum_q: np.ndarray | None = None   # 倒频谱（辅助）
    cepstrum_v: np.ndarray | None = None
    match: MatchResult | None = None
    fault_freqs_used: dict[str, float] = field(default_factory=dict)
    bearing_info: dict | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        return self.match.verdict if self.match else "normal"

    @property
    def verdict_cn(self) -> str:
        return self.match.verdict_cn if self.match else "未知（未执行匹配）"

    @property
    def confidence(self) -> float:
        return self.match.confidence if self.match else 0.0

    @property
    def level(self) -> str:
        return self.match.level if self.match else "-"


def diagnose(
    x: np.ndarray,
    fs: float,
    shaft_freq: float,
    fault_freqs: dict[str, float] | None = None,
    bearing: Bearing | str | None = None,
    use_kurtogram: bool = True,
    manual_band: tuple[float, float] | None = None,
    kurtogram_level: int = 4,
    env_max_freq: float | None = None,
    n_harmonics: int = 5,
) -> DiagnosisResult:
    """执行完整诊断流水线。

    必须提供 fault_freqs（已知特征频率）或 bearing（Bearing 对象或内置名称，由几何参数计算）之一。
    use_kurtogram=False 且无 manual_band 时对全带做包络（不推荐，强噪声下失效）。
    """
    x = np.asarray(x, dtype=float).ravel()
    if fs <= 0:
        raise ValueError("fs 必须为正数")
    if shaft_freq <= 0:
        raise ValueError("shaft_freq 必须为正数（Hz）")
    if isinstance(bearing, str):
        from .fault_freqs import get_bearing

        bearing = get_bearing(bearing)
    if fault_freqs is None:
        if bearing is None:
            raise ValueError("需提供 fault_freqs 或 bearing 参数之一")
        fault_freqs = bearing.fault_frequencies(shaft_freq)
    if env_max_freq is None:
        # 默认覆盖到 5 次谐波再留余量；不高于奈奎斯特
        fc_max = max(fault_freqs.values())
        env_max_freq = min(6.0 * fc_max, fs / 2.0)

    notes: list[str] = []
    # 1. 预处理：去趋势
    xd = detrend(x)
    if x.size < 4 * fs / shaft_freq:
        notes.append(
            f"采集时长 {x.size / fs:.2f}s 不足 4 个转轴周期，"
            "特征频率谐波可能分辨不开，建议加长采集"
        )

    # 2. 时域指标
    indicators = time_domain_indicators(xd)

    # 3. 解调频带选择
    nyq = fs / 2.0
    if manual_band is not None:
        band = (float(manual_band[0]), float(manual_band[1]))
        if not (0 < band[0] < band[1] <= nyq):
            raise ValueError(f"manual_band 须满足 0 < low < high ≤ {nyq} Hz")
        band_source, kg = "manual", None
    elif use_kurtogram:
        kg = kurtogram(xd, fs, max_level=kurtogram_level)
        band, band_source = kg.best_band, "kurtogram"
        if kg.best_kurtosis < 4.0:
            notes.append(
                f"Kurtogram 最佳频带包络峭度仅 {kg.best_kurtosis:.2f}（≈3 为无冲击），"
                "信号中瞬态冲击成分弱"
            )
    else:
        band, band_source, kg = (0.0, nyq), "full", None

    # 4. 带通 + 包络谱
    xb = bandpass_signal(xd, fs, *band) if band_source != "full" else xd
    env_f, env_a = envelope_spectrum(xb, fs, max_freq=env_max_freq)

    # 5. 倒频谱（辅助证据）
    cep_q, cep_v = real_cepstrum(xb, fs)

    # 6. 谐波匹配判别
    match = match_fault_frequencies(env_f, env_a, fault_freqs, n_harmonics=n_harmonics)

    # 7. 汇总与交叉验证提示
    spec_f, spec_a = amplitude_spectrum(xd, fs)
    if match.verdict != "normal":
        fc = fault_freqs[match.verdict]
        notes.append(
            f"判定为 {match.verdict_cn}（特征频率 {fc:.2f} Hz，"
            f"置信度 {match.confidence:.2f}），请在包络谱图上人工核对谐波族"
        )
    k_ind = indicators["kurtosis"]
    if match.verdict != "normal" and k_ind < 3.5:
        notes.append(
            f"注意：时域峭度 {k_ind:.2f} 接近 3（无明显冲击），但包络谱匹配到谐波族；"
            "可能为轻度早期故障或干扰峰，建议人工复核"
        )
    if match.verdict == "normal" and k_ind > 5:
        notes.append(
            f"注意：时域峭度 {k_ind:.2f} 偏高但未匹配到特征频率谐波族；"
            "冲击可能来自轴承之外的振源（齿轮、松动等）"
        )

    return DiagnosisResult(
        fs=fs,
        shaft_freq=shaft_freq,
        signal=xd,
        indicators=indicators,
        health_hint=health_hint(indicators),
        spec_freqs=spec_f,
        spec_amps=spec_a,
        kurtogram_result=kg,
        demod_band=band,
        band_source=band_source,
        env_freqs=env_f,
        env_amps=env_a,
        cepstrum_q=cep_q,
        cepstrum_v=cep_v,
        match=match,
        fault_freqs_used=dict(fault_freqs),
        bearing_info=bearing_to_dict(bearing) if bearing else None,
        notes=notes + match.notes,
    )

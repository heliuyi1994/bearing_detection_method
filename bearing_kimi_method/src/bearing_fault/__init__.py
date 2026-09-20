"""bearing_fault：基于传统信号处理的滚动轴承故障诊断工具包。

公开 API：

- :class:`Bearing` — 轴承几何参数与特征频率（BPFO/BPFI/BSF/FTF）
- :class:`BearingSimulator` — Randall 冲击响应模型的故障信号仿真器
- :func:`envelope_analysis` / :func:`analytic_envelope` — 共振解调包络分析
- :func:`time_domain_features` — 时域统计特征
- :func:`amplitude_spectrum` / :func:`envelope_spectrum` / :func:`find_peaks_near`
- :func:`angular_resample` / :func:`order_spectrum` — 阶次分析
- :func:`stft_spectrogram` / :func:`cwt_spectrogram` — 时频分析
- :func:`diagnose` / :func:`fault_scores` / :class:`HealthBaseline` — 状态判别
- :class:`DiagnosisPipeline` / :class:`DiagnosisReport` — 端到端诊断
"""

from .bearing import Bearing
from .diagnosis import HealthBaseline, diagnose, detected_peaks, fault_scores
from .envelope import analytic_envelope, auto_resonance_band, envelope_analysis
from .features import feature_vector, time_domain_features
from .order_analysis import angular_resample, order_spectrum
from .pipeline import DiagnosisPipeline, DiagnosisReport
from .preprocessing import bandpass, detrend, remove_dc
from .simulator import FAULT_TYPES, BearingSimulator
from .spectrum import amplitude_spectrum, envelope_spectrum, find_peaks_near, power_spectrum
from .timefreq import cwt_spectrogram, morlet_wavelet, stft_spectrogram

__version__ = "0.1.0"

__all__ = [
    "Bearing",
    "BearingSimulator",
    "FAULT_TYPES",
    "DiagnosisPipeline",
    "DiagnosisReport",
    "HealthBaseline",
    "analytic_envelope",
    "auto_resonance_band",
    "envelope_analysis",
    "time_domain_features",
    "feature_vector",
    "amplitude_spectrum",
    "power_spectrum",
    "envelope_spectrum",
    "find_peaks_near",
    "angular_resample",
    "order_spectrum",
    "stft_spectrogram",
    "cwt_spectrogram",
    "morlet_wavelet",
    "bandpass",
    "detrend",
    "remove_dc",
    "diagnose",
    "detected_peaks",
    "fault_scores",
]

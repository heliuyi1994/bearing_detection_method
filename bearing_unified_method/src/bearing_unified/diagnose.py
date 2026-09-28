# -*- coding: utf-8 -*-
"""v2 诊断流水线编排（统一包原生，唯一内置规则）。

数据流（对应 README 与模块 docstring）：

    原始信号
      → 轻预处理（去直流 + 去趋势）→ Kurtogram 选带 → 零相位解调 → 包络谱
      → 盲梳检 combdet（A 梳扫 + B 倒谱互证 → 干扰防护）
      → 有梳 → 定位 localize（频带落入 + fr_hyp 补救 + 理论复核 ±4%）
      → 无梳 → 马氏健康基线触发（辅助判据，需提供标定基线）
      → 判决：正常 / 轴承（BPFO/BPFI/BSF）/ 轴承-未定位 / 异常-非轴承(摩擦)

转速自估仅记录（v2 匹配不依赖它）；判据阈值三档预设（保守/平衡/灵敏）
与合并原型口径一致。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal as sps

from .baseline import FEATURE_NAMES, HealthBaseline
from .combdet import (
    COMB_ENV_MAX_HZ,
    COMB_F0_RANGE,
    cepstrum_confirm,
    comb_scan,
    mark_interference,
)
from .fault_freqs import Bearing, bearing_to_dict, get_bearing
from .indicators import health_hint, time_domain_indicators
from .kurtogram import KurtogramResult, kurtogram
from .localize import (
    CLASS_CN,
    RPM_WINDOW_DEFAULT,
    class_bands,
    fr_window,
    localize_combs,
)
from .preprocessing import bandpass, clamp_band, detrend, remove_dc
from .spectrum import amplitude_spectrum, envelope_spectrum, real_cepstrum

#: 判定"轴承"的三档置信度预设（与合并原型一致，默认平衡档）
PRESETS = {"保守": 0.3, "平衡": 0.2, "灵敏": 0.1}
DEFAULT_PRESET = "平衡"

#: 转速自估的 1×fr 峰显著度阈值与噪声地板窗（与合并原型一致）
RPM_PROM_MIN = 5.0
RPM_BASELINE_HZ = (10.0, 25.0)

VERDICT_CN = {
    "normal": "正常（未检出轴承故障特征）",
    "friction": "异常-非轴承（摩擦；无轴承梳特征但马氏距离越限）",
    "bearing_unlocalized": "异常-轴承（检出故障特征但无法定位部位）",
    "BPFO": "外圈故障",
    "BPFI": "内圈故障",
    "BSF": "滚动体故障",
}


@dataclass
class DiagnosisResult:
    """一次 v2 诊断的全部中间量与结论（供报告与二次分析使用）。"""

    fs: float
    signal: np.ndarray                     # 轻预处理（去直流/去趋势）后的信号
    indicators: dict[str, float]
    health_hint: str
    spec_freqs: np.ndarray                 # FFT 幅值谱
    spec_amps: np.ndarray
    kurtogram_result: KurtogramResult | None
    demod_band: tuple[float, float]        # 实际使用的解调频带 (low, high)
    band_source: str                       # 'kurtogram' | 'manual'
    env_freqs: np.ndarray                  # 包络谱（0–800 Hz）
    env_amps: np.ndarray
    cepstrum_q: np.ndarray                 # 倒频谱
    cepstrum_v: np.ndarray
    rpm_est: float                         # 转速自估（仅记录；v2 匹配不依赖）
    rpm_fallback: bool
    rpm_fr_hz: float = 0.0                 # 自估轴频（仅记录）
    rpm_prom_ratio: float = 0.0            # 1×fr 峰突出度（仅记录）
    rpm_window: tuple[float, float] = (1100.0, 1200.0)
    combs: list[dict] = field(default_factory=list)   # A∩B 确认的梳（含被标记干扰的）
    comb_confirmed: int = 0                # A 确认的梳数
    cepstrum_confirmed: int = 0            # 其中 B 互证通过数
    n_interference: int = 0                # 被干扰防护标记的梳数
    n_clean_combs: int = 0
    n_localized: int = 0
    localization: dict | None = None       # localize_combs 的完整返回
    conf_outer: float = 0.0
    conf_inner: float = 0.0
    conf_ball: float = 0.0
    maha_dist: float | None = None         # 马氏距离（未提供基线时为 None）
    trigger: bool = False                  # 马氏触发（辅助判据）
    verdict: str = "normal"                # normal/friction/bearing_unlocalized/BPFO/BPFI/BSF
    low_conf: bool = False
    bearing_info: dict | None = None
    fault_bands: list[dict] = field(default_factory=list)  # 定位频带表
    notes: list[str] = field(default_factory=list)

    @property
    def verdict_cn(self) -> str:
        return VERDICT_CN[self.verdict]

    @property
    def confidence(self) -> float:
        return float(max(self.conf_outer, self.conf_inner, self.conf_ball))

    @property
    def max_conf(self) -> float:
        return self.confidence

    @property
    def argmax_class(self) -> str:
        confs = {"outer": self.conf_outer, "inner": self.conf_inner, "ball": self.conf_ball}
        return CLASS_CN[max(confs, key=lambda k: confs[k])]

    @property
    def level(self) -> str:
        if self.verdict == "normal":
            return "未检出轴承故障特征"
        if self.verdict == "bearing_unlocalized":
            return "检出但无法定位（建议人工读谱）"
        if self.confidence >= 0.5:
            return "明确"
        return "疑似（建议人工复核包络谱）"


def estimate_rpm(
    x_pre: np.ndarray,
    fs: float,
    rpm_window: tuple[float, float] = RPM_WINDOW_DEFAULT,
    search_hz: tuple[float, float] | None = None,
    baseline_hz: tuple[float, float] = RPM_BASELINE_HZ,
    prom_min: float = RPM_PROM_MIN,
) -> dict:
    """包络谱 1×fr 峰自估转速（仅记录用；prominence 不足回退窗中点）。

    在宽带预处理信号的包络谱上，以 baseline_hz 区段中位数为噪声地板，
    在 search_hz 窗内找局部峰；最强峰的 幅值/地板 ≥ prom_min 才接受，
    峰值频率经抛物线插值细化。search_hz 缺省按轴频窗外放
    （1100–1200 rpm ↔ (17.5, 20.5) Hz）。
    """
    flo, fhi = fr_window(rpm_window)
    if search_hz is None:
        search_hz = (flo - 0.83, fhi + 0.5)
    fallback_rpm = 0.5 * sum(rpm_window)
    info = {"rpm": fallback_rpm, "fr_hz": fallback_rpm / 60.0, "rpm_fallback": True,
            "rpm_prom_ratio": 0.0, "rpm_n_peaks": 0}
    env_f, env_a = envelope_spectrum(x_pre, fs, max_freq=120.0)
    region = (env_f >= baseline_hz[0]) & (env_f <= baseline_hz[1])
    if np.count_nonzero(region) < 8:
        return info
    floor = max(float(np.median(env_a[region])), np.finfo(float).eps)
    ridx = np.flatnonzero(region)
    pk, _ = sps.find_peaks(env_a[region])
    cands = [int(i) for i in pk if search_hz[0] <= float(env_f[ridx[i]]) <= search_hz[1]]
    info["rpm_n_peaks"] = len(cands)
    if not cands:
        return info
    best = max(cands, key=lambda i: float(env_a[ridx[i]]))
    j = int(ridx[best])
    ratio = float(env_a[j]) / floor
    info["rpm_prom_ratio"] = ratio
    if ratio < prom_min:
        return info
    fr = float(env_f[j])
    if 0 < j < env_a.size - 1:
        y0, y1, y2 = float(env_a[j - 1]), float(env_a[j]), float(env_a[j + 1])
        denom = y0 - 2.0 * y1 + y2
        if denom > 0:
            df = float(env_f[1] - env_f[0])
            fr += 0.5 * (y0 - y2) / denom * df
    info.update(rpm=60.0 * fr, fr_hz=fr, rpm_fallback=False)
    return info


def diagnose(
    x: np.ndarray,
    fs: float,
    bearing: Bearing | str,
    rpm: float | None = None,
    rpm_window: tuple[float, float] = RPM_WINDOW_DEFAULT,
    use_kurtogram: bool = True,
    manual_band: tuple[float, float] | None = None,
    kurtogram_level: int = 4,
    n_harmonics: int = 5,
    rpm_band: tuple[float, float] | None = None,
    comb_f0_range: tuple[float, float] = COMB_F0_RANGE,
    env_max_hz: float = COMB_ENV_MAX_HZ,
    calibrator: HealthBaseline | None = None,
    maha_threshold: float | None = None,
    conf_threshold: float = PRESETS[DEFAULT_PRESET],
    mains_hz: float = 50.0,
) -> DiagnosisResult:
    """执行 v2 诊断流水线。

    参数
    ----
    bearing : 轴承几何（对象或内置名，如 "FIELD7"）——定位频带由它现算。
    rpm : 真实转速（可选，仅显示参考；v2 匹配不依赖，缺省时记录自估值）。
    rpm_window : 标称转速窗（定位频带与轴频谐波防护的基准）。
    calibrator / maha_threshold : 马氏辅助判据（可选；不给则无梳样本
        只能按"未检出"处理，并在 notes 说明）。
    conf_threshold : 「轴承」判决后低置信标记的置信度门槛（三档预设）。
    """
    x = np.asarray(x, dtype=float).ravel()
    if fs <= 0:
        raise ValueError("fs 必须为正数")
    if isinstance(bearing, str):
        bearing = get_bearing(bearing)
    if x.size < 256:
        raise ValueError("信号过短（建议 ≥ 256 点）")

    notes: list[str] = []
    flo, fhi = fr_window(rpm_window)
    fr_nominal = 0.5 * (flo + fhi)

    # 1. 轻预处理（Kurtogram 需要各候选频带均有真实内容）与宽带预处理
    #    （转速自估与触发级时域特征用；默认 (500, min(7500, 0.47·fs)) Hz，
    #    16 kHz 时与合并原型一致）
    x_dt = detrend(remove_dc(x))
    if rpm_band is None:
        # 与合并原型一致的宽带预处理带：16 kHz → (500, 7500) Hz。
        # 低 fs 下退化为 (0.25·fs, 0.47·fs) 并保持 0<low<high<fs/2 校验兜底。
        rpm_band = (min(500.0, 0.25 * fs), min(7500.0, 0.47 * fs))
    if not (0 < rpm_band[0] < rpm_band[1] < fs / 2):
        raise ValueError(f"rpm_band 非法: {rpm_band} @ fs={fs}")
    x_pre = bandpass(x_dt, fs, rpm_band[0], rpm_band[1])

    # 2. 转速自估（仅记录）
    rpm_info = estimate_rpm(x_pre, fs, rpm_window=rpm_window)
    if rpm_info["rpm_fallback"]:
        notes.append(
            f"包络谱 1×fr 峰显著度不足（{rpm_info['rpm_prom_ratio']:.1f}<{RPM_PROM_MIN:g}），"
            f"转速记录回退为窗中点 {rpm_info['rpm']:.0f} rpm——v2 匹配不依赖转速，不影响判决"
        )

    # 3. 触发级时域特征（辅助判据用）
    indicators = time_domain_indicators(x_pre)
    feats = {k: float(indicators[k]) for k in FEATURE_NAMES}
    feats = {k: (v if np.isfinite(v) else 0.0) for k, v in feats.items()}

    # 4. Kurtogram 选带 → 零相位解调 → 包络谱
    if manual_band is not None:
        band = clamp_band(manual_band, fs)
        band_source, kg = "manual", None
    elif use_kurtogram:
        kg = kurtogram(x_dt, fs, max_level=kurtogram_level)
        band, band_source = clamp_band(kg.best_band, fs), "kurtogram"
        if kg.best_kurtosis < 4.0:
            notes.append(
                f"Kurtogram 最佳频带包络峭度仅 {kg.best_kurtosis:.2f}"
                "（≈3 为无冲击），信号中瞬态冲击成分弱"
            )
    else:
        raise ValueError("v2 流水线要求 Kurtogram 选带或 manual_band（全带包络不支持）")
    xb = bandpass(x_dt, fs, band[0], band[1])  # 零相位解调带通（移植项 2）
    env_f, env_a = envelope_spectrum(xb, fs, max_freq=min(env_max_hz, fs / 2.0))
    cep_q, cep_v = real_cepstrum(xb, fs, max_quefrency=0.2)

    # 5. 盲梳检：A 梳扫 → B 倒谱互证 → 干扰防护
    df = float(env_f[1] - env_f[0])
    dive_floor = flo - 2.0  # 下探下限（轴频窗下沿减 2 Hz）
    combs_a = comb_scan(env_f, env_a, f0_range=comb_f0_range,
                        n_harmonics=n_harmonics, dive_floor=dive_floor)
    combs: list[dict] = []
    n_cep = n_interf = 0
    for c in combs_a:
        ok, cep_hits = cepstrum_confirm(cep_q, cep_v, c["f0"])
        c["cep_confirmed"] = bool(ok)
        c["cep_hits"] = int(cep_hits)
        if not ok:
            continue
        n_cep += 1
        if c["low_f0"]:
            c["suspected_interference"] = True
            c["guard_reason"] = f"低频梳(f0<{comb_f0_range[0]:g}Hz,疑轴频族)"
        else:
            reason = mark_interference(c["f0"], df, fr_nominal=fr_nominal, mains=mains_hz)
            c["suspected_interference"] = bool(reason)
            c["guard_reason"] = reason
        if c["suspected_interference"]:
            n_interf += 1
            notes.append(f"梳 {c['f0']:.2f} Hz 命中干扰防护（{c['guard_reason']}），已从轴承判决排除")
        combs.append(c)
    clean = [c for c in combs if not c["suspected_interference"]]

    # 6. 定位（频带落入 + 补救 + 理论复核 ±4%）
    loc = localize_combs(env_f, env_a, [c["f0"] for c in clean], bearing, rpm_window,
                         n_harmonics=n_harmonics)
    for c, per in zip(clean, loc["per_comb"]):
        c["loc"] = per

    # 7. 马氏辅助判据（可选）
    maha_dist = None
    trigger = False
    if calibrator is not None:
        maha_dist = float(calibrator.distance(feats))
        if maha_threshold is not None:
            trigger = bool(maha_dist >= maha_threshold)
    else:
        notes.append("未提供健康基线：马氏辅助判据未启用，无梳样本按未检出处理"
                     "（域内标定见 baseline.calibrate_oof）")

    # 8. 判决
    confs = {"outer": loc["conf_outer"], "inner": loc["conf_inner"], "ball": loc["conf_ball"]}
    mx = float(max(confs.values()))
    if clean:
        if loc["n_localized"] > 0:
            best_cls = max(confs, key=lambda k: confs[k])
            verdict = {"outer": "BPFO", "inner": "BPFI", "ball": "BSF"}[best_cls]
            low_conf = bool(mx < conf_threshold)
            fr_hyp = next((p["fr_hyp"] for p in loc["per_comb"] if p["localized"]), None)
            notes.append(
                f"检出 {len(clean)} 个有效梳（{', '.join(f'{c['f0']:.1f}' for c in clean)} Hz），"
                f"定位为{CLASS_CN[best_cls]}（置信度 {mx:.2f}）"
                + (f"，反演轴频 {fr_hyp * 60:.0f} rpm" if fr_hyp else "")
            )
        else:
            verdict = "bearing_unlocalized"
            low_conf = False
            notes.append(
                f"检出 {len(clean)} 个有效梳（{', '.join(f'{c['f0']:.1f}' for c in clean)} Hz），"
                "但频带落入与补救路径均无法定位——轴承故障特征存在但部位未定，"
                "建议人工读谱（可能为转速出窗或其他周期振源）"
            )
    elif trigger:
        verdict = "friction"
        low_conf = bool(maha_dist < 2.0 * maha_threshold)
        notes.append(
            f"未检出轴承梳特征，但马氏距离 {maha_dist:.1f} ≥ 阈值 {maha_threshold:.1f}"
            "（辅助判据触发）——判异常-非轴承(摩擦)"
        )
    else:
        verdict = "normal"
        low_conf = False

    spec_f, spec_a = amplitude_spectrum(x_dt, fs)
    return DiagnosisResult(
        fs=float(fs),
        signal=x_dt,
        indicators=indicators,
        health_hint=health_hint(indicators),
        spec_freqs=spec_f,
        spec_amps=spec_a,
        kurtogram_result=kg,
        demod_band=(float(band[0]), float(band[1])),
        band_source=band_source,
        env_freqs=env_f,
        env_amps=env_a,
        cepstrum_q=cep_q,
        cepstrum_v=cep_v,
        rpm_est=float(rpm_info["rpm"]),
        rpm_fallback=bool(rpm_info["rpm_fallback"]),
        rpm_fr_hz=float(rpm_info["fr_hz"]),
        rpm_prom_ratio=float(rpm_info["rpm_prom_ratio"]),
        rpm_window=(float(rpm_window[0]), float(rpm_window[1])),
        combs=combs,
        comb_confirmed=len(combs_a),
        cepstrum_confirmed=n_cep,
        n_interference=n_interf,
        n_clean_combs=len(clean),
        n_localized=loc["n_localized"],
        localization=loc,
        conf_outer=float(confs["outer"]),
        conf_inner=float(confs["inner"]),
        conf_ball=float(confs["ball"]),
        maha_dist=maha_dist,
        trigger=trigger,
        verdict=verdict,
        low_conf=low_conf,
        bearing_info=bearing_to_dict(bearing),
        fault_bands=class_bands(bearing, rpm_window),
        notes=notes,
    )

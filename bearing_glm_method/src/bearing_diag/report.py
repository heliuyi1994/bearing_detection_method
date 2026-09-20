"""诊断报告输出：四联图 PNG + 中文文本报告 + JSON。

图片布局：
    (1) 时域波形（前 100 ms）+ 时域指标表格
    (2) FFT 幅值谱（解调频带高亮）
    (3) Kurtogram 频带-峭度图（若启用）
    (4) 包络谱 + 判定故障的特征频率谐波标注
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无显示环境安全
import matplotlib.pyplot as plt
import numpy as np

from .diagnose import DiagnosisResult
from .fault_freqs import FAULT_NAMES_CN

plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC", "Noto Sans CJK HK", "Noto Sans CJK TC",
    "WenQuanYi Micro Hei", "Droid Sans Fallback", "AR PL UKai CN",
    "SimHei", "Microsoft YaHei", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

_INDICATOR_LABELS = {
    "rms": "RMS 均方根",
    "peak": "峰值 |x|max",
    "peak_to_peak": "峰峰值",
    "kurtosis": "峭度",
    "skewness": "偏度",
    "crest_factor": "峰值因子",
    "impulse_factor": "脉冲因子",
    "shape_factor": "波形因子",
    "clearance_factor": "裕度因子",
}
# 图内表格用英文短标签（等宽字体对齐且无 CJK 字形缺失问题），中文含义见文本报告
_INDICATOR_PLOT_LABELS = {
    "rms": "RMS", "peak": "Peak", "peak_to_peak": "Peak-to-Peak",
    "kurtosis": "Kurtosis", "skewness": "Skewness",
    "crest_factor": "Crest Factor", "impulse_factor": "Impulse Factor",
    "shape_factor": "Shape Factor", "clearance_factor": "Clearance Factor",
}


def plot_report(result: DiagnosisResult, out_path: str | Path) -> Path:
    """生成四联诊断图 PNG。"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle(
        f"轴承诊断报告  |  判定: {result.verdict_cn}（置信度 {result.confidence:.2f}）",
        fontsize=14,
    )

    # (1) 时域波形 + 指标表
    ax = axes[0, 0]
    span = min(int(0.1 * result.fs), result.signal.size)
    t = np.arange(span) / result.fs
    ax.plot(t * 1000, result.signal[:span], lw=0.6, color="steelblue")
    ax.set_xlabel("时间 (ms)")
    ax.set_ylabel("加速度")
    ax.set_title(f"时域波形（前 {span / result.fs * 1000:.0f} ms）")
    lines = [f"{_INDICATOR_PLOT_LABELS[k]:16s} {v:10.4f}" for k, v in result.indicators.items()]
    ax.text(
        0.98, 0.97, "\n".join(lines), transform=ax.transAxes,
        ha="right", va="top", family="monospace", fontsize=9,
        bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.9),
    )

    # (2) FFT 幅值谱
    ax = axes[0, 1]
    ax.plot(result.spec_freqs, result.spec_amps, lw=0.6, color="dimgray")
    lo, hi = result.demod_band
    ax.axvspan(lo, hi, color="orange", alpha=0.25,
               label=f"解调频带 [{lo:.0f}, {hi:.0f}] Hz（{result.band_source}）")
    ax.set_xlabel("频率 (Hz)")
    ax.set_ylabel("幅值")
    ax.set_title("FFT 幅值谱")
    ax.legend(fontsize=8)

    # (3) Kurtogram 图
    ax = axes[1, 0]
    kg = result.kurtogram_result
    if kg is not None:
        cmap = plt.get_cmap("viridis")
        all_k = [k for row in kg.levels for (_, _, k) in row]
        kmin, kmax = min(all_k), max(all_k)
        # 条带高度与频带宽度成比例：浅层宽带为高条，深层窄带叠画其上，
        # 形成层级式峭度热图（按层序绘制，深层精细频带最后画在最上）
        for row in kg.levels:
            for (bl, bh, k) in row:
                color = cmap((k - kmin) / max(kmax - kmin, 1e-9))
                ax.barh(bl, bh - bl, left=bl, height=0.75 * (bh - bl),
                        color=color, alpha=0.8, edgecolor="none")
        lo, hi = result.demod_band
        ax.barh(lo, hi - lo, left=lo, height=0.75 * (hi - lo), fill=False,
                edgecolor="red", lw=2.5, zorder=10,
                label=f"选中频带（峭度 {kg.best_kurtosis:.1f}）")
        norm = plt.Normalize(kmin, kmax)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        fig.colorbar(sm, ax=ax, label="包络峭度")
        ax.set_ylim(bottom=0)
        ax.set_xlabel("频率 (Hz)")
        ax.set_ylabel("频带下边界 (Hz)")
        ax.set_title("Kurtogram：各频带包络峭度")
        ax.legend(fontsize=8, loc="upper right")
    else:
        ax.text(0.5, 0.5, f"未启用 Kurtogram（频带来源: {result.band_source}）",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Kurtogram")

    # (4) 包络谱 + 谐波标注
    ax = axes[1, 1]
    ax.plot(result.env_freqs, result.env_amps, lw=0.8, color="steelblue")
    ax.set_xlabel("频率 (Hz)")
    ax.set_ylabel("包络幅值")
    ax.set_title(f"包络谱（判为 {result.verdict_cn}，置信度 {result.confidence:.2f}）")
    m = result.match
    if m is not None:
        for cs in m.candidates:
            if not cs.harmonics:
                continue
            is_best = cs.fault == m.verdict
            color = "red" if is_best else "gray"
            alpha = 1.0 if is_best else 0.55
            for h in cs.harmonics:
                if h["target_hz"] > result.env_freqs[-1]:
                    continue
                ax.axvline(h["target_hz"], color=color, lw=0.8, alpha=alpha,
                           ls="-" if is_best else ":")
            ax.axvline(
                float(cs.freq), color=color, lw=1.8, alpha=alpha,
                label=f"{cs.fault} {cs.freq:.1f} Hz（得分 {cs.score:.2f}）",
            )
        ax.legend(fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def build_text_report(result: DiagnosisResult, source_desc: str = "") -> str:
    """生成中文文本诊断报告。"""
    lines: list[str] = []
    lines.append("=" * 62)
    lines.append("滚动轴承故障诊断报告（传统信号处理全流程）")
    lines.append("=" * 62)
    if source_desc:
        lines.append(f"数据来源      : {source_desc}")
    lines.append(f"采样率        : {result.fs:.0f} Hz")
    lines.append(f"轴频 fr       : {result.shaft_freq:.3f} Hz "
                 f"({result.shaft_freq * 60:.0f} rpm)")
    lines.append(f"信号长度      : {result.signal.size} 点 "
                 f"({result.signal.size / result.fs:.2f} s)")
    if result.bearing_info:
        b = result.bearing_info
        lines.append(f"轴承参数      : {b['name']}（n={b['n_balls']}, "
                     f"D={b['pitch_diameter']} mm, d={b['ball_diameter']} mm, "
                     f"α={b['contact_angle_deg']}°）")
    lines.append("")
    lines.append("—— 特征频率理论值 ——")
    for k, v in result.fault_freqs_used.items():
        lines.append(f"  {k:5s} ({FAULT_NAMES_CN[k]}) : {v:8.2f} Hz")
    lines.append("")
    lines.append("—— 时域统计指标 ——")
    for k, v in result.indicators.items():
        lines.append(f"  {_INDICATOR_LABELS[k]:14s}: {v:10.4f}")
    lines.append(f"  指标解读: {result.health_hint}")
    lines.append("")
    lines.append("—— 解调频带选择 ——")
    lo, hi = result.demod_band
    if result.kurtogram_result is not None:
        lines.append(f"  Kurtogram 选中 [{lo:.0f}, {hi:.0f}] Hz"
                     f"（中心 {result.kurtogram_result.center:.0f} Hz, "
                     f"包络峭度 {result.kurtogram_result.best_kurtosis:.2f}）")
    else:
        lines.append(f"  频带 [{lo:.0f}, {hi:.0f}] Hz（来源: {result.band_source}）")
    lines.append("")
    lines.append("—— 谐波匹配得分 ——")
    if result.match is not None:
        for cs in result.match.candidates:
            tag = " ← 判定" if cs.fault == result.match.verdict else ""
            conf = "（继承谐波已打折）" if cs.confounded else ""
            lines.append(f"  {cs.fault:5s} {cs.freq:8.2f} Hz  "
                         f"得分 {cs.score:.3f}  命中 {cs.hits} 次{conf}{tag}")
    lines.append("")
    lines.append("—— 诊断结论 ——")
    lines.append(f"  判定    : {result.verdict_cn}")
    lines.append(f"  置信度  : {result.confidence:.2f}（{result.level}）")
    if result.notes:
        lines.append("")
        lines.append("—— 提示与注意事项 ——")
        for i, note in enumerate(result.notes, 1):
            lines.append(f"  {i}. {note}")
    lines.append("=" * 62)
    return "\n".join(lines)


def build_json_payload(result: DiagnosisResult, source_desc: str = "") -> dict:
    """机器可读结果（供 AI 助手/上游系统解析）。"""
    m = result.match
    return {
        "source": source_desc,
        "fs": result.fs,
        "shaft_freq": result.shaft_freq,
        "bearing": result.bearing_info,
        "fault_freqs": result.fault_freqs_used,
        "indicators": result.indicators,
        "demod_band": list(result.demod_band),
        "band_source": result.band_source,
        "kurtogram_best_kurtosis": (
            result.kurtogram_result.best_kurtosis
            if result.kurtogram_result else None
        ),
        "match": {
            "verdict": m.verdict,
            "verdict_cn": m.verdict_cn,
            "confidence": round(m.confidence, 3),
            "level": m.level,
            "candidates": [
                {
                    "fault": cs.fault, "freq_hz": round(cs.freq, 3),
                    "score": round(cs.score, 3), "hits": cs.hits,
                    "confounded": cs.confounded,
                    "harmonics": cs.harmonics,
                }
                for cs in m.candidates
            ],
        } if m else None,
        "notes": result.notes,
    }


def write_reports(
    result: DiagnosisResult, out_dir: str | Path, source_desc: str = ""
) -> dict[str, Path]:
    """输出 PNG + TXT + JSON 三件套，返回路径字典。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    png = plot_report(result, out / "diagnosis_report.png")
    txt_path = out / "diagnosis_report.txt"
    txt_path.write_text(build_text_report(result, source_desc), encoding="utf-8")
    json_path = out / "diagnosis_report.json"
    json_path.write_text(
        json.dumps(build_json_payload(result, source_desc), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"png": png, "txt": txt_path, "json": json_path}

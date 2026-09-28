# -*- coding: utf-8 -*-
"""诊断报告输出：四联图 PNG + 中文文本报告 + JSON（v2 适配）。

图片布局：
    (1) 时域波形（前 100 ms）+ 时域指标表格
    (2) FFT 幅值谱（解调频带高亮）
    (3) Kurtogram 频带-峭度图（若启用）
    (4) 包络谱 + 四个定位频带底纹 + 检出梳谐波标注（干扰梳灰色虚线）
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无显示环境安全
import matplotlib.pyplot as plt
import numpy as np

from .diagnose import DiagnosisResult

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

_BAND_COLORS = {"滚珠(BSF基频带)": "#DD8452", "外圈(BPFO带)": "#C44E52",
                "滚珠(2×BSF主峰带)": "#DD8452", "内圈(BPFI带)": "#55A868"}


def plot_report(result: DiagnosisResult, out_path: str | Path) -> Path:
    """生成四联诊断图 PNG（v2 版式）。"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle(
        f"轴承诊断报告（v2 盲梳检） |  判定: {result.verdict_cn}"
        f"（置信度 {result.confidence:.2f}）",
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

    # (4) 包络谱 + 定位频带底纹 + 梳谐波标注
    ax = axes[1, 1]
    ax.plot(result.env_freqs, result.env_amps, lw=0.8, color="steelblue")
    for b in result.fault_bands:
        ax.axvspan(b["lo"], b["hi"], color=_BAND_COLORS.get(b["label"], "gray"),
                   alpha=0.13, label=b["label"])
    ymax = float(result.env_amps.max()) if result.env_amps.size else 1.0
    for c in result.combs:
        marked = c["suspected_interference"]
        color = "gray" if marked else "red"
        ls = ":" if marked else "-"
        alpha = 0.5 if marked else 0.85
        for h in c.get("covered", []):
            if h <= result.env_freqs[-1]:
                ax.axvline(h, color=color, lw=0.7, alpha=alpha * 0.6, ls=ls)
        tag = "（干扰已排除）" if marked else ""
        ax.axvline(c["f0"], color=color, lw=1.6, alpha=alpha, ls=ls,
                   label=f"梳 {c['f0']:.1f} Hz（得分 {c['score']:.2f}）{tag}")
    ax.set_ylim(0, ymax * 1.15)
    ax.set_xlabel("频率 (Hz)")
    ax.set_ylabel("包络幅值")
    ax.set_title(f"包络谱与检出梳（判为 {result.verdict_cn}）")
    if result.combs or result.fault_bands:
        ax.legend(fontsize=7.5, loc="upper right")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def build_text_report(result: DiagnosisResult, source_desc: str = "") -> str:
    """生成中文文本诊断报告（v2 版式）。"""
    lines: list[str] = []
    lines.append("=" * 62)
    lines.append("滚动轴承故障诊断报告（统一包 v2 盲梳检规则）")
    lines.append("=" * 62)
    if source_desc:
        lines.append(f"数据来源      : {source_desc}")
    lines.append(f"采样率        : {result.fs:.0f} Hz")
    lines.append(f"信号长度      : {result.signal.size} 点 "
                 f"({result.signal.size / result.fs:.2f} s)")
    lines.append(f"转速（记录）  : {result.rpm_est:.0f} rpm"
                 + ("（自估回退）" if result.rpm_fallback else "（自估）")
                 + "——v2 匹配不依赖转速")
    if result.bearing_info:
        b = result.bearing_info
        lines.append(f"轴承参数      : {b['name']}（n={b['n_balls']}, "
                     f"D={b['pitch_diameter']} mm, d={b['ball_diameter']} mm, "
                     f"α={b['contact_angle_deg']}°）")
    lines.append("")
    lines.append("—— 定位频带（转速窗 × 几何现算） ——")
    for band in result.fault_bands:
        lines.append(f"  {band['label']:16s}: [{band['lo']:6.2f}, {band['hi']:6.2f}] Hz")
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
    lines.append("—— 盲梳检（不依赖理论频率） ——")
    lines.append(f"  梳确认: A 梳扫 {result.comb_confirmed} 个 → "
                 f"B 倒谱互证 {result.cepstrum_confirmed} 个 → "
                 f"干扰防护标记 {result.n_interference} 个 → "
                 f"有效 {result.n_clean_combs} 个")
    if result.combs:
        for c in result.combs:
            mark = f"（干扰: {c['guard_reason']}）" if c["suspected_interference"] else ""
            loc = c.get("loc") or {}
            loc_s = ""
            if loc.get("localized"):
                from .localize import CLASS_CN as _CN
                loc_s = f" → 定位 {_CN[loc['cls']]}（fr_hyp={loc['fr_hyp'] * 60:.0f} rpm）"
            lines.append(f"    梳 {c['f0']:7.2f} Hz  得分 {c['score']:.3f}  "
                         f"谐波命中 {c['hits']}  倒谱命中 {c['cep_hits']}{mark}{loc_s}")
    else:
        lines.append("    （未检出显著梳）")
    lines.append("")
    lines.append("—— 三类置信度（定位合成，不归一） ——")
    lines.append(f"  外圈 BPFO : {result.conf_outer:.3f}")
    lines.append(f"  内圈 BPFI : {result.conf_inner:.3f}")
    lines.append(f"  滚珠 BSF  : {result.conf_ball:.3f}")
    lines.append("")
    if result.maha_dist is not None:
        lines.append("—— 马氏距离健康基线（辅助判据） ——")
        lines.append(f"  距离 {result.maha_dist:.2f}，触发: {'是' if result.trigger else '否'}")
        lines.append("")
    lines.append("—— 诊断结论 ——")
    lines.append(f"  判定    : {result.verdict_cn}")
    lines.append(f"  置信度  : {result.confidence:.2f}（{result.level}）")
    if result.low_conf:
        lines.append("  低置信  : 是（触发/定位证据皆弱，建议人工复核）")
    if result.notes:
        lines.append("")
        lines.append("—— 提示与注意事项 ——")
        for i, note in enumerate(result.notes, 1):
            lines.append(f"  {i}. {note}")
    lines.append("=" * 62)
    return "\n".join(lines)


def build_json_payload(result: DiagnosisResult, source_desc: str = "") -> dict:
    """机器可读结果（供 AI 助手/上游系统解析）。"""
    return {
        "source": source_desc,
        "rules": "v2",
        "fs": result.fs,
        "rpm_est": result.rpm_est,
        "rpm_fallback": result.rpm_fallback,
        "rpm_window": list(result.rpm_window),
        "bearing": result.bearing_info,
        "fault_bands": result.fault_bands,
        "indicators": result.indicators,
        "demod_band": list(result.demod_band),
        "band_source": result.band_source,
        "kurtogram_best_kurtosis": (
            result.kurtogram_result.best_kurtosis
            if result.kurtogram_result else None
        ),
        "combs": {
            "confirmed": result.comb_confirmed,
            "cepstrum_confirmed": result.cepstrum_confirmed,
            "n_interference": result.n_interference,
            "n_clean": result.n_clean_combs,
            "n_localized": result.n_localized,
            "detail": [
                {"f0": round(c["f0"], 3), "score": round(c["score"], 4),
                 "hits": c["hits"], "cep_hits": c["cep_hits"],
                 "suspected_interference": c["suspected_interference"],
                 "guard_reason": c.get("guard_reason", ""),
                 "band": (c.get("loc") or {}).get("band", ""),
                 "cls": (c.get("loc") or {}).get("cls"),
                 "fr_hyp": (c.get("loc") or {}).get("fr_hyp")}
                for c in result.combs
            ],
        },
        "confidence": {
            "outer": round(result.conf_outer, 4),
            "inner": round(result.conf_inner, 4),
            "ball": round(result.conf_ball, 4),
            "argmax_class": result.argmax_class,
        },
        "maha_dist": result.maha_dist,
        "trigger": result.trigger,
        "verdict": result.verdict,
        "verdict_cn": result.verdict_cn,
        "low_conf": result.low_conf,
        "level": result.level,
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

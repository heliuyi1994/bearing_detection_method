"""matplotlib 绘图封装：CLI 与 Notebook 共用的统一绘图接口。"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

# 中文字体回退列表（按常见 Linux/macOS/Windows 顺序），缺字体时自动退回 DejaVu
matplotlib.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC",
    "Droid Sans Fallback",
    "WenQuanYi Zen Hei",
    "Microsoft YaHei",
    "SimHei",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False

# 故障类型的中文标注与绘图颜色
FAULT_LABELS = {
    "healthy": "健康",
    "inner": "内圈故障",
    "outer": "外圈故障",
    "ball": "滚动体故障",
    "cage": "保持架故障",
}
FAULT_COLORS = {"inner": "tab:red", "outer": "tab:blue", "ball": "tab:green", "cage": "tab:purple"}


def plot_waveform(
    t: NDArray[np.float64],
    x: NDArray[np.float64],
    title: str = "振动信号时域波形",
    t_range: tuple[float, float] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """时域波形图，可用 t_range 截取局部放大。"""
    fig, ax = plt.subplots(figsize=(10, 3.5))
    mask = np.ones(t.size, dtype=bool)
    if t_range is not None:
        mask = (t >= t_range[0]) & (t <= t_range[1])
    ax.plot(t[mask], x[mask], lw=0.6, color="0.2")
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("幅值")
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def plot_spectrum(
    freqs: NDArray[np.float64],
    amps: NDArray[np.float64],
    char_freqs: dict[str, float] | None = None,
    title: str = "频谱",
    xlim: tuple[float, float] | None = None,
    harmonics: int = 3,
) -> tuple[plt.Figure, plt.Axes]:
    """频谱/包络谱图，可选标注故障特征频率及其谐波竖线。"""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(freqs, amps, lw=0.8, color="0.2")
    if char_freqs:
        for name, f0 in char_freqs.items():
            color = FAULT_COLORS.get(name, "0.5")
            for h in range(1, harmonics + 1):
                f = h * f0
                if xlim and f > xlim[1]:
                    break
                ax.axvline(f, color=color, ls="--", lw=0.8, alpha=0.7)
            ax.axvline(f0, color=color, ls="--", lw=0.8, alpha=0.7,
                       label=f"{FAULT_LABELS.get(name, name)} {f0:.1f} Hz")
        ax.legend(loc="upper right", fontsize=8)
    if xlim:
        ax.set_xlim(*xlim)
    ax.set_xlabel("频率 (Hz)")
    ax.set_ylabel("幅值")
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def plot_order_spectrum(
    orders: NDArray[np.float64],
    amps: NDArray[np.float64],
    char_orders: dict[str, float] | None = None,
    xlim: tuple[float, float] | None = None,
    title: str = "阶次谱",
) -> tuple[plt.Figure, plt.Axes]:
    """阶次谱图，可选标注理论故障阶次。"""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(orders, amps, lw=0.8, color="0.2")
    if char_orders:
        for name, o0 in char_orders.items():
            color = FAULT_COLORS.get(name, "0.5")
            ax.axvline(o0, color=color, ls="--", lw=0.9,
                       label=f"{FAULT_LABELS.get(name, name)} {o0:.3f} 阶")
        ax.legend(loc="upper right", fontsize=8)
    if xlim:
        ax.set_xlim(*xlim)
    ax.set_xlabel("阶次 (次/转)")
    ax.set_ylabel("幅值")
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def plot_timefreq(
    t: NDArray[np.float64],
    f: NDArray[np.float64],
    z: NDArray[np.float64],
    title: str = "时频谱",
    ylim: tuple[float, float] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """时频能量图（STFT 或 CWT 幅值）。"""
    fig, ax = plt.subplots(figsize=(10, 4))
    mesh = ax.pcolormesh(t, f, z, shading="auto", cmap="viridis")
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("频率 (Hz)")
    ax.set_title(title)
    fig.colorbar(mesh, ax=ax, label="幅值")
    fig.tight_layout()
    return fig, ax


def plot_feature_bars(
    samples: dict[str, dict[str, float]],
    title: str = "时域特征对比",
) -> tuple[plt.Figure, plt.Axes]:
    """多样本时域特征对比柱状图（各特征按健康样本归一化）。"""
    names = list(next(iter(samples.values())).keys())
    labels = list(samples.keys())
    base = np.array([samples[labels[0]][n] for n in names], dtype=float)
    base[base == 0] = 1.0
    x_pos = np.arange(len(names))
    width = 0.8 / len(labels)
    fig, ax = plt.subplots(figsize=(11, 4))
    for i, label in enumerate(labels):
        vals = np.array([samples[label][n] for n in names], dtype=float) / base
        ax.bar(x_pos + i * width, vals, width=width, label=label)
    ax.set_xticks(x_pos + width * (len(labels) - 1) / 2, names, rotation=30, ha="right")
    ax.set_ylabel(f"相对 {labels[0]} 的倍数")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, ax


def plot_envelope_report(report) -> tuple[plt.Figure, plt.Axes]:
    """按诊断报告绘制包络谱并标注理论特征频率与命中峰。"""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(report.env_freqs, report.env_amps, lw=0.8, color="0.2")
    f_max_view = max(report.char_freqs.values()) * 3.5
    for name, f0 in report.char_freqs.items():
        color = FAULT_COLORS.get(name, "0.5")
        for h in range(1, 4):
            f = h * f0
            if f > f_max_view:
                break
            ax.axvline(f, color=color, ls="--", lw=0.8, alpha=0.6)
        ax.axvline(f0, color=color, ls="--", lw=0.9,
                   label=f"{FAULT_LABELS.get(name, name)} {f0:.1f} Hz")
        for h, f_peak, a_peak in report.peaks.get(name, []):
            if f_peak <= f_max_view:
                ax.plot(f_peak, a_peak, "v", color=color, ms=7)
    ax.set_xlim(0, f_max_view)
    ax.set_xlabel("频率 (Hz)")
    ax.set_ylabel("包络幅值")
    ax.set_title(f"包络谱（解调频带 {report.band[0]:.0f}–{report.band[1]:.0f} Hz，▼ 为命中峰）")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig, ax


def plot_scores(
    scores: dict[str, float],
    verdict: str,
    title: str = "故障类型匹配得分",
) -> tuple[plt.Figure, plt.Axes]:
    """谱峰匹配得分柱状图，最高分（诊断结论）高亮。"""
    names = list(scores.keys())
    vals = [scores[n] for n in names]
    colors = ["tab:red" if n == verdict else "0.6" for n in names]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar([FAULT_LABELS.get(n, n) for n in names], vals, color=colors)
    ax.set_ylabel("得分（命中峰/噪声地板倍数）")
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax

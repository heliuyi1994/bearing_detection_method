# -*- coding: utf-8 -*-
"""EDA：labeled_dataset_4class 数据质量、时域/频域统计、选带分布、转速证据。

输出：reports/eda_summary.json、reports/eda_per_file.json、reports/figures/eda_*.png
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict

import numpy as np

import common

common.setup_matplotlib()
import matplotlib.pyplot as plt

FIG = common.FIG_DIR
FIG.mkdir(parents=True, exist_ok=True)

CLASS_ORDER = ["正常", "外圈", "内圈", "滚珠"]
CLASS_COLOR = {"正常": "#2ca02c", "外圈": "#1f77b4", "内圈": "#d62728", "滚珠": "#ff7f0e"}


def fr_peak_from_envelope(x: np.ndarray, fs: float, band: tuple[float, float]) -> float:
    """包络谱 [18.0, 20.5] Hz 窗内最强峰的频率（1×fr 证据）。"""
    from bearing_diag.indicators import detrend
    from bearing_diag.kurtogram import bandpass_signal
    from bearing_diag.spectrum import envelope_spectrum

    xb = bandpass_signal(detrend(x), fs, *band)
    f, a = envelope_spectrum(xb, fs, max_freq=25.0)
    m = (f >= 18.0) & (f <= 20.5)
    if not m.any():
        return float("nan")
    return float(f[m][np.argmax(a[m])])


def main() -> None:
    t_start = time.time()
    rows = common.load_manifest()
    per_file = []

    for i, r in enumerate(rows, 1):
        x, clip = common.read_wav(r["relpath"])
        band, bk = common.kurtogram_band(x, common.FS)
        fr_pk = fr_peak_from_envelope(x, common.FS, band)
        from bearing_diag.indicators import detrend, time_domain_indicators

        ind = time_domain_indicators(detrend(x))
        per_file.append({
            "file_id": r["file_id"],
            "relpath": r["relpath"],
            "class": r["class"],
            "remark": r["remark"],
            "batch": r["batch"],
            "session": r["session"],
            "n_samples": int(x.size),
            "duration_s": r["duration_s"],
            "clip_ratio": clip,
            "rms": ind["rms"],
            "kurtosis": ind["kurtosis"],
            "crest_factor": ind["crest_factor"],
            "clearance_factor": ind["clearance_factor"],
            "band_low": band[0],
            "band_high": band[1],
            "band_center": (band[0] + band[1]) / 2,
            "band_width": band[1] - band[0],
            "best_kurtosis": bk,
            "fr_peak_hz": fr_pk,
            "fr_est_hz": r["fr_est_hz"],
        })
        if i % 20 == 0 or i == len(rows):
            print(f"  EDA 进度 {i}/{len(rows)} ({time.time() - t_start:.0f}s)")

    # ---------- 汇总统计 ----------
    summary: dict = {"n_files": len(per_file)}
    summary["class_counts"] = dict(Counter(p["class"] for p in per_file))
    summary["batch_counts"] = dict(Counter(p["batch"] for p in per_file))
    summary["session_counts"] = dict(Counter(p["session"] for p in per_file))
    summary["class_x_session"] = {
        f"{c}/{s}": n
        for (c, s), n in Counter((p["class"], p["session"]) for p in per_file).items()
    }
    fr_main = common.RPM_MAIN / 60.0
    by_class = defaultdict(list)
    for p in per_file:
        by_class[p["class"]].append(p)

    def stats(vals):
        v = np.array([x for x in vals if x == x])
        if v.size == 0:
            return None
        return {"min": float(v.min()), "median": float(np.median(v)),
                "max": float(v.max()), "mean": float(v.mean())}

    summary["per_class"] = {}
    for c in CLASS_ORDER:
        ps = by_class[c]
        dev_frpk = [(p["fr_peak_hz"] - fr_main) / fr_main * 100 for p in ps]
        dev_frest = [(p["fr_est_hz"] - fr_main) / fr_main * 100
                     for p in ps if p["fr_est_hz"] == p["fr_est_hz"]]
        summary["per_class"][c] = {
            "n": len(ps),
            "clip_ratio": stats([p["clip_ratio"] for p in ps]),
            "n_clipped": sum(1 for p in ps if p["clip_ratio"] > 0),
            "rms": stats([p["rms"] for p in ps]),
            "kurtosis": stats([p["kurtosis"] for p in ps]),
            "best_kurtosis": stats([p["best_kurtosis"] for p in ps]),
            "band_center": stats([p["band_center"] for p in ps]),
            "band_width": stats([p["band_width"] for p in ps]),
            "fr_peak_dev_vs_1150_pct": stats(dev_frpk),
            "fr_est_dev_vs_1150_pct": stats(dev_frest),
        }

    # 转速证据总表：fr_peak 与 fr_est 的互差
    both = [p for p in per_file if p["fr_est_hz"] == p["fr_est_hz"] and p["fr_peak_hz"] == p["fr_peak_hz"]]
    diff = [abs(p["fr_peak_hz"] - p["fr_est_hz"]) / p["fr_est_hz"] * 100 for p in both]
    summary["fr_check"] = {
        "n_with_fr_est": len(both),
        "abs_dev_frpeak_vs_frest_pct": stats(diff),
        "fr_window_nominal_hz": {"low": common.RPM_WINDOW[0] / 60, "high": common.RPM_WINDOW[1] / 60,
                                 "main_assumption": fr_main},
        "max_abs_dev_frpeak_vs_main_pct": max(
            abs((p["fr_peak_hz"] - fr_main) / fr_main * 100) for p in per_file if p["fr_peak_hz"] == p["fr_peak_hz"]),
    }
    common.save_json(summary, common.REPORTS_DIR / "eda_summary.json")
    common.save_json(per_file, common.REPORTS_DIR / "eda_per_file.json")

    # ---------- 图 1：数据概览 ----------
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    ax = axes[0]
    counts = [summary["class_counts"].get(c, 0) for c in CLASS_ORDER]
    ax.bar(CLASS_ORDER, counts, color=[CLASS_COLOR[c] for c in CLASS_ORDER])
    for i, n in enumerate(counts):
        ax.text(i, n + 0.3, str(n), ha="center")
    ax.set_title("类别分布（共 81 条）")
    ax.set_ylabel("样本数")

    ax = axes[1]
    batches = ["9-16", "9-17", "9-18"]
    bottom = np.zeros(len(batches))
    for c in CLASS_ORDER:
        vals = np.array([sum(1 for p in per_file if p["batch"] == b and p["class"] == c) for b in batches])
        ax.bar(batches, vals, bottom=bottom, color=CLASS_COLOR[c], label=c)
        bottom += vals
    ax.set_title("批次 × 类别")
    ax.legend(fontsize=8)

    ax = axes[2]
    sessions = ["morning", "afternoon"]
    bottom = np.zeros(len(sessions))
    for c in CLASS_ORDER:
        vals = np.array([sum(1 for p in per_file if p["session"] == s and p["class"] == c) for s in sessions])
        ax.bar(sessions, vals, bottom=bottom, color=CLASS_COLOR[c], label=c)
        bottom += vals
    ax.set_title("session × 类别")
    fig.tight_layout()
    fig.savefig(FIG / "eda_overview.png")
    plt.close(fig)

    # ---------- 图 2：时域质量（RMS / 峭度 / 削波 / 包络峭度） ----------
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4))
    for ax, key, title, ylim in zip(
        axes,
        ["rms", "kurtosis", "clip_ratio", "best_kurtosis"],
        ["RMS（归一化幅值）", "时域峭度", "削波比例", "Kurtogram 最佳包络峭度"],
        [None, None, None, None],
    ):
        data = [[p[key] for p in by_class[c]] for c in CLASS_ORDER]
        bp = ax.boxplot(data, tick_labels=CLASS_ORDER, patch_artist=True)
        for patch, c in zip(bp["boxes"], CLASS_ORDER):
            patch.set_facecolor(CLASS_COLOR[c])
            patch.set_alpha(0.6)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        if key == "clip_ratio":
            ax.set_yscale("symlog", linthresh=1e-4)
    fig.suptitle("时域与选带质量指标（按类）——注意外圈类削波与高 RMS", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "eda_time_domain.png", bbox_inches="tight")
    plt.close(fig)

    # ---------- 图 3：按类中位 Welch 谱 ----------
    from scipy.signal import welch

    fig, ax = plt.subplots(figsize=(9, 4.2))
    for c in CLASS_ORDER:
        psds = []
        for p in by_class[c]:
            x, _ = common.read_wav(p["relpath"])
            f, pxx = welch(x, fs=common.FS, nperseg=8192)
            psds.append(pxx)
        ax.loglog(f[1:], np.median(psds, axis=0)[1:], color=CLASS_COLOR[c], label=c)
    ax.axvspan(common.FS / 6, 0.45 * common.FS, color="gray", alpha=0.15,
               label="kimi 原生固定带")
    ax.set_xlabel("频率 (Hz)")
    ax.set_ylabel("PSD 中位数")
    ax.set_title("按类 Welch 谱中位数（真实共振区 vs kimi 固定带）")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "eda_spectra.png")
    plt.close(fig)

    # ---------- 图 4：kurtogram 选带分布 ----------
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for c in CLASS_ORDER:
        ps = by_class[c]
        ax.scatter([p["band_center"] for p in ps], [p["band_width"] for p in ps],
                   color=CLASS_COLOR[c], alpha=0.7, label=c, edgecolors="none", s=28)
    ax.axvspan(common.FS / 6, 0.45 * common.FS, color="gray", alpha=0.15,
               label="kimi 原生固定带")
    ax.set_xlabel("选带中心频率 (Hz)")
    ax.set_ylabel("选带宽度 (Hz)")
    ax.set_title("Kurtogram 自适应选带分布（按类）")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "eda_bands.png")
    plt.close(fig)

    # ---------- 图 5：转速证据 ----------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    for c in CLASS_ORDER:
        ps = [p for p in by_class[c] if p["fr_peak_hz"] == p["fr_peak_hz"]]
        ax.scatter([p["fr_est_hz"] for p in ps if p["fr_est_hz"] == p["fr_est_hz"]],
                   [p["fr_peak_hz"] for p in ps if p["fr_est_hz"] == p["fr_est_hz"]],
                   color=CLASS_COLOR[c], alpha=0.75, s=28, label=c, edgecolors="none")
    ax.axhline(fr_main, color="k", ls="--", lw=1, label="主口径 1150 rpm (19.17 Hz)")
    ax.set_xlabel("manifest fr_est (Hz)")
    ax.set_ylabel("包络谱 1×fr 峰 (Hz)")
    ax.set_title("转速证据：fr_est vs 包络谱 1×fr 峰")
    ax.legend(fontsize=8)

    ax = axes[1]
    devs = []
    labels = []
    for c in CLASS_ORDER:
        d = [(p["fr_peak_hz"] - fr_main) / fr_main * 100
             for p in by_class[c] if p["fr_peak_hz"] == p["fr_peak_hz"]]
        devs.append(d)
        labels.append(c)
    bp = ax.boxplot(devs, tick_labels=labels, patch_artist=True)
    for patch, c in zip(bp["boxes"], CLASS_ORDER):
        patch.set_facecolor(CLASS_COLOR[c])
        patch.set_alpha(0.6)
    ax.axhline(0, color="k", lw=1)
    ax.axhspan(-2, 2, color="green", alpha=0.12, label="±2% 匹配容差")
    ax.set_ylabel("fr_peak 相对 19.17 Hz 偏差 (%)")
    ax.set_title("逐文件转速偏差：超出 ±2% 容差则谐波脱靶")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "eda_rpm.png")
    plt.close(fig)

    print(f"EDA 完成：{len(per_file)} 条，图 5 张，json 2 份，耗时 {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()

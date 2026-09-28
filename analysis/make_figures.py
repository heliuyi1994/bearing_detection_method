# -*- coding: utf-8 -*-
"""分歧深挖：对轮次 A 中"任一方案错 或 两方案不一致"的全部样本出多联图，
并生成逐样本主表（markdown 片段，供报告附录引用）。

输出：reports/figures/divergence/*.png、reports/table_samples.md
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

import common

common.setup_matplotlib()
import matplotlib.pyplot as plt

CLASS_ORDER = ["正常", "外圈", "内圈", "滚珠"]
FAULT_COLOR = {"BPFO": "#1f77b4", "BPFI": "#d62728", "BSF": "#ff7f0e", "FTF": "#2ca02c"}
FAULT_CN = {"BPFO": "外圈", "BPFI": "内圈", "BSF": "滚珠", "FTF": "保持架"}
FMAX = 560.0


def harm_positions(rpm: float) -> list[tuple[float, str, int]]:
    ff = common.characteristic_freqs_hz(rpm)
    out = []
    for name in ("BPFO", "BPFI", "BSF", "FTF"):
        for k in range(1, 6):
            out.append((k * ff[name], name, k))
    return out


def plot_env(ax, env_f, env_a, rpm, title, fr_est):
    m = env_f <= FMAX
    ax.plot(env_f[m], env_a[m], lw=0.8, color="#333")
    for f_hz, name, k in harm_positions(rpm):
        ax.axvline(f_hz, color=FAULT_COLOR[name], alpha=0.5 - 0.07 * (k - 1), lw=0.8)
    ax.axvline(rpm / 60, color="k", ls=":", lw=1.2, label=f"1×fr({rpm / 60:.2f}Hz)")
    if fr_est == fr_est:
        ax.axvline(fr_est, color="purple", ls="--", lw=1.0, label=f"fr_est({fr_est:.2f}Hz)")
    ax.set_xlim(0, FMAX)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("频率 (Hz)")
    ax.legend(fontsize=7, loc="upper right")


def main() -> None:
    with open(common.REPORTS_DIR / "results.csv", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    manifest = {r["file_id"]: r for r in common.load_manifest()}

    def get(scheme, rnd):
        return {r["file_id"]: r for r in rows
                if r["scheme"] == scheme and r["round"] == rnd}

    ga, ka = get("glm", "A_main_1150"), get("kimi", "A_main_1150")
    gb, kb = get("glm", "B_fr_est"), get("kimi", "B_fr_est")

    div_dir = common.FIG_DIR / "divergence"
    div_dir.mkdir(parents=True, exist_ok=True)

    # 选择：轮次 A 任一方案错 或 两方案不一致
    targets = []
    for fid in sorted(ga):
        wrong = ga[fid]["correct"] == "0" or ka[fid]["correct"] == "0"
        disagree = ga[fid]["verdict_cn"] != ka[fid]["verdict_cn"]
        if wrong or disagree:
            targets.append(fid)
    print(f"深挖样本数：{len(targets)}/{len(ga)}")

    table = ["| 编号 | 类 | 组 | 批次/会话 | 削波 | glm A | conf | kimi A | top分 | glm B | kimi B | 深挖图 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    all_fids = sorted(ga)

    for n, fid in enumerate(all_fids, 1):
        ra, rb = ga[fid], gb[fid]
        meta = manifest[fid]
        x, clip = common.read_wav(meta["relpath"])
        fr_est = meta["fr_est_hz"]
        in_div = fid in targets

        link = ""
        if in_div:
            fig, axes = plt.subplots(2, 3, figsize=(15, 7.6))
            band, bk = common.kurtogram_band(x, common.FS)
            rpm_b = float(rb["rpm_used"])
            gA = common.run_glm(x, common.FS, common.RPM_MAIN)
            kA = common.run_kimi(x, common.FS, common.RPM_MAIN, band)
            gB = common.run_glm(x, common.FS, rpm_b)
            kB = common.run_kimi(x, common.FS, rpm_b, band)

            # (0,0) 波形段
            ax = axes[0, 0]
            t = np.arange(int(0.25 * common.FS)) / common.FS
            ax.plot(t, x[: t.size], lw=0.5, color="#333")
            if clip > 0:
                ax.axhline(1.0, color="red", ls="--", lw=1)
                ax.axhline(-1.0, color="red", ls="--", lw=1, label=f"削波线（比例 {clip:.1%}）")
                ax.legend(fontsize=7)
            ax.set_xlabel("时间 (s)")
            ax.set_title(f"波形段 0.25s（RMS={gA['_indicators']['rms']:.3f} K={gA['_indicators']['kurtosis']:.2f}）", fontsize=9)

            # (0,1) 幅值谱 + 频带
            ax = axes[0, 1]
            from scipy.signal import welch
            fw, pw = welch(x, fs=common.FS, nperseg=8192)
            ax.loglog(fw[1:], pw[1:], lw=0.7, color="#333")
            ax.axvspan(band[0], band[1], color="gold", alpha=0.4,
                       label=f"kurtogram 选带({band[0]:.0f},{band[1]:.0f}) bk={bk:.1f}")
            ax.axvspan(common.FS / 6, 0.45 * common.FS, facecolor="none",
                       edgecolor="gray", ls="--", label="kimi 原生固定带")
            ax.set_title("Welch 谱与解调频带", fontsize=9)
            ax.legend(fontsize=7)

            # (0,2) glm 包络谱  (1,0) kimi 包络谱
            plot_env(axes[0, 2], gA["_env_f"], gA["_env_a"], common.RPM_MAIN,
                     f"glm 包络谱（A 判 {gA['verdict_cn']} conf={gA['confidence']:.2f}）", fr_est)
            plot_env(axes[1, 0], kA["_env_f"], kA["_env_a"], common.RPM_MAIN,
                     f"kimi 包络谱（A 判 {kA['verdict_cn']} top={kA['top_score']:.0f}）", fr_est)

            # (1,1) glm 候选得分 A/B
            ax = axes[1, 1]
            names = ["BPFO", "BPFI", "BSF", "FTF"]
            scA = dict((c, s) for c, s in gA["candidates"])
            scB = dict((c, s) for c, s in gB["candidates"])
            y = np.arange(len(names))
            ax.barh(y + 0.2, [scA.get(n_, 0) for n_ in names], height=0.38, label=f"A(1150rpm) 判{gA['verdict_cn']}")
            ax.barh(y - 0.2, [scB.get(n_, 0) for n_ in names], height=0.38, label=f"B({rpm_b:.0f}rpm) 判{gB['verdict_cn']}")
            ax.set_yticks(y, [f"{FAULT_CN[n_]}({n_})" for n_ in names])
            ax.axvline(0.15, color="k", ls=":", lw=1, label="判决阈值 0.15")
            ax.set_title("glm 候选得分（两轮口径）", fontsize=9)
            ax.legend(fontsize=7)

            # (1,2) kimi 得分 A/B（log）
            ax = axes[1, 2]
            knames = ["outer", "inner", "ball", "cage"]
            kcn = {"outer": "外圈", "inner": "内圈", "ball": "滚珠", "cage": "保持架"}
            y = np.arange(len(knames))
            ax.barh(y + 0.2, [kA["scores"].get(n_, 0) or 1e-3 for n_ in knames], height=0.38,
                    label=f"A(1150rpm) 判{kA['verdict_cn']}")
            ax.barh(y - 0.2, [kB["scores"].get(n_, 0) or 1e-3 for n_ in knames], height=0.38,
                    label=f"B({rpm_b:.0f}rpm) 判{kB['verdict_cn']}")
            ax.set_xscale("log")
            ax.set_yticks(y, [f"{kcn[n_]}({n_})" for n_ in knames])
            ax.axvline(15.0, color="red", ls=":", lw=1.2, label="健康阈值 15（仿真标定）")
            ax.set_title("kimi 得分（两轮口径，log）", fontsize=9)
            ax.legend(fontsize=7)

            verdicts = (f"[{fid}|{ra['class']}|{ra['remark']}|{ra['batch']}/{ra['session']}] "
                        f"A: glm={gA['verdict_cn']}({gA['confidence']:.2f}) kimi={kA['verdict_cn']}({kA['top_score']:.0f}) | "
                        f"B: glm={gB['verdict_cn']}({gB['confidence']:.2f}) kimi={kB['verdict_cn']}({kB['top_score']:.0f}) | "
                        f"clip={clip:.1%}")
            fig.suptitle(verdicts, fontsize=10)
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            fname = f"{fid}_{ra['class']}_{ra['remark']}.png"
            fig.savefig(div_dir / fname, dpi=300)
            plt.close(fig)
            link = f"[图](figures/divergence/{fname})"

        table.append(
            f"| {fid} | {ra['class']} | {ra['remark']} | {ra['batch']}/{ra['session']} "
            f"| {float(ra['clip_ratio']):.1%} | {ga[fid]['verdict_cn']} | {ga[fid]['confidence']} "
            f"| {ka[fid]['verdict_cn']} | {ka[fid]['confidence']} "
            f"| {gb[fid]['verdict_cn']} | {kb[fid]['verdict_cn']} | {link} |"
        )
        if n % 10 == 0:
            print(f"  出图进度 {n}/{len(all_fids)}")

    (common.REPORTS_DIR / "table_samples.md").write_text(
        "\n".join(table), encoding="utf-8")
    print(f"完成：{len(targets)} 张深挖图 + 逐样本主表 table_samples.md")


if __name__ == "__main__":
    main()

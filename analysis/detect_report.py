# -*- coding: utf-8 -*-
"""故障检出分析（不以工厂标签为基准）。

口径：只看两方案自身输出——是否检出故障（glm: verdict≠normal；kimi: verdict≠healthy）、
匹配到的故障类别与其置信度、以及头部判决的谐波证据。轮次 B（逐文件 fr_est）为
转速校正后主视图，轮次 A 并列展示。

输出：reports/detect_stats.json、reports/table_matched.md、
figures/detect_summary.png、figures/detect_conf_dist.png、控制台摘要。
"""

from __future__ import annotations

import csv
import json
from collections import Counter

import numpy as np

import common

common.setup_matplotlib()
import matplotlib.pyplot as plt

FAULT_CN_ORDER = ["外圈", "内圈", "滚珠", "其他"]


def load_rows() -> list[dict]:
    with open(common.REPORTS_DIR / "results.csv", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def evidence_glm(relpath: str, rpm: float) -> dict:
    """重跑 glm 完整诊断，提取头部候选的逐谐波证据。"""
    from bearing_diag.diagnose import diagnose

    x, _ = common.read_wav(relpath)
    r = diagnose(x, fs=common.FS, shaft_freq=rpm / 60.0,
                 bearing=common.make_bearings()[0], use_kurtogram=True)
    c = r.match.candidates[0] if r.match and r.match.candidates else None
    out = {"verdict": r.verdict, "confidence": r.confidence, "band": tuple(r.demod_band),
           "best_kurtosis": float(r.kurtogram_result.best_kurtosis), "harmonics": []}
    if c is not None:
        out["fault"] = c.fault
        out["score"] = c.score
        out["hits"] = c.hits
        out["confounded"] = c.confounded
        for h in c.harmonics:
            if h.get("hit"):
                out["harmonics"].append({
                    "k": h["k"], "target_hz": round(h["target_hz"], 2),
                    "found_hz": round(h["found_hz"], 2),
                    "prominence": round(h["prominence"], 1),
                })
    return out


def evidence_kimi(relpath: str, rpm: float, band: tuple[float, float]) -> dict:
    """重跑 kimi 提取判决类的命中峰。"""
    from bearing_fault.pipeline import DiagnosisPipeline

    x, _ = common.read_wav(relpath)
    band, _ = common.clamp_band_kimi(band, common.FS)
    r = DiagnosisPipeline(bearing=common.make_bearings()[1], fs=common.FS, band=band).run(x, rpm)
    med = float(np.median(r.env_amps))
    peaks = r.peaks.get(r.verdict) or []
    return {
        "verdict": r.verdict,
        "score": float(r.scores[r.verdict]),
        "global_median": round(med, 6),
        "peaks": [{"k": h, "freq_hz": round(f, 2), "amp": round(a, 2), "ratio": round(a / med, 1)}
                  for h, f, a in sorted(peaks)],
    }


def main() -> None:
    rows = load_rows()
    manifest = {r["file_id"]: r for r in common.load_manifest()}
    stats: dict = {"rounds": {}}

    def sel(scheme, rnd):
        return {r["file_id"]: r for r in rows if r["scheme"] == scheme and r["round"] == rnd}

    for rnd_name, rnd_key in (("A_1150rpm", "A_main_1150"), ("B_fr_est", "B_fr_est")):
        rd = {}
        for scheme in ("glm", "kimi"):
            d = sel(scheme, rnd_key)
            det = {fid: r for fid, r in d.items() if r["verdict_cn"] != "正常"}
            type_dist = Counter(r["verdict_cn"] for r in det.values())
            group_dist = Counter(r["class"] for r in det.values())  # 描述性分组，非对错
            rd[scheme] = {
                "n_detected": len(det),
                "detect_rate": len(det) / len(d),
                "type_distribution": dict(type_dist),
                "by_recorded_group": dict(group_dist),
            }
        # glm 检出样本上两方案类型一致性
        gd = sel("glm", rnd_key)
        kd = sel("kimi", rnd_key)
        det_glm = [fid for fid in gd if gd[fid]["verdict_cn"] != "正常"]
        type_agree = sum(1 for fid in det_glm if gd[fid]["verdict_cn"] == kd[fid]["verdict_cn"])
        rd["type_agreement_on_glm_detections"] = {"agree": type_agree, "n_glm_det": len(det_glm)}
        stats["rounds"][rnd_name] = rd

    # ---------- 排行榜（轮次 B 为主视图） ----------
    gd, kd = sel("glm", "B_fr_est"), sel("kimi", "B_fr_est")
    ga, ka = sel("glm", "A_main_1150"), sel("kimi", "A_main_1150")

    glm_rank = sorted(
        (r for r in gd.values() if r["verdict_cn"] != "正常"),
        key=lambda r: -float(r["confidence"]))

    def margin(r):
        top, sec = float(r["confidence"]), float(r["second"])
        return top / sec if sec > 0 else float("inf")

    kimi_rank = sorted(
        (r for r in kd.values() if r["verdict_cn"] != "正常"),
        key=lambda r: -margin(r))

    stats["glm_top10_B"] = [
        {"file_id": r["file_id"], "group": r["class"], "verdict": r["verdict_cn"],
         "confidence": float(r["confidence"]), "verdict_A": ga[r["file_id"]]["verdict_cn"]}
        for r in glm_rank[:10]]
    stats["kimi_top10_B_by_margin"] = [
        {"file_id": r["file_id"], "group": r["class"], "verdict": r["verdict_cn"],
         "top_score": float(r["confidence"]), "margin": round(margin(r), 2),
         "verdict_A": ka[r["file_id"]]["verdict_cn"]}
        for r in kimi_rank[:10]]
    # kimi 按 top_score 的前十（另一种"置信"口径）
    kimi_rank_score = sorted(
        (r for r in kd.values() if r["verdict_cn"] != "正常"),
        key=lambda r: -float(r["confidence"]))
    stats["kimi_top10_B_by_score"] = [
        {"file_id": r["file_id"], "group": r["class"], "verdict": r["verdict_cn"],
         "top_score": float(r["confidence"])}
        for r in kimi_rank_score[:10]]

    # ---------- 头部样本谐波证据 ----------
    ev = {}
    for r in glm_rank[:8]:
        fid = r["file_id"]
        ev[f"glm_{fid}"] = evidence_glm(manifest[fid]["relpath"], float(r["rpm_used"]))
    for r in kimi_rank[:8]:
        fid = r["file_id"]
        band, _ = common.kurtogram_band(*common.read_wav(manifest[fid]["relpath"])[:1], common.FS) \
            if False else (None, None)
        x, _ = common.read_wav(manifest[fid]["relpath"])
        band, _ = common.kurtogram_band(x, common.FS)
        ev[f"kimi_{fid}"] = evidence_kimi(manifest[fid]["relpath"], float(r["rpm_used"]), band)
    stats["evidence"] = ev

    # ---------- 逐样本匹配表 ----------
    tbl = ["| 编号 | 记录组 | glm A 判/置信 | glm B 判/置信 | kimi A 判/top分 | kimi B 判/top分/领先比 |",
           "|---|---|---|---|---|---|"]
    for fid in sorted(gd):
        gA, gB = ga[fid], gd[fid]
        kA, kB = ka[fid], kd[fid]
        def fmt(r, val):
            v = f"{float(val):.2f}"
            return f"{r['verdict_cn']} / {v}" if r["verdict_cn"] != "正常" else "—未检出—"
        m = margin(kB) if kB["verdict_cn"] != "正常" else float("nan")
        ms = f"{m:.1f}" if m == m else "—"
        tbl.append(
            f"| {fid} | {gB['class']} | {fmt(gA, gA['confidence'])} | {fmt(gB, gB['confidence'])} "
            f"| {fmt(kA, kA['confidence'])} | {fmt(kB, kB['confidence'])} / {ms} |")
    (common.REPORTS_DIR / "table_matched.md").write_text("\n".join(tbl), encoding="utf-8")

    # ---------- 图 ----------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    x = np.arange(len(FAULT_CN_ORDER))
    w = 0.2
    combos = [("glm", "A_main_1150", "glm A(1150)"), ("glm", "B_fr_est", "glm B(fr_est)"),
              ("kimi", "A_main_1150", "kimi A"), ("kimi", "B_fr_est", "kimi B")]
    for i, (scheme, rnd, label) in enumerate(combos):
        d = sel(scheme, rnd)
        det = [r for r in d.values() if r["verdict_cn"] != "正常"]
        cnt = Counter(r["verdict_cn"] for r in det)
        axes[0].bar(x + (i - 1.5) * w, [cnt.get(t, 0) for t in FAULT_CN_ORDER],
                    width=w, label=f"{label} 共{len(det)}")
    axes[0].set_xticks(x, FAULT_CN_ORDER)
    axes[0].set_ylabel("检出样本数")
    axes[0].set_title("检出故障类别分布（81 条全体，不对照标签）")
    axes[0].legend(fontsize=8)

    conf_glm_B = [float(r["confidence"]) for r in gd.values() if r["verdict_cn"] != "正常"]
    conf_kimi_B = [float(r["confidence"]) for r in kd.values()]
    axes[1].hist(conf_glm_B, bins=20, alpha=0.7, label=f"glm B 检出置信度（n={len(conf_glm_B)}）")
    axes[1].set_xlabel("glm 置信度")
    axes[1].set_ylabel("样本数")
    ax2 = axes[1].twinx()
    ax2.hist(np.log10(conf_kimi_B), bins=20, alpha=0.5, color="orange", label="kimi B top分(log10)")
    ax2.set_ylabel("kimi top 分数（log10 轴）")
    lines1, labels1 = axes[1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[1].legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    axes[1].set_title("判决置信/得分分布")
    fig.tight_layout()
    fig.savefig(common.FIG_DIR / "detect_summary.png")
    plt.close(fig)
    common.save_json(stats, common.REPORTS_DIR / "detect_stats.json")

    # ---------- 控制台摘要 ----------
    for rnd_name, rd in stats["rounds"].items():
        print(f"\n=== {rnd_name} ===")
        for scheme in ("glm", "kimi"):
            s = rd[scheme]
            print(f"{scheme}: 检出 {s['n_detected']}/81 ({s['detect_rate']:.1%})  "
                  f"类别分布={s['type_distribution']}  按记录组={s['by_recorded_group']}")
        print(f"glm 检出样本上两方案类型一致: {rd['type_agreement_on_glm_detections']['agree']}"
              f"/{rd['type_agreement_on_glm_detections']['n_glm_det']}")
    print("\nglm B 轮检出置信度前十:")
    for it in stats["glm_top10_B"]:
        print(f"  {it['file_id']} [组:{it['group']}] 判{it['verdict']} conf={it['confidence']:.3f} (A轮:{it['verdict_A']})")
    print("\nkimi B 轮按领先比(top/2nd)前十:")
    for it in stats["kimi_top10_B_by_margin"]:
        print(f"  {it['file_id']} [组:{it['group']}] 判{it['verdict']} top={it['top_score']:.0f} "
              f"margin={it['margin']} (A轮:{it['verdict_A']})")
    print("\n头部证据（glm 前 5）:")
    for k, v in list(ev.items()):
        if k.startswith("glm_"):
            print(f"  {k}: {v.get('fault')} score={v.get('score'):.3f} hits={v.get('hits')} "
                  f"confounded={v.get('confounded')} bk={v['best_kurtosis']:.1f}")
            print(f"     谐波: {v['harmonics']}")
    print("\n头部证据（kimi 前 5）:")
    for k, v in list(ev.items()):
        if k.startswith("kimi_"):
            print(f"  {k}: {v['verdict']} score={v['score']:.0f} med={v['global_median']}")
            print(f"     峰: {v['peaks']}")


if __name__ == "__main__":
    main()

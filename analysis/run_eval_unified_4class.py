#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""统一包验收复测（二）：labeled_dataset_4class 81 条四类台架数据。

用 bearing_unified.diagnose（v2 规则、FIELD7、转速窗 1100–1200）对 81 条
真实台架 WAV 逐条诊断；几何断言照跑（common.verify_geometry）。

验收门槛（建议稿 §6.7 + 评审口径）：
- 正常类误报保持 0（10 条正常全部判 normal——v2 盲梳检 + 干扰防护的
  门控性质；马氏辅助判据在 10 条正常样本上无法稳定标定，本批不启用，
  与门槛口径一致）；
- 给出四类一致率，与 glm/kimi 轮次 B 数字对照（统一包无保持架判别，
  滚珠主峰 2×BSF 由滚珠(2×BSF主峰带)频带归入滚珠类）。

用法（在 analysis/ 目录下运行）：
    python run_eval_unified_4class.py            # 全量 81 条（约 1 分钟）
    python run_eval_unified_4class.py --smoke    # 8 条冒烟
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ANALYSIS_DIR = Path(__file__).resolve().parent
for _p in (str(ANALYSIS_DIR),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import common  # noqa: E402
from bearing_unified.diagnose import diagnose  # noqa: E402

OUT_DIR = ANALYSIS_DIR / "reports" / "unified_4class"

BEARING_NAME = "FIELD7"
RPM_WINDOW = (1100.0, 1200.0)

#: 统一包 verdict → 中文四类标签（外圈/内圈/滚珠/正常）；未定位单列
VERDICT2CN = {"BPFO": "外圈", "BPFI": "内圈", "BSF": "滚珠", "normal": "正常"}

#: 前次评测轮次 B（逐文件 fr_est）的对照数字（主报告/建议稿 §4.2）
ROUND_B = {
    "glm": {"overall": 21, "n": 81, "正常": (10, 10), "外圈": (7, 20),
            "内圈": (2, 29), "滚珠": (2, 22)},
    "kimi": {"overall": 17, "n": 81, "正常": (0, 10), "外圈": (4, 20),
             "内圈": (7, 29), "滚珠": (6, 22)},
}

CSV_COLUMNS = [
    "file_id", "relpath", "label", "batch", "fr_est_hz",
    "verdict", "verdict_cn", "hit", "conf_outer", "conf_inner", "conf_ball",
    "argmax_class", "comb_freqs", "comb_confirmed", "cepstrum_confirmed",
    "suspected_interference", "n_clean_combs", "n_localized",
    "bearing_unlocalized", "rpm_est", "rpm_fallback",
    "band_low", "band_high", "elapsed_s", "note", "comb_detail",
]


def process_one(rec: dict) -> dict:
    t0 = time.perf_counter()
    try:
        x, clip = common.read_wav(rec["relpath"])
        r = diagnose(x, fs=common.FS, bearing=BEARING_NAME, rpm_window=RPM_WINDOW)
        v_cn = VERDICT2CN.get(r.verdict, "未定位")
        hit = int(v_cn == rec["class"])
        d = {
            "verdict": r.verdict,
            "verdict_cn": v_cn,
            "hit": hit,
            "conf_outer": r.conf_outer,
            "conf_inner": r.conf_inner,
            "conf_ball": r.conf_ball,
            "argmax_class": r.argmax_class,
            "comb_freqs": ";".join(
                f"{c['f0']:.2f}{'!' if c['suspected_interference'] else ''}"
                for c in r.combs),
            "comb_confirmed": r.comb_confirmed,
            "cepstrum_confirmed": r.cepstrum_confirmed,
            "suspected_interference": r.n_interference,
            "n_clean_combs": r.n_clean_combs,
            "n_localized": r.n_localized,
            "bearing_unlocalized": int(r.localization["unlocalized"]),
            "rpm_est": r.rpm_est,
            "rpm_fallback": int(r.rpm_fallback),
            "band_low": r.demod_band[0],
            "band_high": r.demod_band[1],
            "note": f"clip={clip:.4f}",
            "comb_detail": json.dumps([
                {"f0": round(c["f0"], 3), "score": round(c["score"], 4),
                 "hits": c["hits"], "cep_hits": c["cep_hits"],
                 "interference": c["suspected_interference"],
                 "guard": c.get("guard_reason", ""),
                 "cls": (c.get("loc") or {}).get("cls")}
                for c in r.combs], ensure_ascii=False),
            "error": "",
        }
    except Exception as exc:  # noqa: BLE001
        d = {"error": f"{type(exc).__name__}: {exc}"}
    d["elapsed_s"] = round(time.perf_counter() - t0, 4)
    out = {
        "file_id": rec["file_id"], "relpath": rec["relpath"],
        "label": rec["class"], "batch": rec["batch"],
        "fr_est_hz": rec["fr_est_hz"],
        **d,
    }
    return out


def aggregate(results: list[dict], total_s: float) -> dict:
    per_class: dict[str, dict] = {}
    for r in results:
        d = per_class.setdefault(r["label"], {"n": 0, "hit": 0})
        d["n"] += 1
        d["hit"] += int(r["hit"])
    overall_hit = sum(d["hit"] for d in per_class.values())
    n = len(results)
    normal_rows = [r for r in results if r["label"] == "正常"]
    fp = sum(1 for r in normal_rows if r["verdict"] != "normal")
    detail = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "script": "analysis/run_eval_unified_4class.py",
        "engine": "bearing_unified.diagnose（统一包，v2 原生；无保持架判别）",
        "bearing": BEARING_NAME,
        "rpm_window": RPM_WINDOW,
        "counts": {"n": n, "hit": overall_hit, "一致率": overall_hit / max(n, 1)},
        "per_class": per_class,
        "normal_false_alarm": fp,
        "normal_gate_ok": fp == 0,
        "round_b_reference": ROUND_B,
        "unlocalized_ids": [r["file_id"] for r in results
                            if r["verdict"] == "bearing_unlocalized"],
        "timing": {"total_s": total_s, "per_file_s": total_s / max(n, 1)},
        "results_note": "标签为工厂声称值（非拆解确认），一致率为与标签的一致率",
    }
    return detail


def run_smoke(rows: list[dict]) -> None:
    import itertools

    picked = [r for g in ("正常", "外圈", "内圈", "滚珠")
              for r in itertools.islice((x for x in rows if x["class"] == g), 2)]
    for rec in picked:
        r = process_one(rec)
        if r["error"]:
            print(f"{rec['file_id']} {rec['class']}: 失败 {r['error']}")
            continue
        print(f"{rec['file_id']} 真={rec['class']} 判={r['verdict_cn']}"
              f"（{r['verdict']}）hit={r['hit']} "
              f"conf(o/i/b)={r['conf_outer']:.2f}/{r['conf_inner']:.2f}/{r['conf_ball']:.2f} "
              f"梳=[{r['comb_freqs']}]")


def main() -> None:
    ap = argparse.ArgumentParser(description="统一包 81 条四类验收复测")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    common.verify_geometry()
    rows = common.load_manifest()
    print(f"共 {len(rows)} 条四类台架数据（引擎: bearing_unified.diagnose）")
    if args.smoke:
        run_smoke(rows)
        return

    t0 = time.time()
    results, skipped = [], []
    for i, rec in enumerate(rows, 1):
        r = process_one(rec)
        (skipped if r["error"] else results).append(r)
        if i % 20 == 0 or i == len(rows):
            print(f"[进度 {i}/{len(rows)}] 失败 {len(skipped)}", flush=True)
    total_s = time.time() - t0
    assert not skipped, f"有失败文件: {skipped}"

    detail = aggregate(results, total_s)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "results.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in CSV_COLUMNS})
    common.save_json(detail, OUT_DIR / "eval_detail.json")

    print(f"\n完成 {len(results)}/{len(rows)} 条，耗时 {total_s:.0f}s")
    print(f"总体一致率: {detail['counts']['hit']}/{detail['counts']['n']} = "
          f"{detail['counts']['一致率']:.1%}")
    for cn in ("正常", "外圈", "内圈", "滚珠"):
        d = detail["per_class"].get(cn, {"n": 0, "hit": 0})
        print(f"  {cn}: {d['hit']}/{d['n']}"
              + ("（门槛：0 误报" + ("✓ 达标" if detail["normal_gate_ok"] else "✗ 未达标") + "）"
                 if cn == "正常" else ""))
    print(f"输出: {OUT_DIR}/results.csv / eval_detail.json")


if __name__ == "__main__":
    main()

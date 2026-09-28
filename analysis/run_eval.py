# -*- coding: utf-8 -*-
"""主评测：81 文件 × 两方案 × 两轮次（主口径 1150 rpm / 敏感性 fr_est）。

输出：reports/results.csv（逐样本扁平表）、reports/eval_detail.json（候选/备注明细）。
"""

from __future__ import annotations

import csv
import json
import time

import numpy as np

import common

ROUNDS = ("A_main_1150", "B_fr_est")


def main() -> None:
    rows = common.load_manifest()
    common.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_rows = []
    detail = []
    t0 = time.time()

    for i, r in enumerate(rows, 1):
        x, clip = common.read_wav(r["relpath"])
        band, bk = common.kurtogram_band(x, common.FS)
        rpm_b = (r["fr_est_hz"] * 60.0) if r["fr_est_hz"] == r["fr_est_hz"] else common.RPM_MAIN
        fr_source_b = "fr_est" if r["fr_est_hz"] == r["fr_est_hz"] else "fallback_1150"

        for round_name, rpm, fr_src in (
            ("A_main_1150", common.RPM_MAIN, "fixed_1150"),
            ("B_fr_est", rpm_b, fr_source_b),
        ):
            g = common.run_glm(x, common.FS, rpm)
            k = common.run_kimi(x, common.FS, rpm, band)
            base = {
                "file_id": r["file_id"],
                "relpath": r["relpath"],
                "class": r["class"],
                "remark": r["remark"],
                "batch": r["batch"],
                "session": r["session"],
                "clip_ratio": f"{clip:.5f}",
                "best_kurtosis": f"{bk:.2f}",
                "band_low": f"{band[0]:.0f}",
                "band_high": f"{band[1]:.0f}",
                "round": round_name,
                "rpm_used": f"{rpm:.1f}",
                "fr_source": fr_src,
            }
            out_rows.append({
                **base,
                "scheme": "glm",
                "verdict_raw": g["verdict_raw"],
                "verdict_cn": g["verdict_cn"],
                "correct": int(g["verdict_cn"] == r["class"]),
                "confidence": f"{g['confidence']:.4f}",
                "second": f"{(g['candidates'][1][1] if len(g['candidates']) > 1 else 0.0):.4f}",
                "level": g["level"],
                "band_clamped": "",
                "peak_error": "",
            })
            out_rows.append({
                **base,
                "scheme": "kimi",
                "verdict_raw": k["verdict_raw"],
                "verdict_cn": k["verdict_cn"],
                "correct": int(k["verdict_cn"] == r["class"]),
                "confidence": f"{k['top_score']:.2f}",
                "second": f"{k['second_score']:.2f}",
                "level": "",
                "band_clamped": int(k["band_clamped"]),
                "peak_error": f"{k['peak_error']:.4f}" if k["peak_error"] is not None else "",
            })
            detail.append({
                **base,
                "glm": {
                    "verdict": g["verdict_raw"], "confidence": g["confidence"], "level": g["level"],
                    "candidates": [(c, round(s, 4)) for c, s in g["candidates"]],
                    "notes": g["notes"],
                },
                "kimi": {
                    "verdict": k["verdict_raw"],
                    "scores": {kk: round(vv, 2) for kk, vv in k["scores"].items()},
                    "band": list(k["band"]), "band_clamped": k["band_clamped"],
                    "peak_error": k["peak_error"],
                },
            })
        if i % 10 == 0 or i == len(rows):
            print(f"  评测进度 {i}/{len(rows)} ({time.time() - t0:.0f}s)")

    cols = list(out_rows[0].keys())
    with open(common.REPORTS_DIR / "results.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(out_rows)
    with open(common.REPORTS_DIR / "eval_detail.json", "w", encoding="utf-8") as f:
        json.dump(detail, f, ensure_ascii=False, indent=1)

    # 快速汇总打印
    for round_name in ROUNDS:
        for scheme in ("glm", "kimi"):
            sel = [r for r in out_rows if r["round"] == round_name and r["scheme"] == scheme]
            acc = sum(int(r["correct"]) for r in sel) / len(sel)
            print(f"{round_name} {scheme}: acc={acc:.3f} ({sum(int(r['correct']) for r in sel)}/{len(sel)})")
    print(f"评测完成，results.csv {len(out_rows)} 行，总耗时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

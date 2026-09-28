#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""统一包正式验收复测：audio-assets 1349 条（bearing_unified.diagnose 引擎）。

与 run_eval_audio_assets.py（原型 v2）同一批数据、同一子集划分、同一指标
口径、同一份马氏标定协议（bearing_unified.baseline.calibrate_oof，与原型
calibrate_oof 同算法同 seed），产出可与 rules_v2/ 逐项对照的验收数字。

用法（在 analysis/ 目录下运行）：

.. code-block:: bash

    python run_eval_unified.py --smoke           # 24 条分层冒烟（不落盘）
    python run_eval_unified.py                   # 全量 1349 条（后台、进度日志、可重跑）
    python run_eval_unified.py --aggregate-only  # 由已有 results.csv 重出聚合

验收门槛（merged_proposal/双方案优缺点汇总与合并建议.md §6.7 评审口径）：
- 轴承全体口径检出率 ≥ 48.8%（原型 v2 基线）不回退；
- 正常误报率 ≤ 2.1%；
- 摩擦检出率 ≥ 15.5%（v1 基线）；
- 「检出但未定位」显著低于原型 v2 的 45/168 且可归因。
诚实纪律：门槛不达标就如实报告并归因，绝不为过门槛而调参。
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
ROOT = ANALYSIS_DIR.parent
for _p in (str(ANALYSIS_DIR),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import common  # noqa: E402  复用：中文字体、几何断言、JSON 落盘
import run_eval_audio_assets as base  # noqa: E402  复用：清单/子集/聚合函数
from bearing_unified.baseline import calibrate_oof  # noqa: E402
from bearing_unified.diagnose import diagnose  # noqa: E402
from bearing_unified.fault_freqs import get_bearing  # noqa: E402

OUT_DIR = ANALYSIS_DIR / "reports" / "audio_assets" / "unified"
RESULTS_CSV = OUT_DIR / "results.csv"
DETAIL_JSON = OUT_DIR / "eval_detail.json"
FIG_DIR = OUT_DIR / "figures"

#: 原型 v2 基线（rules_v2/eval_detail.json），门槛对照用
PROTO_V2_JSON = ANALYSIS_DIR / "reports" / "audio_assets" / "rules_v2" / "eval_detail.json"

FS = 16000.0
BEARING_NAME = "FIELD7"
RPM_WINDOW = (1100.0, 1200.0)

CSV_COLUMNS = [
    "id", "storage_path", "file_label_status", "description",
    "is_normal", "is_abnormal", "is_bearing", "is_friction", "is_detailed",
    "subset", "detailed_label",
    "rpm_est", "rpm_fallback", "rpm_fr_hz", "rpm_prom_ratio",
    "maha_dist", "trigger",
    "conf_outer", "conf_inner", "conf_ball", "max_conf",
    "pred_normal_abnormal", "pred_fault_type", "low_conf", "argmax_class",
    "verdict",
    "band_low", "band_high", "best_kurtosis",
    "feat_kurtosis", "feat_crest_factor", "feat_clearance_factor",
    "clip_ratio", "duration_s", "elapsed_s", "note",
    "comb_freqs", "comb_confirmed", "cepstrum_confirmed",
    "suspected_interference", "bearing_unlocalized",
    "n_clean_combs", "n_localized", "comb_detail",
]


# ---------------------------------------------------------------- WAV 读取

def read_wav_any(path: str) -> tuple[np.ndarray, float, float, str]:
    """读 WAV → (满幅归一信号, fs, 削波比例, 备注)。

    与合并原型 read_wav 同规则：非 16 kHz 用 polyphase 有理数重采样
    （audio-assets 中 id=20101 为 48 kHz/10 s）。
    """
    from fractions import Fraction

    from scipy.io import wavfile

    sr, raw = wavfile.read(path)
    notes: list[str] = []
    if raw.ndim > 1:
        raw = raw[:, 0]
        notes.append("立体声取第1声道")
    clip_ratio = float(np.mean(np.abs(raw) >= 32767))
    x = raw.astype(np.float64) / 32768.0
    if sr != int(FS):
        frac = Fraction(int(FS), int(sr))
        from scipy import signal as sps

        x = sps.resample_poly(x, frac.numerator, frac.denominator)
        notes.append(f"重采样{sr}→{int(FS)}Hz(resample_poly {frac.numerator}/{frac.denominator})")
        sr = int(FS)
    return x, float(sr), clip_ratio, "; ".join(notes)


# ---------------------------------------------------------------- 逐文件处理

def process_one(rec: dict) -> dict:
    """单个文件的统一包 v2 诊断；异常不抛出，记录到 error 字段。"""
    t0 = time.perf_counter()
    try:
        x, fs, clip_ratio, note = read_wav_any(rec["storage_path"])
        r = diagnose(x, fs=fs, bearing=BEARING_NAME, rpm_window=RPM_WINDOW)
        feats = {k: float(r.indicators[k])
                 for k in ("kurtosis", "crest_factor", "clearance_factor")}
        d = {
            "rpm_est": r.rpm_est,
            "rpm_fallback": int(r.rpm_fallback),
            "rpm_fr_hz": r.rpm_fr_hz,
            "rpm_prom_ratio": r.rpm_prom_ratio,
            "feat_kurtosis": feats["kurtosis"],
            "feat_crest_factor": feats["crest_factor"],
            "feat_clearance_factor": feats["clearance_factor"],
            "conf_outer": r.conf_outer,
            "conf_inner": r.conf_inner,
            "conf_ball": r.conf_ball,
            "max_conf": r.confidence,
            "argmax_class": r.argmax_class,
            "band_low": r.demod_band[0],
            "band_high": r.demod_band[1],
            "best_kurtosis": (float(r.kurtogram_result.best_kurtosis)
                              if r.kurtogram_result else float("nan")),
            "comb_freqs": ";".join(
                f"{c['f0']:.2f}{'!' if c['suspected_interference'] else ''}"
                for c in r.combs),
            "comb_confirmed": r.comb_confirmed,
            "cepstrum_confirmed": r.cepstrum_confirmed,
            "suspected_interference": r.n_interference,
            "bearing_unlocalized": int(r.localization["unlocalized"]),
            "n_clean_combs": r.n_clean_combs,
            "n_localized": r.n_localized,
            "comb_detail": json.dumps([
                {"f0": round(c["f0"], 3), "score": round(c["score"], 4),
                 "hits": c["hits"], "cep_hits": c["cep_hits"],
                 "interference": c["suspected_interference"],
                 "guard": c.get("guard_reason", ""),
                 "band": (c.get("loc") or {}).get("band", ""),
                 "cls": (c.get("loc") or {}).get("cls"),
                 "fr_hyp": (c.get("loc") or {}).get("fr_hyp")}
                for c in r.combs], ensure_ascii=False),
            "verdict": r.verdict,
            "clip_ratio": clip_ratio,
            "duration_s": float(x.size / fs),
            "note": note,
            "error": "",
        }
    except Exception as exc:  # noqa: BLE001 逐文件隔离，失败记录后跳过
        d = {"error": f"{type(exc).__name__}: {exc}"}
    d["elapsed_s"] = round(time.perf_counter() - t0, 4)
    return {**rec, **d}


def feats_of(r: dict) -> dict[str, float]:
    return {k: float(r[f"feat_{k}"]) for k in ("kurtosis", "crest_factor", "clearance_factor")}


def run_all(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """全量串行处理，每 50 条打一行进度日志，每 200 条写一次断点。"""
    results, skipped = [], []
    t0 = time.time()
    n = len(rows)
    for i, rec in enumerate(rows, 1):
        r = process_one(rec)
        (skipped if r["error"] else results).append(r)
        if i % 200 == 0:
            common.save_json({"results": results, "skipped": skipped},
                             OUT_DIR / "checkpoint.json")
        if i % 50 == 0 or i == n:
            dt = time.time() - t0
            print(f"[进度 {i}/{n}] 已耗时 {dt:.0f}s，均值 {dt / i:.2f}s/条，"
                  f"预计总耗时 {dt / i * n / 60:.1f}min，失败 {len(skipped)}",
                  flush=True)
    return results, skipped


# ---------------------------------------------------------------- 标定与判决

def calibrate_and_decide(results: list[dict]) -> dict:
    """触发级防泄漏标定（统一包 baseline，与原型同算法同 seed）+ v2 判决。

    判决与原型 decide_v2 完全一致：有净梳 → 轴承/未定位；无梳且马氏触发 →
    摩擦；皆无 → 正常。低置信：判摩擦但距离 < 2×阈值，或判轴承但 max 置信度
    < 平衡档 0.2。
    """
    normals = [r for r in results if r["is_normal"]]
    cal = calibrate_oof([feats_of(r) for r in normals])  # 默认 5 折 seed=42
    thr = cal["thresholds"]["p99"]
    for r, d in zip(normals, cal["oof"]):
        r["maha_dist"] = float(d)
    for r in results:
        if not r["is_normal"]:
            r["maha_dist"] = float(cal["baseline"].distance(feats_of(r)))
    for r in results:
        trigger = bool(r["maha_dist"] >= thr)
        clean = int(r["n_clean_combs"]) > 0
        if clean:
            pred = "轴承" if int(r["n_localized"]) > 0 else "轴承-未定位"
            low_conf = bool(pred == "轴承" and float(r["max_conf"]) < 0.2)
        elif trigger:
            pred = "摩擦"
            low_conf = bool(r["maha_dist"] < 2.0 * thr)
        else:
            pred = "正常"
            low_conf = False
        r["trigger"] = int(trigger)
        r["pred_fault_type"] = pred
        r["pred_normal_abnormal"] = "异常" if pred != "正常" else "正常"
        r["low_conf"] = int(low_conf)
    return {
        "thresholds": cal["thresholds"],
        "fold_sizes": cal["fold_sizes"],
        "n_normal": cal["n"],
        "n_splits": cal["n_splits"],
        "seed": cal["seed"],
        "oof_stats": {
            "min": float(np.min(cal["oof"])),
            "median": float(np.median(cal["oof"])),
            "mean": float(np.mean(cal["oof"])),
            "max": float(np.max(cal["oof"])),
        },
    }


# ---------------------------------------------------------------- 落盘

def write_results_csv(results: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow({k: base._fmt(r.get(k, "")) for k in CSV_COLUMNS})


def read_results_csv(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            for k in ("is_normal", "is_abnormal", "is_bearing", "is_friction",
                      "is_detailed", "file_label_status", "rpm_fallback",
                      "trigger", "low_conf", "comb_confirmed",
                      "cepstrum_confirmed", "suspected_interference",
                      "bearing_unlocalized", "n_clean_combs", "n_localized"):
                r[k] = int(r[k])
            for k in ("rpm_est", "rpm_fr_hz", "rpm_prom_ratio", "maha_dist",
                      "conf_outer", "conf_inner", "conf_ball", "max_conf",
                      "feat_kurtosis", "feat_crest_factor", "feat_clearance_factor",
                      "clip_ratio", "duration_s", "elapsed_s",
                      "band_low", "band_high", "best_kurtosis"):
                r[k] = float(r[k])
            rows.append(r)
    return rows


# ---------------------------------------------------------------- 聚合

def aggregate(results: list[dict], skipped: list[dict], total_s: float) -> dict:
    cal = calibrate_and_decide(results)
    detail: dict = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "script": "analysis/run_eval_unified.py",
        "engine": "bearing_unified.diagnose（统一包，v2 原生）",
        "bearing": BEARING_NAME,
        "rpm_window": RPM_WINDOW,
        "counts": {},
        "calibration": cal,
        "normal_abnormal_v2": base.agg_v2_normal_abnormal(results),
        "bearing_friction_v2": base.agg_v2_bearing_friction(results),
        "low_conf": base.agg_low_conf(results),
        "detailed_reference": base.agg_detailed(results),
        "rpm": base.agg_rpm(results),
        "v2": base.agg_v2(results),
        "skipped_files": [{"id": s["id"], "storage_path": s["storage_path"],
                           "error": s["error"]} for s in skipped],
        "timing": {"total_s": total_s,
                   "per_file_s": total_s / max(len(results), 1)},
    }
    counts = base.subset_counts(results)
    counts["processed"] = len(results)
    counts["skipped"] = len(skipped)
    counts["total"] = len(results) + len(skipped)
    detail["counts"] = counts
    if PROTO_V2_JSON.exists():
        detail["proto_v2_baseline"] = json.loads(
            PROTO_V2_JSON.read_text(encoding="utf-8"))

    write_results_csv(results, RESULTS_CSV)
    common.save_json(detail, DETAIL_JSON)
    common.save_json(detail["skipped_files"], OUT_DIR / "skipped_files.json")

    # 复用 v2 图集（写出到本目录 figures/）
    common.setup_matplotlib()
    base.FIG_DIR = FIG_DIR
    base.make_figures_v2(results, detail)
    return detail


# ---------------------------------------------------------------- 冒烟

def run_smoke(rows: list[dict]) -> None:
    """24 条分层冒烟（8 正常/8 轴承/8 摩擦），打印梳检关键中间量。"""
    rng = np.random.default_rng(42)
    picked: list[dict] = []
    for flag, n in (("is_normal", 8), ("is_bearing", 8), ("is_friction", 8)):
        pool = [r for r in rows if r[flag]]
        idx = rng.choice(len(pool), size=n, replace=False)
        picked.extend(pool[int(i)] for i in idx)

    print(f"冒烟 {len(picked)} 条（8 正常/8 轴承/8 摩擦）")
    print(f"{'id':>6} {'子集':<4} {'耗时s':>5} {'verdict':<20} "
          f"{'梳A/B/干扰/净':>12} {'conf(o/i/b)':>18}  梳基频")
    results = []
    t0 = time.time()
    for rec in picked:
        r = process_one(rec)
        results.append(r)
        if r["error"]:
            print(f"{rec['id']:>6} {rec['subset']:<4} 失败: {r['error']}")
            continue
        confs = f"{r['conf_outer']:.2f}/{r['conf_inner']:.2f}/{r['conf_ball']:.2f}"
        fun = f"{r['comb_confirmed']}/{r['cepstrum_confirmed']}/{r['suspected_interference']}/{r['n_clean_combs']}"
        print(f"{r['id']:>6} {r['subset']:<4} {r['elapsed_s']:>5.2f} "
              f"{r['verdict']:<20} {fun:>12} {confs:>18}  [{r['comb_freqs']}]")
    dt = time.time() - t0
    ok = [r for r in results if not r["error"]]
    n_comb_brg = sum(1 for r in ok if r["is_bearing"] and r["n_clean_combs"] > 0)
    n_comb_nrm = sum(1 for r in ok if r["is_normal"] and r["n_clean_combs"] > 0)
    confs = np.array([[r["conf_outer"], r["conf_inner"], r["conf_ball"]] for r in ok])
    assert not np.isnan(confs).any(), "存在 NaN 置信度"
    print(f"\n冒烟汇总：成功 {len(ok)}/{len(picked)}，"
          f"均值 {dt / max(len(picked), 1):.2f}s/条 → 全量 ETA "
          f"{dt / max(len(picked), 1) * len(rows) / 60:.1f} min；"
          f"轴承有净梳 {n_comb_brg}/{sum(1 for r in ok if r['is_bearing'])}；"
          f"正常有净梳 {n_comb_nrm}/{sum(1 for r in ok if r['is_normal'])}（应稀少）")


# ---------------------------------------------------------------- 主流程

def main() -> None:
    ap = argparse.ArgumentParser(description="统一包验收复测（audio-assets 1349 条）")
    ap.add_argument("--smoke", action="store_true", help="24 条分层冒烟，不落盘")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="跳过逐文件计算，由已有 results.csv 重出聚合/图")
    args = ap.parse_args()

    common.verify_geometry()
    f7 = get_bearing(BEARING_NAME)
    ff = f7.fault_frequencies(1.0)
    assert abs(ff["BPFO"] - 2.578320) < 2e-4 and abs(ff["BPFI"] - 4.421680) < 2e-4
    rows = base.load_manifest()
    mc = base.subset_counts(rows)
    print(f"引擎: bearing_unified.diagnose（FIELD7 @ {RPM_WINDOW}）")
    print(f"manifest 共 {mc['total']} 条：正常 {mc['is_normal']} / 异常 {mc['is_abnormal']}"
          f"（轴承 {mc['is_bearing']} / 摩擦 {mc['is_friction']} / "
          f"其他 {mc['other_abnormal']}）")

    if args.smoke:
        run_smoke(rows)
        return

    if args.aggregate_only:
        results = read_results_csv(RESULTS_CSV)
        skipped = []
        total_s = sum(r["elapsed_s"] for r in results)
        detail = aggregate(results, skipped, total_s)
        print(f"聚合完成：{RESULTS_CSV} / {DETAIL_JSON}")
        return

    t0 = time.time()
    results, skipped = run_all(rows)
    total_s = time.time() - t0
    detail = aggregate(results, skipped, total_s)
    na = detail["normal_abnormal_v2"]
    bf = detail["bearing_friction_v2"]
    print(f"\n完成 {detail['counts']['processed']}/{detail['counts']['total']} 条，"
          f"耗时 {total_s / 60:.1f} min")
    print(f"正常误报率 {na['正常误报率']:.1%}（门槛 ≤2.1%）| "
          f"异常检出率 {na['异常检出率']:.1%}")
    print(f"轴承全体口径（含未定位）{bf['bearing']['全体口径检出率_含未定位']:.1%}（门槛 ≥48.8%）| "
          f"摩擦全体口径 {bf['friction']['全体口径检出率_仅定位']:.1%}（门槛 ≥15.5%）")
    print(f"输出：{RESULTS_CSV}\n      {DETAIL_JSON}\n      {FIG_DIR}/")


if __name__ == "__main__":
    main()

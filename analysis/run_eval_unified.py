#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""统一包正式验收复测：audio-assets 1349 条（bearing_unified.diagnose 引擎）。

与原型 v2 产物（reports/audio_assets/rules_v2/）同一批数据、同一子集划分、
同一指标口径、同一份马氏标定协议（bearing_unified.baseline.calibrate_oof，
与原型同算法同 seed），产出可与 rules_v2/ 逐项对照的验收数字。
清单/聚合/绘图工具已内联本文件（原复用 run_eval_audio_assets.py，该脚本
随旧双方案清理移除）。

用法（在 analysis/ 目录下运行）：

.. code-block:: bash

    python run_eval_unified.py --smoke           # 24 条分层冒烟（不落盘）
    python run_eval_unified.py                   # 全量 1349 条（后台、进度日志、可重跑）
    python run_eval_unified.py --aggregate-only  # 由已有 results.csv 重出聚合

验收门槛（合并评审稿 §6.7 评审口径；该稿已随旧项目清理移除，见 git 历史）：
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
from bearing_unified.baseline import calibrate_oof  # noqa: E402
from bearing_unified.diagnose import diagnose  # noqa: E402
from bearing_unified.fault_freqs import get_bearing  # noqa: E402
from bearing_unified.localize import class_bands  # noqa: E402

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

# ---------------------------------------------------------------- 清单与子集

#: 音频资产清单（与原型 v1/v2 共用的同一份 manifest.tsv）
MANIFEST = ANALYSIS_DIR / "reports" / "audio_assets" / "manifest.tsv"
DETAIL_KEYWORDS = ("外圈", "内圈", "滚珠")

#: 判决/绘图常量（与验收口径一致；原内联自合并原型脚本）
MAHA_QUANTILE = 99.0          # 马氏触发主阈值分位（p99）
FR_NOMINAL = 1150.0 / 60.0    # 19.167 Hz，轴频谐波梳防护基准
MAINS_HZ = 50.0               # 工频干扰（恰落在外圈频带内，必须防护）
CONF_REF = 0.2                # 定位置信度参考线（平衡档）


def load_manifest() -> list[dict]:
    """读 manifest.tsv 并按子集定义打标（严格遵守任务口径）。"""
    rows: list[dict] = []
    with open(MANIFEST, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            desc = (r["description"] or "").strip()
            status = int(r["file_label_status"])
            is_normal = status == 1
            is_abnormal = status == -1
            is_bearing = is_abnormal and desc == "轴承"          # 精确匹配
            is_friction = is_abnormal and "摩擦" in desc          # 包含匹配
            detail_label = next((k for k in DETAIL_KEYWORDS if k in desc), "")
            is_detailed = is_abnormal and bool(detail_label)      # 参考用
            if is_normal:
                subset = "正常"
            elif is_bearing:
                subset = "轴承"
            elif is_friction:
                subset = "摩擦"
            else:
                subset = "其他异常"   # 含详细部位、故障/ng/数字/空/爆量程等
            rows.append({
                "id": r["id"],
                "storage_path": r["storage_path"],
                "file_label_status": status,
                "description": desc,
                "is_normal": int(is_normal),
                "is_abnormal": int(is_abnormal),
                "is_bearing": int(is_bearing),
                "is_friction": int(is_friction),
                "is_detailed": int(is_detailed),
                "subset": subset,
                "detailed_label": detail_label if is_detailed else "",
            })
    return rows


def subset_counts(rows: list[dict]) -> dict[str, int]:
    keys = ("is_normal", "is_abnormal", "is_bearing", "is_friction", "is_detailed")
    out = {k: sum(r[k] for r in rows) for k in keys}
    out["other_abnormal"] = sum(1 for r in rows if r["subset"] == "其他异常")
    out["total"] = len(rows)
    return out


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


# ---------------------------------------------------------------- 聚合统计

def agg_v2_normal_abnormal(results: list[dict]) -> dict:
    """v2 判决的 正常/异常 混淆矩阵（pred_normal_abnormal 由 decide_v2 给出），
    并分解异常检出的驱动来源（梳证据 / 马氏触发兜底）。"""
    tp = sum(1 for r in results if r["is_abnormal"] and r["pred_normal_abnormal"] == "异常")
    fn = sum(1 for r in results if r["is_abnormal"] and r["pred_normal_abnormal"] == "正常")
    fp = sum(1 for r in results if r["is_normal"] and r["pred_normal_abnormal"] == "异常")
    tn = sum(1 for r in results if r["is_normal"] and r["pred_normal_abnormal"] == "正常")
    tp_comb = sum(1 for r in results
                  if r["is_abnormal"] and r["pred_normal_abnormal"] == "异常"
                  and int(r["n_clean_combs"]) > 0)
    tp_maha = tp - tp_comb
    fp_comb = sum(1 for r in results
                  if r["is_normal"] and r["pred_normal_abnormal"] == "异常"
                  and int(r["n_clean_combs"]) > 0)
    return {
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "异常检出率": tp / max(tp + fn, 1),
        "正常误报率": fp / max(fp + tn, 1),
        "tp_梳驱动": tp_comb, "tp_马氏兜底": tp_maha, "fp_梳驱动": fp_comb,
        "fp_马氏兜底": fp - fp_comb,
    }


def agg_v2_bearing_friction(results: list[dict]) -> dict:
    """v2 轴承/摩擦判别：全体口径（无梳且未触发=漏检）。

    pred_fault_type ∈ {正常, 轴承, 轴承-未定位, 摩擦}；「轴承-未定位」单列，
    轴承检出率同时给 仅定位 / 含未定位 两个口径；轴承另给有梳口径定位成功率
    （判轴承/净梳>0 文件数），摩擦另给马氏触发口径（判摩擦/触发数，对应 v1
    的触发后口径）。
    """
    out = {}
    for flag in ("is_bearing", "is_friction"):
        rows = [r for r in results if r[flag]]
        n = len(rows)
        ct = {"正常": 0, "轴承": 0, "轴承-未定位": 0, "摩擦": 0}
        for r in rows:
            ct[r["pred_fault_type"]] += 1
        n_comb = sum(1 for r in rows if int(r["n_clean_combs"]) > 0)
        n_trig = sum(1 for r in rows if int(r["trigger"]) == 1)
        key = "bearing" if flag == "is_bearing" else "friction"
        if key == "bearing":
            hit_loc = ct["轴承"]
            hit_all = ct["轴承"] + ct["轴承-未定位"]
        else:
            hit_loc = hit_all = ct["摩擦"]
        out[key] = {
            "n": n, "crosstab": ct,
            "n_clean_comb": n_comb, "n_maha_triggered": n_trig,
            "全体口径检出率_仅定位": hit_loc / max(n, 1),
            "全体口径检出率_含未定位": hit_all / max(n, 1),
            "有梳口径定位成功率": (ct["轴承"] / n_comb) if key == "bearing" and n_comb else None,
            "马氏触发口径检出率": (ct["摩擦"] / n_trig) if key == "friction" and n_trig else None,
        }
    return out


def agg_low_conf(results: list[dict]) -> dict:
    rows = [r for r in results if r["low_conf"]]
    by_subset: dict[str, int] = {}
    for r in rows:
        by_subset[r["subset"]] = by_subset.get(r["subset"], 0) + 1
    return {
        "count": len(rows),
        "by_subset": by_subset,
        "ids": [r["id"] for r in rows],
    }


def agg_detailed(results: list[dict]) -> dict:
    """86 条详细部位标签上 argmax(外圈/内圈/滚珠) 与标签的一致率（参考）。"""
    rows = [r for r in results if r["is_detailed"]]
    per_class: dict[str, dict[str, int]] = {}
    mismatch_ids = []
    n_hit = 0
    n_zero = 0
    n_evid_hit = 0  # max_conf>0（复核级有谐波证据）子集内的一致数
    for r in rows:
        lab = r["detailed_label"]
        d = per_class.setdefault(lab, {"n": 0, "hit": 0})
        d["n"] += 1
        if r["max_conf"] <= 0:
            n_zero += 1
        if r["argmax_class"] == lab:
            d["hit"] += 1
            n_hit += 1
            if r["max_conf"] > 0:
                n_evid_hit += 1
        else:
            mismatch_ids.append(r["id"])
    n_evid = len(rows) - n_zero
    return {
        "n": len(rows),
        "hit": n_hit,
        "一致率": n_hit / max(len(rows), 1),
        "max_conf为零数": n_zero,
        "有证据子集": {"n": n_evid, "hit": n_evid_hit,
                    "一致率": n_evid_hit / max(n_evid, 1)},
        "per_class": per_class,
        "mismatch_ids": mismatch_ids,
    }


def agg_rpm(results: list[dict]) -> dict:
    fb = [r for r in results if r["rpm_fallback"]]
    est = np.array([r["rpm_fr_hz"] for r in results if not r["rpm_fallback"]])
    prom = np.array([r["rpm_prom_ratio"] for r in results])
    per_subset: dict[str, dict[str, float]] = {}
    for s in ("正常", "轴承", "摩擦", "其他异常"):
        sub = [r for r in results if r["subset"] == s]
        per_subset[s] = {
            "n": len(sub),
            "fallback": sum(r["rpm_fallback"] for r in sub),
        }
    out = {
        "fallback_count": len(fb),
        "fallback_rate": len(fb) / max(len(results), 1),
        "per_subset": per_subset,
        "prom_ratio_median": float(np.median(prom)),
        "prom_ratio_p90": float(np.percentile(prom, 90)),
    }
    if est.size:
        out["est_stats"] = {
            "n": int(est.size),
            "min": float(est.min()),
            "p25": float(np.percentile(est, 25)),
            "median": float(np.median(est)),
            "p75": float(np.percentile(est, 75)),
            "max": float(est.max()),
            "in_18.33_20Hz_share": float(np.mean((est >= 18.33) & (est <= 20.0))),
        }
    return out


def _iter_combs(results: list[dict]):
    """展开所有文件的梳明细（comb_detail JSON）为 (row, comb) 迭代。"""
    for r in results:
        detail = r.get("comb_detail") or "[]"
        for c in json.loads(detail):
            yield r, c


def agg_v2(results: list[dict]) -> dict:
    """v2 特有统计：检出漏斗、梳基频分布、干扰防护、无法定位、带/证一致性。"""
    funnel: dict[str, dict[str, int]] = {}
    for s in ("正常", "轴承", "摩擦", "其他异常"):
        sub = [r for r in results if r["subset"] == s]
        funnel[s] = {
            "n": len(sub),
            "A确认>0": sum(1 for r in sub if int(r["comb_confirmed"]) > 0),
            "B互证>0": sum(1 for r in sub if int(r["cepstrum_confirmed"]) > 0),
            "含干扰标记": sum(1 for r in sub if int(r["suspected_interference"]) > 0),
            "净梳>0": sum(1 for r in sub if int(r["n_clean_combs"]) > 0),
            "定位成功": sum(1 for r in sub if int(r["n_localized"]) > 0),
            "未定位": sum(1 for r in sub if int(r["bearing_unlocalized"]) == 1),
        }
    # 干扰防护明细（按原因分桶）
    guard_reasons: dict[str, int] = {}
    guard_files = set()
    comb_f0_clean: list[float] = []
    band_vs_theory = {"n_localized": 0, "有频带": 0, "带类一致": 0}
    band_cls = {"滚珠(BSF基频带)": "ball", "外圈(BPFO带)": "outer",
                "滚珠(2×BSF主峰带)": "ball", "内圈(BPFI带)": "inner"}
    for r, c in _iter_combs(results):
        if c["interference"]:
            guard_reasons[c.get("guard", "?")] = guard_reasons.get(c.get("guard", "?"), 0) + 1
            guard_files.add(r["id"])
        else:
            comb_f0_clean.append(c["f0"])
            if c.get("cls"):
                band_vs_theory["n_localized"] += 1
                if c.get("band"):
                    band_vs_theory["有频带"] += 1
                    if band_cls.get(c["band"]) == c["cls"]:
                        band_vs_theory["带类一致"] += 1
    unlocalized = [{"id": r["id"], "subset": r["subset"],
                    "comb_freqs": r["comb_freqs"], "description": r["description"]}
                   for r in results if int(r["bearing_unlocalized"]) == 1]
    return {
        "funnel": funnel,
        "guard_reasons": guard_reasons,
        "guard_files_n": len(guard_files),
        "comb_f0_clean": comb_f0_clean,
        "band_vs_theory": band_vs_theory,
        "unlocalized": unlocalized,
        "unlocalized_by_subset": {s: funnel[s]["未定位"] for s in funnel},
    }


# ---------------------------------------------------------------- 落盘

def _fmt(v):
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, float):
        return f"{v:.6g}"
    return v


def write_results_csv(results: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow({k: _fmt(r.get(k, "")) for k in CSV_COLUMNS})


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
        "normal_abnormal_v2": agg_v2_normal_abnormal(results),
        "bearing_friction_v2": agg_v2_bearing_friction(results),
        "low_conf": agg_low_conf(results),
        "detailed_reference": agg_detailed(results),
        "rpm": agg_rpm(results),
        "v2": agg_v2(results),
        "skipped_files": [{"id": s["id"], "storage_path": s["storage_path"],
                           "error": s["error"]} for s in skipped],
        "timing": {"total_s": total_s,
                   "per_file_s": total_s / max(len(results), 1)},
    }
    counts = subset_counts(results)
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

    # v2 图集（写出到本目录 figures/）
    common.setup_matplotlib()
    make_figures_v2(results, detail)
    return detail


# ---------------------------------------------------------------- 图

def _cm_figure(matrix: np.ndarray, row_labels: list[str], col_labels: list[str],
               title: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    im = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(im, ax=ax, label="样本数")
    ax.set_xticks(range(len(col_labels)), [f"判{c}" for c in col_labels])
    ax.set_yticks(range(len(row_labels)), [f"真{r}" for r in row_labels])
    for i in range(matrix.shape[0]):
        row_sum = max(matrix[i].sum(), 1)
        for j in range(matrix.shape[1]):
            v = int(matrix[i, j])
            ax.text(j, i, f"{v}\n({v / row_sum:.1%})",
                    ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def make_figures_v2(results: list[dict], detail: dict) -> None:
    """v2 图集：判决矩阵、梳基频分布（含频带与干扰线）、证据散点、距离分布。"""
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    thr = detail["calibration"]["thresholds"][f"p{MAHA_QUANTILE:g}"]

    # 1. 正常/异常混淆矩阵（v2 判决）
    na = detail["normal_abnormal_v2"]
    cm = np.array([[na["tn"], na["fp"]], [na["fn"], na["tp"]]])
    _cm_figure(cm, ["正常", "异常"], ["正常", "异常"],
               "v2 正常/异常混淆矩阵（盲梳检 + 马氏兜底）",
               FIG_DIR / "fig_cm_normal_abnormal.png")

    # 2. 轴承/摩擦判决矩阵（4 类判决）
    bf = detail["bearing_friction_v2"]
    cols = ["正常", "轴承", "轴承-未定位", "摩擦"]
    cm2 = np.array([[bf["bearing"]["crosstab"][c] for c in cols],
                    [bf["friction"]["crosstab"][c] for c in cols]])
    _cm_figure(cm2, ["轴承", "摩擦"], cols,
               "v2 复核判决矩阵（含未定位单列）",
               FIG_DIR / "fig_cm_bearing_friction.png")

    # 3. 净梳基频分布（应聚类到四个特征频带）
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    f0s = detail["v2"]["comb_f0_clean"]
    if f0s:
        ax.hist(f0s, bins=80, range=(20, 150), color="#4C72B0", alpha=0.8)
    band_style = {"ball1": ("#DD8452", "滚珠 BSF"), "outer1": ("#C44E52", "外圈 BPFO"),
                  "ball2": ("#DD8452", "滚珠 2×BSF"), "inner1": ("#55A868", "内圈 BPFI")}
    for b in class_bands(get_bearing(BEARING_NAME), RPM_WINDOW):
        color, label = band_style[f"{b['cls']}{b['k']}"]
        ax.axvspan(b["lo"], b["hi"], color=color, alpha=0.18, label=label)
    for k in range(1, 8):
        ax.axvline(k * FR_NOMINAL, color="gray", ls=":", lw=0.8)
    ax.axvline(MAINS_HZ, color="red", ls="--", lw=1, label="工频 50 Hz")
    ax.text(19.167, ax.get_ylim()[1] * 0.95, "灰点线=k×轴频", fontsize=8, color="gray")
    ax.set_xlabel("确认梳基频 f0 (Hz)")
    ax.set_ylabel("梳个数")
    ax.set_title(f"v2 净梳基频分布（n={len(f0s)}，已排除干扰标记梳）")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_comb_f0_hist.png")
    plt.close(fig)

    # 4. 马氏距离 × 最大置信度散点（v2 置信度 = 定位置信度）
    colors = {"正常": "#4C72B0", "轴承": "#C44E52", "摩擦": "#DD8452", "其他异常": "#937860"}
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for s, c in colors.items():
        sub = [r for r in results if r["subset"] == s]
        if sub:
            ax.scatter([r["maha_dist"] for r in sub],
                       [max(r["conf_outer"], r["conf_inner"], r["conf_ball"]) for r in sub],
                       s=6, alpha=0.45, color=c, label=f"{s} (n={len(sub)})")
    ax.set_xscale("log")
    ax.axvline(thr, color="gray", ls="--", lw=1, label=f"马氏触发阈值 p99={thr:.1f}")
    ax.axhline(CONF_REF, color="red", ls="--", lw=1,
               label=f"定位置信度参考线 {CONF_REF}")
    ax.set_xlabel("马氏距离（log 刻度，辅助判据）")
    ax.set_ylabel("max(外圈/内圈/滚珠 定位置信度)")
    ax.set_title("v2：马氏距离 × 定位置信度（判决主依据是梳存在性，见 results.csv）")
    ax.legend(fontsize=8, markerscale=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_scatter_maha_conf.png")
    plt.close(fig)

    # 5. 马氏距离分组分布（特征同 v1，重出便于本目录自洽）
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for s, c in colors.items():
        vals = np.array([r["maha_dist"] for r in results if r["subset"] == s])
        if vals.size:
            ax.hist(np.log10(vals + 1e-12), bins=60, alpha=0.55, color=c,
                    label=f"{s} (n={vals.size})")
    for qname, qthr in detail["calibration"]["thresholds"].items():
        ax.axvline(np.log10(qthr), ls="--", lw=1, label=f"阈值 {qname}={qthr:.1f}")
    ax.set_xlabel("log10(马氏距离)")
    ax.set_ylabel("样本数")
    ax.set_title("马氏距离分布（v2 中为辅助判据；正常组为 5 折折外距离）")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_maha_distribution.png")
    plt.close(fig)


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
    rows = load_manifest()
    mc = subset_counts(rows)
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

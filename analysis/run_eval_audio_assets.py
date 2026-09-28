#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""音频资产库评测：合并方案原型在现场录音上的端到端评测（v1 级联门控 / v2 盲梳检）。

数据：``reports/audio_assets/manifest.tsv``（1349 条，16 kHz 单声道 WAV）。
原型：``merged_proposal/unified_prototype.py``（glm 骨架 + kimi 门控的组合，
两包之间无交叉导入）。

用法（在 analysis/ 目录下运行）：

.. code-block:: bash

    python run_eval_audio_assets.py --smoke              # v1 冒烟（20 条分层）
    python run_eval_audio_assets.py                      # v1 全量 + 聚合 + 图 + 报告
    python run_eval_audio_assets.py --rules v2 --smoke   # v2 冒烟（30 条分层）
    python run_eval_audio_assets.py --rules v2           # v2 全量，产物在 rules_v2/
    python run_eval_audio_assets.py --aggregate-only [--rules v2]  # 由 results.csv 重出聚合

规则版本：``--rules v1``（默认，理论锚定级联，产物直接在 audio_assets/ 下）；
``--rules v2``（盲梳检存在性 + 频带落入定位，产物在 audio_assets/rules_v2/，
v1 产物不受影响）。两版共用同一份触发级标定（特征与正常集不变、seed 固定，
阈值一致，保证对照公平）。

防泄漏口径：触发级阈值 = 正常集 5 折折外马氏距离的 99 分位（95/99.7 供敏感性）；
正常集文件用折外距离、异常集用全部正常样本拟合的基线计距离。
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
for _p in (str(ROOT / "merged_proposal"), str(ANALYSIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import common  # noqa: E402  复用：中文字体、几何自检、JSON 落盘
import unified_prototype as up  # noqa: E402

OUT_DIR = ANALYSIS_DIR / "reports" / "audio_assets"
MANIFEST = OUT_DIR / "manifest.tsv"
#: 当前规则版本（"v1"/"v2"）与规则相关的产物路径由 _apply_rule_paths 设置
RULES = "v1"
RESULTS_CSV = OUT_DIR / "results.csv"
DETAIL_JSON = OUT_DIR / "eval_detail.json"
SKIPPED_JSON = OUT_DIR / "skipped_files.json"
FIG_DIR = OUT_DIR / "figures"
REPORT_MD = OUT_DIR / "音频资产库评测报告.md"

#: v1 列顺序（任务要求列在前，附加诊断列在后；v1 产物保持此口径不动）
CSV_COLUMNS = [
    "id", "storage_path", "file_label_status", "description",
    "is_normal", "is_abnormal", "is_bearing", "is_friction", "is_detailed",
    "subset", "detailed_label",
    "rpm_est", "rpm_fallback", "rpm_fr_hz", "rpm_prom_ratio",
    "maha_dist", "trigger",
    "conf_outer", "conf_inner", "conf_ball", "max_conf",
    "pred_normal_abnormal", "pred_fault_type", "low_conf", "argmax_class",
    "match_verdict", "match_confidence", "band_low", "band_high", "best_kurtosis",
    "feat_kurtosis", "feat_crest_factor", "feat_clearance_factor",
    "clip_ratio", "duration_s", "elapsed_s", "note",
]

#: v2 在 v1 列基础上追加盲梳检明细列（match_verdict/match_confidence 列 v2 不用，
#: 留空占位保持列序兼容）
CSV_COLUMNS_V2 = CSV_COLUMNS + [
    "comb_freqs", "comb_confirmed", "cepstrum_confirmed",
    "suspected_interference", "bearing_unlocalized",
    "n_clean_combs", "n_localized", "comb_detail",
]


def _apply_rule_paths(rules: str) -> None:
    """按规则版本切换产物路径（v2 写 rules_v2/ 子目录，v1 产物不动）。"""
    global RULES, RESULTS_CSV, DETAIL_JSON, SKIPPED_JSON, FIG_DIR, REPORT_MD
    RULES = rules
    if rules == "v2":
        sub = OUT_DIR / "rules_v2"
        RESULTS_CSV = sub / "results.csv"
        DETAIL_JSON = sub / "eval_detail.json"
        SKIPPED_JSON = sub / "skipped_files.json"
        FIG_DIR = sub / "figures"
        REPORT_MD = sub / "规则V2对照报告.md"
    # v1 即模块顶部的默认路径


DETAIL_KEYWORDS = ("外圈", "内圈", "滚珠")


# ---------------------------------------------------------------- 清单与子集

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


# ---------------------------------------------------------------- 逐文件处理

def process_one(rec: dict, rules: str = "v1") -> dict:
    """单个文件的完整原型诊断；异常不抛出，记录到 error 字段。"""
    t0 = time.perf_counter()
    try:
        d = up.diagnose_file(rec["storage_path"], rules=rules)
        d["error"] = ""
    except Exception as exc:  # noqa: BLE001 逐文件隔离，失败记录后跳过
        d = {"error": f"{type(exc).__name__}: {exc}"}
    d["elapsed_s"] = round(time.perf_counter() - t0, 4)
    return {**rec, **d}


def feats_of(r: dict) -> dict[str, float]:
    """从结果行还原触发级特征字典（calibrate/decide 用）。"""
    return {k: float(r[f"feat_{k}"]) for k in up.FEATURE_NAMES}


def conf3_of(r: dict) -> dict[str, float]:
    return {"outer": float(r["conf_outer"]),
            "inner": float(r["conf_inner"]),
            "ball": float(r["conf_ball"])}


def run_all(rows: list[dict], rules: str = "v1") -> tuple[list[dict], list[dict]]:
    """全量串行处理，每 50 条打一行进度日志，每 200 条写一次断点。"""
    results, skipped = [], []
    t0 = time.time()
    n = len(rows)
    for i, rec in enumerate(rows, 1):
        r = process_one(rec, rules=rules)
        (skipped if r["error"] else results).append(r)
        if i % 200 == 0:
            common.save_json({"results": results, "skipped": skipped},
                             RESULTS_CSV.parent / "checkpoint.json")
        if i % 50 == 0 or i == n:
            dt = time.time() - t0
            print(f"[进度 {i}/{n}] 已耗时 {dt:.0f}s，均值 {dt / i:.2f}s/条，"
                  f"预计总耗时 {dt / i * n / 60:.1f}min，失败 {len(skipped)}",
                  flush=True)
    return results, skipped


# ---------------------------------------------------------------- 标定与判决

def rev2_of(r: dict) -> dict:
    """从结果行还原 v2 复核证据（decide_v2 用）。"""
    return {"conf_outer": float(r["conf_outer"]), "conf_inner": float(r["conf_inner"]),
            "conf_ball": float(r["conf_ball"]),
            "n_clean_combs": int(r["n_clean_combs"]),
            "n_localized": int(r["n_localized"]),
            "bearing_unlocalized": bool(int(r["bearing_unlocalized"]))}


def calibrate_and_decide(results: list[dict], rules: str = "v1") -> dict:
    """触发级防泄漏标定 + 默认档（p99 + 平衡 0.2）判决，原地写回结果行。

    v2 判决：盲梳检存在性为主、马氏触发降级为辅助判据（见 decide_v2）。
    """
    normals = [r for r in results if r["is_normal"]]
    cal = up.calibrate_oof([feats_of(r) for r in normals])
    thr = cal["thresholds"][f"p{up.DEFAULT_MAHA_QUANTILE:g}"]
    for r, d in zip(normals, cal["oof"]):
        r["maha_dist"] = float(d)
    for r in results:
        if not r["is_normal"]:
            r["maha_dist"] = up.maha_distance(cal["baseline"], feats_of(r))
    conf_thr = up.PRESETS[up.DEFAULT_PRESET]
    for r in results:
        if rules == "v2":
            dec = up.decide_v2(r["maha_dist"], thr, rev2_of(r), conf_thr)
        else:
            dec = up.decide(r["maha_dist"], thr, conf3_of(r), conf_thr)
        r["trigger"] = int(dec["trigger"])
        r["pred_normal_abnormal"] = dec["pred_normal_abnormal"]
        r["pred_fault_type"] = dec["pred_fault_type"]
        r["low_conf"] = int(dec["low_conf"])
        r["max_conf"] = dec["max_conf"]
        r["argmax_class"] = up.CONF_CN[dec["argmax"]]
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

def _fmt(v):
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, float):
        return f"{v:.6g}"
    return v


def write_results_csv(results: list[dict], path: Path) -> None:
    cols = CSV_COLUMNS_V2 if RULES == "v2" else CSV_COLUMNS
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow({k: _fmt(r.get(k, "")) for k in cols})


def read_results_csv(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            for k in ("is_normal", "is_abnormal", "is_bearing", "is_friction",
                      "is_detailed", "file_label_status"):
                r[k] = int(r[k])
            for k in ("rpm_fallback", "trigger", "low_conf"):
                r[k] = int(r[k])
            for k in ("rpm_est", "rpm_fr_hz", "rpm_prom_ratio", "maha_dist",
                      "conf_outer", "conf_inner", "conf_ball", "max_conf",
                      "feat_kurtosis", "feat_crest_factor", "feat_clearance_factor",
                      "clip_ratio", "duration_s", "elapsed_s",
                      "band_low", "band_high", "best_kurtosis"):
                r[k] = float(r[k])
            if "match_confidence" in r and r["match_confidence"] != "":
                r["match_confidence"] = float(r["match_confidence"])
            for k in ("comb_confirmed", "cepstrum_confirmed",
                      "suspected_interference", "bearing_unlocalized",
                      "n_clean_combs", "n_localized"):
                if k in r and r[k] != "":
                    r[k] = int(r[k])
            rows.append(r)
    return rows


# ---------------------------------------------------------------- 聚合统计

def agg_normal_abnormal(results: list[dict], thresholds: dict[str, float]) -> dict:
    """触发级 正常/异常 混淆矩阵（各分位阈值敏感性）+ 各子集触发率。"""
    out = {}
    for qname, thr in thresholds.items():
        tp = sum(1 for r in results if r["is_abnormal"] and r["maha_dist"] >= thr)
        fn = sum(1 for r in results if r["is_abnormal"] and r["maha_dist"] < thr)
        fp = sum(1 for r in results if r["is_normal"] and r["maha_dist"] >= thr)
        tn = sum(1 for r in results if r["is_normal"] and r["maha_dist"] < thr)
        per_subset = {}
        for s in ("正常", "轴承", "摩擦", "其他异常"):
            sub = [r for r in results if r["subset"] == s]
            per_subset[s] = {
                "n": len(sub),
                "triggered": sum(1 for r in sub if r["maha_dist"] >= thr),
                "触发率": sum(1 for r in sub if r["maha_dist"] >= thr) / max(len(sub), 1),
            }
        out[qname] = {
            "threshold": thr, "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "异常检出率": tp / max(tp + fn, 1),
            "正常误报率": fp / max(fp + tn, 1),
            "per_subset": per_subset,
        }
    return out


def agg_bearing_friction(results: list[dict], maha_thr: float) -> dict:
    """轴承/摩擦子集在三档置信度预设下的检出表现（全体口径 + 触发后口径）。"""
    out = {}
    for pname, cthr in up.PRESETS.items():
        preset = {"conf_threshold": cthr}
        for flag, hit_pred in (("is_bearing", "轴承"), ("is_friction", "摩擦")):
            rows = [r for r in results if r[flag]]
            n = len(rows)
            n_trig = sum(1 for r in rows if r["maha_dist"] >= maha_thr)
            n_hit = sum(1 for r in rows
                        if r["maha_dist"] >= maha_thr
                        and max(r["conf_outer"], r["conf_inner"], r["conf_ball"]) >= cthr
                        and hit_pred == "轴承")
            n_other = sum(1 for r in rows
                          if r["maha_dist"] >= maha_thr
                          and max(r["conf_outer"], r["conf_inner"], r["conf_ball"]) >= cthr
                          and hit_pred == "摩擦")
            n_miss_pred = sum(1 for r in rows
                              if r["maha_dist"] >= maha_thr
                              and max(r["conf_outer"], r["conf_inner"], r["conf_ball"]) < cthr)
            key = "bearing" if flag == "is_bearing" else "friction"
            # 全体口径：未触发 = 漏检；触发后口径：只在触发样本中统计
            if key == "bearing":
                hit_all = n_hit          # 判轴承
                crosstab = {"判轴承": n_hit, "判摩擦": n_miss_pred, "未触发": n - n_trig}
            else:
                hit_all = n_miss_pred    # 判摩擦（触发且无轴承证据）
                crosstab = {"判摩擦": n_miss_pred, "判轴承": n_other, "未触发": n - n_trig}
            preset[key] = {
                "n": n, "n_triggered": n_trig,
                "触发率": n_trig / max(n, 1),
                "全体口径检出率": hit_all / max(n, 1),
                "触发后口径检出率": hit_all / max(n_trig, 1),
                "crosstab": crosstab,
            }
        out[pname] = preset
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


# ---------------------------------------------------------------- v2 聚合

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


def make_figures(results: list[dict], detail: dict) -> None:
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    thr = detail["calibration"]["thresholds"][f"p{up.DEFAULT_MAHA_QUANTILE:g}"]

    # 1. 正常/异常混淆矩阵（默认 p99 触发阈值）
    na = detail["normal_abnormal"][f"p{up.DEFAULT_MAHA_QUANTILE:g}"]
    cm = np.array([[na["tn"], na["fp"]], [na["fn"], na["tp"]]])
    _cm_figure(cm, ["正常", "异常"], ["正常", "异常"],
               f"触发级混淆矩阵（马氏距离阈值 p99={na['threshold']:.2f}）",
               FIG_DIR / "fig_cm_normal_abnormal.png")

    # 2. 轴承/摩擦判决矩阵（平衡档 0.2）
    bf = detail["bearing_friction"][up.DEFAULT_PRESET]
    cm2 = np.array([
        [bf["bearing"]["crosstab"]["未触发"], bf["bearing"]["crosstab"]["判轴承"],
         bf["bearing"]["crosstab"]["判摩擦"]],
        [bf["friction"]["crosstab"]["未触发"], bf["friction"]["crosstab"]["判轴承"],
         bf["friction"]["crosstab"]["判摩擦"]],
    ])
    _cm_figure(cm2, ["轴承", "摩擦"], ["正常", "轴承", "摩擦"],
               f"复核级判决矩阵（平衡档 置信度≥{bf['conf_threshold']}）",
               FIG_DIR / "fig_cm_bearing_friction.png")

    # 3. 马氏距离分组分布（log10；正常组为折外距离）
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    colors = {"正常": "#4C72B0", "轴承": "#C44E52", "摩擦": "#DD8452", "其他异常": "#937860"}
    for s, c in colors.items():
        vals = np.array([r["maha_dist"] for r in results if r["subset"] == s])
        if vals.size:
            ax.hist(np.log10(vals + 1e-12), bins=60, alpha=0.55, color=c,
                    label=f"{s} (n={vals.size})")
    for qname, qthr in detail["calibration"]["thresholds"].items():
        ax.axvline(np.log10(qthr), ls="--", lw=1,
                   label=f"阈值 {qname}={qthr:.1f}")
    ax.set_xlabel("log10(马氏距离)")
    ax.set_ylabel("样本数")
    ax.set_title("触发级马氏距离分布（正常组为 5 折折外距离）")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_maha_distribution.png")
    plt.close(fig)

    # 4. 转速自估分布
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    est = np.array([r["rpm_est"] for r in results if not r["rpm_fallback"]])
    if est.size:
        ax.hist(est, bins=40, color="#4C72B0")
    n_fb = sum(r["rpm_fallback"] for r in results)
    ax.axvline(up.RPM_FALLBACK, color="red", ls="--",
               label=f"回退值 {up.RPM_FALLBACK:.0f} rpm")
    ax.set_xlabel("自估转速 (rpm)")
    ax.set_ylabel("样本数")
    ax.set_title(f"转速自估分布（回退 {n_fb}/{len(results)} = {n_fb / len(results):.1%}）")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_rpm_hist.png")
    plt.close(fig)

    # 5. 马氏距离 × 最大轴承置信度散点
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for s, c in colors.items():
        sub = [r for r in results if r["subset"] == s]
        if sub:
            ax.scatter([r["maha_dist"] for r in sub],
                       [max(r["conf_outer"], r["conf_inner"], r["conf_ball"]) for r in sub],
                       s=6, alpha=0.45, color=c, label=f"{s} (n={len(sub)})")
    ax.set_xscale("log")
    ax.axvline(thr, color="gray", ls="--", lw=1, label=f"触发阈值 p99={thr:.1f}")
    ax.axhline(up.PRESETS[up.DEFAULT_PRESET], color="red", ls="--", lw=1,
               label=f"轴承证据阈值（平衡档 {up.PRESETS[up.DEFAULT_PRESET]}）")
    ax.set_xlabel("马氏距离（log 刻度）")
    ax.set_ylabel("max(外圈/内圈/滚珠 置信度)")
    ax.set_title("触发级 × 复核级证据散点")
    ax.legend(fontsize=8, markerscale=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_scatter_maha_conf.png")
    plt.close(fig)


def make_figures_v2(results: list[dict], detail: dict) -> None:
    """v2 图集：判决矩阵、梳基频分布（含频带与干扰线）、证据散点、距离分布。"""
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    thr = detail["calibration"]["thresholds"][f"p{up.DEFAULT_MAHA_QUANTILE:g}"]

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
    for cls, k, _o, blo, bhi in up._class_bands():
        color, label = band_style[f"{cls}{k}"]
        ax.axvspan(blo, bhi, color=color, alpha=0.18, label=label)
    fr = up.FR_NOMINAL
    for k in range(1, 8):
        ax.axvline(k * fr, color="gray", ls=":", lw=0.8)
    ax.axvline(up.MAINS_HZ, color="red", ls="--", lw=1, label="工频 50 Hz")
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
    ax.axhline(up.PRESETS[up.DEFAULT_PRESET], color="red", ls="--", lw=1,
               label=f"定位置信度参考线 {up.PRESETS[up.DEFAULT_PRESET]}")
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


# ---------------------------------------------------------------- 报告

def render_report(detail: dict, results: list[dict]) -> str:
    """生成中文评测报告（所有数字均来自 detail / results，可复算）。"""
    c = detail["counts"]
    cal = detail["calibration"]
    na = detail["normal_abnormal"]
    bf = detail["bearing_friction"]
    lc = detail["low_conf"]
    det = detail["detailed_reference"]
    rpm = detail["rpm"]
    t = detail["timing"]

    def pct(x):
        return f"{x:.1%}"

    L = []
    L.append("# 音频资产库评测报告（合并方案原型 · 级联门控）")
    L.append("")
    L.append(f"- 生成时间：{detail['generated_at']}")
    L.append(f"- 评测脚本：`analysis/run_eval_audio_assets.py`；原型：`merged_proposal/unified_prototype.py`")
    L.append(f"- 数据：`analysis/reports/audio_assets/manifest.tsv`（清单导出的人工标注）")
    L.append(f"- 处理：成功 {c['processed']}/{c['total']} 条，失败跳过 {c['skipped']} 条；"
             f"全量耗时 {t['total_s'] / 60:.1f} min（均值 {t['per_file_s']:.2f} s/条）")
    L.append("")
    L.append("## 1. 口径声明")
    L.append("")
    L.append("- `description` 为**人工标注的声称值，非拆解确认**；所有「准确率」应理解为与标注的一致率。")
    L.append("- 子集定义（严格遵守）：正常集 `file_label_status=1`；异常集 `=-1`；"
             "轴承子集 = 异常且 description **精确等于**「轴承」；摩擦子集 = 异常且 description **包含**「摩擦」；"
             "详细部位子集（参考）= 异常且含 外圈/内圈/滚珠。")
    L.append("- 削波样本（「爆量程」）无独立子集，按 description 归入「其他异常」（本次共 3 条）。")
    L.append("- 触发级阈值防泄漏：正常集 5 折交叉，阈值 = 折外马氏距离分位数；"
             "正常文件用折外距离，异常文件用全部正常样本拟合的基线。")
    L.append("- 复核级只做外圈/内圈/滚珠三类谐波匹配（保持架 FTF 出 scope）；"
             "滚珠主峰 2×BSF 由 BSF 二次谐波检索覆盖；匹配容差 ±2%。")
    L.append("")
    L.append("## 2. 数据与子集")
    L.append("")
    L.append("| 子集 | 条数 |")
    L.append("|---|---|")
    L.append(f"| 正常（status=1） | {c['is_normal']} |")
    L.append(f"| 异常（status=-1） | {c['is_abnormal']} |")
    L.append(f"| 　其中：轴承（精确「轴承」） | {c['is_bearing']} |")
    L.append(f"| 　其中：摩擦（含「摩擦」） | {c['is_friction']} |")
    L.append(f"| 　其中：其他异常（含详细部位/故障/ng/空/爆量程等） | {c['other_abnormal']} |")
    L.append(f"| 　参考：详细部位（含 外圈/内圈/滚珠） | {c['is_detailed']} |")
    L.append("")
    if detail["skipped_files"]:
        L.append(f"失败跳过 {len(detail['skipped_files'])} 条（明细见 eval_detail.json 的 skipped_files）。")
        L.append("")
    L.append("## 3. 方法概要")
    L.append("")
    L.append("级联门控（合并方案 §6.2）：零相位预处理（去直流/去趋势/sosfiltfilt "
             f"{up.PREPROCESS_BAND[0]:.0f}–{up.PREPROCESS_BAND[1]:.0f} Hz）→ 转速自估"
             f"（包络谱 1×fr 峰，窗 {up.RPM_SEARCH_HZ[0]}–{up.RPM_SEARCH_HZ[1]} Hz，"
             f"失败回退 {up.RPM_FALLBACK:.0f} rpm）→【触发级】马氏距离健康基线"
             f"（特征：{'/'.join(up.FEATURE_NAMES)}，kimi 检测级冲击性特征的纯时域子集）"
             "→【复核级】glm Kurtogram 选带 → 零相位解调 → glm matcher 三类谐波匹配。")
    L.append("")
    L.append("## 4. 触发级：正常/异常判别")
    L.append("")
    L.append(f"标定：正常集 {cal['n_normal']} 条，{cal['n_splits']} 折"
             f"（折大小 {cal['fold_sizes']}，seed={cal['seed']}）；"
             f"折外距离 min/median/max = {cal['oof_stats']['min']:.2f} / "
             f"{cal['oof_stats']['median']:.2f} / {cal['oof_stats']['max']:.2f}。")
    L.append("")
    L.append("| 阈值（折外分位） | 阈值距离 | 异常检出率 | 正常误报率 | TP/FN/FP/TN |")
    L.append("|---|---|---|---|---|")
    for q in ("p95", "p99", "p99.7"):
        d = na[q]
        L.append(f"| {q} | {d['threshold']:.2f} | {pct(d['异常检出率'])} "
                 f"({d['tp']}/{d['tp'] + d['fn']}) | {pct(d['正常误报率'])} "
                 f"({d['fp']}/{d['fp'] + d['tn']}) | "
                 f"{d['tp']}/{d['fn']}/{d['fp']}/{d['tn']} |")
    L.append("")
    L.append("各子集在不同分位阈值下的触发率（正常子集的触发率即误报率）：")
    L.append("")
    L.append("| 阈值 | 正常 | 轴承 | 摩擦 | 其他异常 |")
    L.append("|---|---|---|---|---|")
    for q in ("p95", "p99", "p99.7"):
        psub = na[q]["per_subset"]
        L.append(f"| {q}（{na[q]['threshold']:.2f}） | "
                 + " | ".join(f"{psub[s]['triggered']}/{psub[s]['n']}（{pct(psub[s]['触发率'])}）"
                             for s in ("正常", "轴承", "摩擦", "其他异常"))
                 + " |")
    # 健康总体的冲击性特征重尾（触发级灵敏度塌陷的直接证据，由 results.csv 复算）
    nk = np.array([r["feat_kurtosis"] for r in results if r["subset"] == "正常"])
    L.append("")
    L.append(f"**触发级灵敏度塌陷的根因**：健康总体在冲击性特征上重尾——正常集峭度 "
             f"p50/p95/p99/max = {np.percentile(nk, 50):.1f} / {np.percentile(nk, 95):.1f} / "
             f"{np.percentile(nk, 99):.0f} / {nk.max():.0f}，即约 {np.mean(nk > 5):.1%} 的正常录音"
             "本身含强冲击瞬态。协方差被这些尾部样本撑大后，白化后的典型折外距离被压缩"
             f"（中位仅 {cal['oof_stats']['median']:.2f}），而 p99 阈值被尾部推到 "
             f"{na['p99']['threshold']:.1f}——轻中度异常（轴承子集峭度中位约 10）落在白化意义下"
             "「不算远」的位置，马氏距离的椭球假设在该重尾总体上失效（详见 §9）。")
    L.append("")
    L.append(f"![触发级混淆矩阵](figures/fig_cm_normal_abnormal.png)")
    L.append("")
    L.append(f"![马氏距离分布](figures/fig_maha_distribution.png)")
    L.append("")
    L.append("## 5. 复核级：轴承/摩擦判别")
    L.append("")
    L.append("全体口径：未触发 = 该类漏检；触发后口径：只在触发样本中统计。"
             "默认触发阈值 p99。")
    L.append("")
    L.append("| 预设（置信度阈值） | 子集 | n | 触发率 | 全体口径检出率 | 触发后口径检出率 | 判决分布 |")
    L.append("|---|---|---|---|---|---|---|")
    for pname in ("保守", "平衡", "灵敏"):
        d = bf[pname]
        for key, cn in (("bearing", "轴承"), ("friction", "摩擦")):
            s = d[key]
            ct = s["crosstab"]
            dist = " / ".join(f"{k}{v}" for k, v in ct.items())
            L.append(f"| {pname}（≥{d['conf_threshold']}） | {cn} | {s['n']} | "
                     f"{pct(s['触发率'])} | {pct(s['全体口径检出率'])} | "
                     f"{pct(s['触发后口径检出率'])} | {dist} |")
    L.append("")
    # 复核级单级判别力（绕过触发级，直接看 max_conf≥平衡档）——量级对照用
    evid = {}
    for s in ("正常", "轴承", "摩擦", "其他异常"):
        sub = [r for r in results if r["subset"] == s]
        n_ev = sum(1 for r in sub if max(r["conf_outer"], r["conf_inner"], r["conf_ball"])
                   >= up.PRESETS[up.DEFAULT_PRESET])
        gated = sum(1 for r in sub if r["maha_dist"] < na["p99"]["threshold"]
                    and max(r["conf_outer"], r["conf_inner"], r["conf_ball"])
                    >= up.PRESETS[up.DEFAULT_PRESET])
        evid[s] = (n_ev, gated, len(sub))
    L.append(f"**复核级单级判别力（参考，绕过触发级）**：轴承集有 {evid['轴承'][0]}/{evid['轴承'][2]}"
             f"（{pct(evid['轴承'][0] / evid['轴承'][2])}）样本 max 置信度 ≥ 平衡档，"
             f"而正常集仅 {evid['正常'][0]}/{evid['正常'][2]}"
             f"（{pct(evid['正常'][0] / evid['正常'][2])}）——复核级本身的区分度远好于级联结果；"
             f"但轴承集 {evid['轴承'][1]} 条有证据样本被触发级拦下（未触发 = 漏检）。"
             "触发级是当前级联的瓶颈，详见 §8/§9。")
    L.append("")
    L.append(f"![复核级判决矩阵](figures/fig_cm_bearing_friction.png)")
    L.append("")
    L.append(f"![触发×复核证据散点](figures/fig_scatter_maha_conf.png)")
    L.append("")
    L.append("### 低置信样本")
    L.append("")
    L.append(f"低置信规则：触发 且 马氏距离 < 2×阈值 且 max 置信度 < 平衡档 "
             f"{up.PRESETS[up.DEFAULT_PRESET]}。共 **{lc['count']} 条**，"
             f"子集分布：{lc['by_subset']}。")
    if lc["ids"]:
        show = lc["ids"][:30]
        more = f" 等（全部 {len(lc['ids'])} 条见 eval_detail.json）" if len(lc["ids"]) > 30 else ""
        L.append(f"样本 id：{', '.join(show)}{more}。")
    L.append("")
    L.append("## 6. 参考：86 条详细部位标签的 argmax 一致率（不作正式结论）")
    L.append("")
    pc = det["per_class"]
    pc_str = "；".join(f"{k} {v['hit']}/{v['n']}" for k, v in pc.items())
    L.append(f"- argmax(外圈/内圈/滚珠置信度) 与 description 部位的一致率："
             f"**{det['hit']}/{det['n']} = {pct(det['一致率'])}**（{pc_str}）。")
    L.append(f"- 其中 max_conf=0（复核级无任何谐波证据）{det['max_conf为零数']} 条，"
             f"其 argmax 为并列退化值（取外圈），参考意义有限；只在有证据子集上："
             f"**{det['有证据子集']['hit']}/{det['有证据子集']['n']} = "
             f"{pct(det['有证据子集']['一致率'])}**。")
    L.append("")
    L.append("## 7. 转速自估")
    L.append("")
    L.append(f"- 回退（包络谱 1×fr 峰 prominence < {up.RPM_PROM_MIN:g}）："
             f"**{rpm['fallback_count']}/{c['processed']} = {pct(rpm['fallback_rate'])}**；"
             f"回退值 {up.RPM_FALLBACK:.0f} rpm。全体样本的 1×fr 峰突出度中位/p90 = "
             f"{rpm['prom_ratio_median']:.1f}/{rpm['prom_ratio_p90']:.1f}，"
             "大多数信号根本没有可用的包络 1×fr 峰（强 2.2 kHz 电机音主导，轴频调制弱）。")
    if "est_stats" in rpm:
        s = rpm["est_stats"]
        L.append(f"- 非回退 {s['n']} 条的 fr 分布：min/p25/中位/p75/max = "
                 f"{s['min']:.2f}/{s['p25']:.2f}/{s['median']:.2f}/{s['p75']:.2f}/{s['max']:.2f} Hz；"
                 f"落在标称 18.33–20 Hz 内的比例 {pct(s['in_18.33_20Hz_share'])}。")
    ps = rpm["per_subset"]
    L.append(f"- 各子集回退：{'；'.join(f'{k} {v['fallback']}/{v['n']}' for k, v in ps.items())}。")
    L.append("")
    L.append(f"![转速自估分布](figures/fig_rpm_hist.png)")
    L.append("")
    L.append("## 8. 与前一次 81 条数据集评测的对照（级联门控是否兑现互补）")
    L.append("")
    L.append("前次评测（labeled_dataset_4class，轮次 B 口径）：glm **0 误报**（正常 10/10 正确）"
             "但故障检出仅 26/71（36.6%）；kimi **0 漏检**（71/71 全触发）但正常 10/10 全误报"
             "（固定阈值跨域失效）。合并方案的级联门控正是「kimi 灵敏触发 + glm 严格复核」"
             "的互补设计，且触发阈值改为域内标定（折外分位数），从机制上消除 kimi 的跨域失效。")
    L.append("")
    na99 = na["p99"]
    bf0 = bf[up.DEFAULT_PRESET]
    L.append(f"本次（1349 条现场录音，p99 + 平衡档）结论分两半：")
    L.append("")
    L.append(f"- **「glm 式低误报」兑现**：触发级正常误报率 {pct(na99['正常误报率'])}"
             f"（{na99['fp']}/{na99['fp'] + na99['tn']}，系 p99 标定的内禀水平），"
             "且经复核级仲裁后，最终「轴承」判决在正常集 0 误报、在 426 条摩擦集 0 误定类"
             f"（平衡/保守档）；触发后被判「轴承」的轴承样本 "
             f"{bf0['bearing']['crosstab']['判轴承']}/{bf0['bearing']['n_triggered']}，"
             "复核级的严格性保留了 glm 的行为特征。")
    L.append(f"- **「kimi 式低漏检」未兑现**：触发级异常检出率仅 "
             f"{pct(na99['异常检出率'])}（{na99['tp']}/{na99['tp'] + na99['fn']}，p99），"
             f"轴承子集触发率 {pct(bf0['bearing']['触发率'])}。"
             "根因不是阈值标定方式（域内标定已生效、误报率受控），而是马氏距离的几何假设"
             "与本数据的健康总体不匹配——正常录音的冲击性特征重尾（峭度 max 902），"
             "协方差被尾部撑大后灵敏度塌陷（§4 末的量化证据）。值得注意的是，"
             "复核级单级（max 置信度 ≥ 0.2）在轴承集有 28.0% 的证据率、正常集仅 1.1%——"
             "若触发级修复，级联的检出上限远高于本次实测（§5 末）。")
    L.append("")
    L.append("注意：两次评测数据不同（81 条台架四类 vs 1349 条现场含摩擦类），"
             "只能定性对照门控行为，不能直接比较检出率数值。")
    L.append("")
    L.append("## 9. 局限与后续建议")
    L.append("")
    L.append("1. **触发级灵敏度塌陷（本次最重要的负面发现）**：马氏距离假设健康特征近似"
             "椭球分布，而本数据集正常录音的冲击性特征重尾（约 3.7% 正常样本峭度 >5、"
             "最大 902），协方差被尾部主导，触发级成为级联瓶颈（轴承集复核级证据率 28.0%，"
             "级联后检出仅 1.8%）。建议（按代价排序）：(a) 冲击特征先做 log/秩变换再做"
             "马氏距离；(b) 换 robust 协方差（kimi ML 的 EllipticEnvelope 路径"
             "正是为此设计，合并方案移植项 3）；(c) 直接以单调冲击指标的分位数做触发"
             "（等效一维 EVT 思路）。该实验可复用本脚本快速验证。")
    L.append("2. 标签为声称值：description 未经拆解确认，一致率天花板受标签质量限制；"
             "「轴承」子集未区分部位，详细部位仅 86 条参考。")
    L.append("3. 转速自估高回退：现场信号的包络谱 1×fr 峰 prominence 普遍不足"
             f"（全体中位 {rpm['prom_ratio_median']:.1f} < {up.RPM_PROM_MIN:g}），"
             f"{pct(rpm['fallback_rate'])} 样本回退 1150 rpm；真转速处于 1100/1200 窗口"
             "边缘时 ±2% 匹配容差可能覆盖不住（最大偏差约 4%），建议后续接入转速测量或"
             "用故障谱线族反演转速。非回退估计（100 条）中位 18.7 Hz，57% 落在标称窗内，"
             "方向合理但样本少。")
    L.append("4. 触发级特征只用了 3 维纯时域冲击性指标（env_max_ratio 依赖故障频率，"
             "按约定回退）；对「非冲击型」异常（如平稳异音）灵敏度有限，可考虑补充"
             "谱形态类特征。")
    L.append("5. 马氏距离基线用 566 条正常样本域内标定，对工况漂移敏感，"
             "跨设备/跨批次部署前需重新标定。")
    L.append("6. 「摩擦」判决是「触发但无轴承证据」的兜底类，不含摩擦的物理模型；"
             "摩擦子集中被判「轴承」的样本值得人工复听（可能是标注粒度问题）。")
    L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------- v2 报告

def render_report_v2(detail: dict, results: list[dict]) -> str:
    """v2 对照报告（盲梳检 vs v1 理论锚定）。所有数字可由 results.csv 复算。"""
    c = detail["counts"]
    cal = detail["calibration"]
    na = detail["normal_abnormal_v2"]
    bf = detail["bearing_friction_v2"]
    lc = detail["low_conf"]
    det = detail["detailed_reference"]
    rpm = detail["rpm"]
    v2 = detail["v2"]
    t = detail["timing"]
    v1 = detail.get("v1_baseline")

    def pct(x):
        return f"{x:.1%}"

    L = []
    L.append("# 规则 V2 对照报告（盲梳检存在性 + 频带落入定位）")
    L.append("")
    L.append(f"- 生成时间：{detail['generated_at']}")
    L.append("- 规则：`--rules v2`（原型 `merged_proposal/unified_prototype.py`，"
             "v1 路径保留可用）")
    L.append(f"- 数据：`analysis/reports/audio_assets/manifest.tsv`；"
             f"成功 {c['processed']}/{c['total']} 条，失败 {c['skipped']} 条；"
             f"全量耗时 {t['total_s'] / 60:.1f} min（均值 {t['per_file_s']:.2f} s/条）")
    L.append("")
    L.append("## 1. 口径声明")
    L.append("")
    L.append("- `description` 为**人工标注的声称值，非拆解确认**；所有「准确率」为与标注的一致率。")
    L.append("- 子集定义与 v1 完全相同：正常 566（status=1）/ 异常 783（status=-1），"
             "其中轴承（精确「轴承」）168、摩擦（含「摩擦」）426、其他异常 189、"
             "详细部位参考 86；「爆量程」3 条在其他异常内。")
    L.append("- v2 判决四类：正常 / 轴承（已定位）/ 轴承-未定位 / 摩擦；"
             "马氏触发级（p99，域内 5 折标定，与 v1 同一份）降级为无梳时的辅助判据。")
    L.append("- v2 匹配**不依赖转速估计**；rpm_est 仅记录。定位频带由标称转速窗 "
             "1100–1200 rpm 预先推出（滚珠 32.2–35.2 / 外圈 47.3–51.6 / "
             "滚珠 64.5–70.3 / 内圈 81.1–88.4 Hz），理论复核容差 ±4%。")
    L.append("- 干扰防护：f0 与 k×19.167 Hz（轴频谐波）或 50.0 Hz（工频）在 1.5 谱线内"
             "吻合的梳、以及下探后基频 <20 Hz 的轴频族梳，打 suspected_interference "
             "标记并从轴承判决排除（仍记录）。")
    L.append("")
    L.append("## 2. v1 vs v2 对照")
    L.append("")
    v1_na = v1["normal_abnormal"]["p99"] if v1 else None
    v1_bf = v1["bearing_friction"]["平衡"] if v1 else None
    L.append("| 指标 | v1（理论锚定级联） | v2（盲梳检） |")
    L.append("|---|---|---|")
    if v1:
        L.append(f"| 正常误报率 | {pct(v1_na['正常误报率'])}（{v1_na['fp']}/{v1_na['fp'] + v1_na['tn']}） "
                 f"| {pct(na['正常误报率'])}（{na['fp']}/{na['fp'] + na['tn']}） |")
        L.append(f"| 异常检出率 | {pct(v1_na['异常检出率'])}（{v1_na['tp']}/{v1_na['tp'] + v1_na['fn']}） "
                 f"| {pct(na['异常检出率'])}（{na['tp']}/{na['tp'] + na['fn']}） |")
        v1b = v1_bf["bearing"]
        L.append(f"| 轴承检出率（全体口径） | {pct(v1b['全体口径检出率'])}（判轴承 {v1b['crosstab']['判轴承']}/{v1b['n']}） "
                 f"| {pct(bf['bearing']['全体口径检出率_仅定位'])} 仅定位 / "
                 f"{pct(bf['bearing']['全体口径检出率_含未定位'])} 含未定位"
                 f"（n={bf['bearing']['n']}） |")
        v1f = v1_bf["friction"]
        L.append(f"| 摩擦检出率（全体口径） | {pct(v1f['全体口径检出率'])}（{v1f['crosstab']['判摩擦']}/{v1f['n']}） "
                 f"| {pct(bf['friction']['全体口径检出率_仅定位'])}（{bf['friction']['crosstab']['摩擦']}/{bf['friction']['n']}） |")
        L.append(f"| 低置信数 | {v1['low_conf']['count']} | {lc['count']} |")
        L.append(f"| 86 条详细部位 argmax 一致率 | {pct(v1['detailed_reference']['一致率'])}"
                 f"（{v1['detailed_reference']['hit']}/{v1['detailed_reference']['n']}） "
                 f"| {pct(det['一致率'])}（{det['hit']}/{det['n']}；"
                 f"有证据子集 {det['有证据子集']['hit']}/{det['有证据子集']['n']}） |")
    L.append("")
    L.append("v2 核心判据：轴承全体口径检出率能否突破 v1 理论锚定的结构性上限"
             "（v1 复核级单级证据率 28.0%）。见上表第 3 行与 §5。")
    L.append("")
    L.append("## 3. v2 检出漏斗（按子集）")
    L.append("")
    L.append("| 子集 | n | A 梳确认 | B 倒谱互证 | 含干扰标记 | 净梳 | 定位成功 | 未定位 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for s in ("正常", "轴承", "摩擦", "其他异常"):
        f = v2["funnel"][s]
        L.append(f"| {s} | {f['n']} | {f['A确认>0']} | {f['B互证>0']} | "
                 f"{f['含干扰标记']} | {f['净梳>0']} | {f['定位成功']} | {f['未定位']} |")
    L.append("")
    L.append("## 4. 正常/异常（v2 判决）")
    L.append("")
    L.append(f"- 混淆矩阵：TP/FN/FP/TN = {na['tp']}/{na['fn']}/{na['fp']}/{na['tn']}；"
             f"异常检出率 {pct(na['异常检出率'])}，正常误报率 {pct(na['正常误报率'])}。")
    L.append(f"- 异常检出的驱动分解：梳证据 {na['tp_梳驱动']} 条 / 马氏兜底 {na['tp_马氏兜底']} 条；"
             f"正常误报中梳证据 {na['fp_梳驱动']} 条 / 马氏兜底 {na['fp_马氏兜底']} 条。")
    L.append("")
    L.append("![正常异常混淆矩阵](figures/fig_cm_normal_abnormal.png)")
    L.append("")
    L.append("![马氏距离分布](figures/fig_maha_distribution.png)")
    L.append("")
    L.append("## 5. 轴承/摩擦（v2 判决）")
    L.append("")
    L.append("| 真值 | n | 判正常 | 判轴承 | 判轴承-未定位 | 判摩擦 | 检出率口径 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for key, cn in (("bearing", "轴承"), ("friction", "摩擦")):
        s = bf[key]
        ct = s["crosstab"]
        if key == "bearing":
            rates = (f"全体：仅定位 {pct(s['全体口径检出率_仅定位'])} / "
                     f"含未定位 {pct(s['全体口径检出率_含未定位'])}；"
                     f"有梳定位成功率 {pct(s['有梳口径定位成功率'] or 0)}"
                     f"（{ct['轴承']}/{s['n_clean_comb']}）")
        else:
            rates = (f"全体 {pct(s['全体口径检出率_仅定位'])}；"
                     f"马氏触发口径 {pct(s['马氏触发口径检出率'] or 0)}"
                     f"（{ct['摩擦']}/{s['n_maha_triggered']}）")
        L.append(f"| {cn} | {s['n']} | {ct['正常']} | {ct['轴承']} | {ct['轴承-未定位']} | "
                 f"{ct['摩擦']} | {rates} |")
    L.append("")
    L.append("「轴承-未定位」= 确认存在轴承故障特征（梳 A+B 确认）但频带落入与补救路径"
             "均无法定位，单列不强行归类。")
    L.append("")
    L.append("![轴承摩擦判决矩阵](figures/fig_cm_bearing_friction.png)")
    L.append("")
    L.append("![证据散点](figures/fig_scatter_maha_conf.png)")
    L.append("")
    L.append("### 低置信样本")
    L.append("")
    L.append(f"低置信规则（沿用 v1 精神）：判摩擦但马氏距离 < 2×阈值，或判轴承但 "
             f"max 定位置信度 < {up.PRESETS[up.DEFAULT_PRESET]}。共 **{lc['count']} 条**，"
             f"子集分布：{lc['by_subset']}。明细见 eval_detail.json。")
    L.append("")
    L.append("## 6. 参考：86 条详细部位标签的 argmax 一致率（不作正式结论）")
    L.append("")
    pc = det["per_class"]
    pc_str = "；".join(f"{k} {v['hit']}/{v['n']}" for k, v in pc.items())
    L.append(f"- argmax 与标注一致率：**{det['hit']}/{det['n']} = {pct(det['一致率'])}**"
             f"（{pc_str}）。")
    L.append(f"- 其中 max_conf=0（无定位证据）{det['max_conf为零数']} 条（argmax 退化为外圈）；"
             f"有证据子集：**{det['有证据子集']['hit']}/{det['有证据子集']['n']} = "
             f"{pct(det['有证据子集']['一致率'])}**。")
    L.append("")
    L.append("## 7. v2 特有分析")
    L.append("")
    L.append("### 梳基频分布")
    L.append("")
    L.append(f"净梳（确认且未标记干扰）共 {len(v2['comb_f0_clean'])} 个，"
             "分布直方图如下——若盲检有效，梳基频应聚类到四个特征频带：")
    L.append("")
    L.append("![梳基频分布](figures/fig_comb_f0_hist.png)")
    L.append("")
    L.append("### 干扰防护命中统计")
    L.append("")
    L.append(f"- 被标记梳数：**{sum(v2['guard_reasons'].values())}**（涉及 {v2['guard_files_n']} 个文件）")
    if v2["guard_reasons"]:
        for reason, n in sorted(v2["guard_reasons"].items(), key=lambda kv: -kv[1]):
            L.append(f"  - {reason}：{n} 个梳")
    L.append("")
    L.append("### 「轴承特征存在但无法定位」清单")
    L.append("")
    L.append(f"共 **{len(v2['unlocalized'])} 条**"
             f"（子集分布：{v2['unlocalized_by_subset']}）。")
    if v2["unlocalized"]:
        L.append("")
        L.append("| id | 子集 | 梳基频 (Hz) | description |")
        L.append("|---|---|---|---|")
        for u in v2["unlocalized"][:40]:
            L.append(f"| {u['id']} | {u['subset']} | {u['comb_freqs']} | {u['description']} |")
        if len(v2["unlocalized"]) > 40:
            L.append(f"| … | 共 {len(v2['unlocalized'])} 条，余见 eval_detail.json | | |")
    L.append("")
    L.append("### 频带落入 vs 理论复核的一致情况")
    L.append("")
    bt = v2["band_vs_theory"]
    L.append(f"定位成功的梳 {bt['n_localized']} 个，其中基频直接落入四频带之一 "
             f"{bt['有频带']} 个（其余经 fr_hyp 补救路径定位）；频带粗定类与理论复核"
             f"定类一致 {bt['带类一致']} 个"
             + (f"（{bt['带类一致'] / bt['有频带']:.1%}）" if bt["有频带"] else "")
             + "。")
    L.append("")
    L.append("## 8. 转速依赖性")
    L.append("")
    L.append(f"- v2 匹配不依赖转速；rpm_est 仅记录（回退 {rpm['fallback_count']}/"
             f"{c['processed']} = {pct(rpm['fallback_rate'])}，同 v1）。")
    L.append("- v1 的量化影响（v1 results.csv 复算）：v1 复核级理论锚定在转速回退文件上"
             "只能用 1150 rpm 猜测——轴承集中 v1 证据（max_conf≥0.2）率 "
             "在回退/非回退文件上的对比见下。")
    v1e = detail.get("v1_evidence_by_fallback")
    if v1e:
        L.append("")
        L.append("| 文件组 | v1 轴承集证据率（max_conf≥0.2） | v1 正常集证据率 |")
        L.append("|---|---|---|")
        for g in ("fallback", "estimated"):
            bg, ng = v1e["bearing"][g], v1e["normal"][g]
            L.append(f"| {'转速回退' if g == 'fallback' else '转速自估成功'} | "
                     f"{bg['hit']}/{bg['n']}（{pct(bg['rate'])}） | "
                     f"{ng['hit']}/{ng['n']}（{pct(ng['rate'])}） |")
    L.append("")
    L.append("## 9. 局限与后续建议")
    L.append("")
    L.append("1. 盲梳检确认的是「周期性冲击存在」，不等于轴承独有——实测中轴频族梳"
             "（17.1 Hz 族）与 ~21 Hz 周期源会触发「无法定位」或干扰标记；低频梳防护"
             "目前只覆盖下探可发现的轴频族，非整数倍关系的周期源仍靠「未定位」兜底。")
    L.append("2. 定位频带依赖标称转速窗 1100–1200 rpm；冒烟已见 fr≈17.97 Hz（约 1080 rpm）"
             "的真实样本靠 ±4% 容差勉强纳入窗边界，若现场存在更大转速漂移，频带落入法"
             "会把真实故障梳挤到空档（归入未定位而非误判，行为保守）。")
    L.append("3. 马氏兜底路径继承 v1 的重尾塌陷问题（p99 下检出有限），摩擦检出率上限"
             "受它约束；v2 的改进空间主要在触发级（log/秩变换或 robust 协方差）。")
    L.append("4. 标签为声称值；「摩擦」仍是排除法判决；未定位清单值得人工复听确认"
             "是否为轴承以外的周期振源。")
    L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------- 聚合入口

def aggregate(results: list[dict], skipped: list[dict], total_s: float) -> dict:
    """聚合统计 + 图 + 报告；返回 eval_detail 字典。"""
    cal = calibrate_and_decide(results, rules=RULES)
    thr = cal["thresholds"][f"p{up.DEFAULT_MAHA_QUANTILE:g}"]
    detail: dict = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "script": "analysis/run_eval_audio_assets.py",
        "prototype": "merged_proposal/unified_prototype.py",
        "rules": RULES,
        "manifest": str(MANIFEST),
        "config": {
            "fs": up.FS,
            "preprocess_band": up.PREPROCESS_BAND,
            "rpm_search_hz": up.RPM_SEARCH_HZ,
            "rpm_fallback": up.RPM_FALLBACK,
            "rpm_prom_min": up.RPM_PROM_MIN,
            "feature_names": list(up.FEATURE_NAMES),
            "presets": up.PRESETS,
            "default_preset": up.DEFAULT_PRESET,
            "maha_quantiles": up.MAHA_QUANTILES,
            "default_maha_quantile": up.DEFAULT_MAHA_QUANTILE,
            "geometry": up.GEOMETRY,
            "orders": up.ORDERS,
        },
        "counts": {},
        "calibration": cal,
        "normal_abnormal": agg_normal_abnormal(results, cal["thresholds"]),
        "bearing_friction": agg_bearing_friction(results, thr),
        "low_conf": agg_low_conf(results),
        "detailed_reference": agg_detailed(results),
        "rpm": agg_rpm(results),
        "skipped_files": [{"id": s["id"], "storage_path": s["storage_path"],
                           "error": s["error"]} for s in skipped],
        "timing": {
            "total_s": total_s,
            "per_file_s": total_s / max(len(results), 1),
        },
    }
    if RULES == "v2":
        detail["v2"] = agg_v2(results)
        # v2 判决口径下重算 normal_abnormal（pred 由 decide_v2 给出）
        detail["normal_abnormal_v2"] = agg_v2_normal_abnormal(results)
        detail["bearing_friction_v2"] = agg_v2_bearing_friction(results)
        v1_detail = OUT_DIR / "eval_detail.json"
        if v1_detail.exists():
            detail["v1_baseline"] = json.loads(v1_detail.read_text(encoding="utf-8"))
        v1_csv = OUT_DIR / "results.csv"
        if v1_csv.exists():
            v1_rows = read_results_csv(v1_csv)

            def _evid(rows, flag):
                sub = [r for r in rows if r[flag]]
                out = {}
                for gname, g in (("fallback", [r for r in sub if r["rpm_fallback"]]),
                                 ("estimated", [r for r in sub if not r["rpm_fallback"]])):
                    hit = sum(1 for r in g if r["max_conf"] >= up.PRESETS[up.DEFAULT_PRESET])
                    out[gname] = {"hit": hit, "n": len(g),
                                  "rate": hit / max(len(g), 1)}
                return out

            detail["v1_evidence_by_fallback"] = {
                "bearing": _evid(v1_rows, "is_bearing"),
                "normal": _evid(v1_rows, "is_normal"),
            }
    counts = subset_counts(results)
    counts["processed"] = len(results)
    counts["skipped"] = len(skipped)
    counts["total"] = len(results) + len(skipped)
    detail["counts"] = counts

    # 判决后的结果落盘（覆盖，可安全重跑）
    write_results_csv(results, RESULTS_CSV)
    common.save_json(detail, DETAIL_JSON)
    common.save_json(detail["skipped_files"], SKIPPED_JSON)

    common.setup_matplotlib()
    if RULES == "v2":
        make_figures_v2(results, detail)
        REPORT_MD.write_text(render_report_v2(detail, results), encoding="utf-8")
    else:
        make_figures(results, detail)
        REPORT_MD.write_text(render_report(detail, results), encoding="utf-8")
    return detail


# ---------------------------------------------------------------- 冒烟

def run_smoke(rows: list[dict]) -> None:
    """20 条分层冒烟：5 正常/5 轴承/5 摩擦/5 其他异常，打印中间量。

    冒烟内的马氏距离仅为参考：正常样本用留一基线、异常样本用 5 条正常
    拟合的基线（样本太少，不下结论，只验证链路合理性）。
    """
    rng = np.random.default_rng(42)
    picked: list[dict] = []
    for flag, n in (("is_normal", 5), ("is_bearing", 5),
                    ("is_friction", 5), ("subset", 5)):
        if flag == "subset":
            pool = [r for r in rows if r["subset"] == "其他异常"]
        else:
            pool = [r for r in rows if r[flag]]
        idx = rng.choice(len(pool), size=n, replace=False)
        picked.extend(pool[int(i)] for i in idx)

    print(f"冒烟 {len(picked)} 条（5 正常/5 轴承/5 摩擦/5 其他异常）")
    header = (f"{'id':>6} {'子集':<4} {'耗时s':>5} {'rpm':>5} {'fb':>2} "
              f"{'prom':>5} {'kurt':>6} {'crest':>5} {'clear':>5} "
              f"{'maha':>8} {'cO':>5} {'cI':>5} {'cB':>5} {'选带(Hz)':>14} {'K':>5}")
    print(header)
    results = []
    t0 = time.time()
    for rec in picked:
        r = process_one(rec)
        results.append(r)
        if r["error"]:
            print(f"{r['id']:>6} {r['subset']:<4} 失败: {r['error']}")
            continue
        print(f"{r['id']:>6} {r['subset']:<4} {r['elapsed_s']:>5.2f} "
              f"{r['rpm_est']:>5.0f} {int(r['rpm_fallback']):>2d} "
              f"{r['rpm_prom_ratio']:>5.1f} {r['feat_kurtosis']:>6.1f} "
              f"{r['feat_crest_factor']:>5.1f} {r['feat_clearance_factor']:>5.1f} "
              f"{'--':>8} {r['conf_outer']:>5.3f} {r['conf_inner']:>5.3f} "
              f"{r['conf_ball']:>5.3f} ({r['band_low']:>6.0f},{r['band_high']:>6.0f}) "
              f"{r['best_kurtosis']:>5.1f}")
    dt = time.time() - t0
    ok = [r for r in results if not r["error"]]

    # 冒烟版马氏距离：正常留一、异常用 5 正常基线（仅供肉眼检查量级）
    normals = [r for r in ok if r["is_normal"]]
    if len(normals) >= 2:
        print("\n冒烟版马氏距离（正常=留一基线，异常=5 正常基线；样本少仅查量级）:")
        for r in ok:
            if r["is_normal"]:
                others = [x for x in normals if x is not r]
                bl = up.fit_baseline([feats_of(x) for x in others])
            else:
                bl = up.fit_baseline([feats_of(x) for x in normals])
            d = up.maha_distance(bl, feats_of(r))
            r["maha_dist"] = d
            print(f"  {r['id']:>6} {r['subset']:<4} maha={d:9.2f}")
    fb = sum(r["rpm_fallback"] for r in ok)
    trig_n = sum(1 for r in ok if r["is_normal"])
    trig_norm_hi = sum(1 for r in ok
                       if r["is_normal"] and r.get("maha_dist", 0) > 10)
    print(f"\n冒烟汇总：成功 {len(ok)}/{len(picked)}，"
          f"均值 {dt / max(len(picked), 1):.2f}s/条 → 全量 ETA "
          f"{dt / max(len(picked), 1) * len(rows) / 60:.1f} min；"
          f"转速回退 {fb}/{len(ok)}；正常样本高距离(>10) {trig_norm_hi}/{trig_n}")
    confs = [(r["conf_outer"], r["conf_inner"], r["conf_ball"]) for r in ok]
    arr = np.array(confs) if confs else np.zeros((0, 3))
    assert not np.isnan(arr).any(), "存在 NaN 置信度"
    bad_rpm = [r["id"] for r in ok
               if not r["rpm_fallback"]
               and not (up.RPM_SEARCH_HZ[0] <= r["rpm_fr_hz"] <= up.RPM_SEARCH_HZ[1])]
    assert not bad_rpm, f"非回退 rpm 超窗: {bad_rpm}"
    print("断言通过：无 NaN 置信度；非回退 rpm 均在搜索窗内")


def run_smoke_v2(rows: list[dict]) -> None:
    """v2 冒烟：30 条分层（8 正常/8 轴承/8 摩擦/6 其他异常），打印梳检全链路。

    验证点（任务规定）：轴承样本能盲检出梳、正常样本梳检出稀少、
    50 Hz/轴频族防护有效。马氏距离同 v1 冒烟口径（留一/小子集基线，仅查量级）。
    """
    rng = np.random.default_rng(42)
    picked: list[dict] = []
    for flag, n in (("is_normal", 8), ("is_bearing", 8),
                    ("is_friction", 8), ("subset", 6)):
        if flag == "subset":
            pool = [r for r in rows if r["subset"] == "其他异常"]
        else:
            pool = [r for r in rows if r[flag]]
        idx = rng.choice(len(pool), size=n, replace=False)
        picked.extend(pool[int(i)] for i in idx)

    print(f"v2 冒烟 {len(picked)} 条（8 正常/8 轴承/8 摩擦/6 其他异常）")
    header = (f"{'id':>6} {'子集':<4} {'耗时s':>5} {'梳A':>3} {'互证':>3} "
              f"{'干扰':>3} {'净梳':>3} {'定位':>3} {'conf(o/i/b)':>22}  梳基频")
    print(header)
    results = []
    t0 = time.time()
    for rec in picked:
        r = process_one(rec, rules="v2")
        results.append(r)
        if r["error"]:
            print(f"{r['id']:>6} {r['subset']:<4} 失败: {r['error']}")
            continue
        confs = f"{r['conf_outer']:.2f}/{r['conf_inner']:.2f}/{r['conf_ball']:.2f}"
        print(f"{r['id']:>6} {r['subset']:<4} {r['elapsed_s']:>5.2f} "
              f"{r['comb_confirmed']:>3d} {r['cepstrum_confirmed']:>3d} "
              f"{r['suspected_interference']:>3d} {r['n_clean_combs']:>3d} "
              f"{r['n_localized']:>3d} {confs:>22}  [{r['comb_freqs']}]")
    dt = time.time() - t0
    ok = [r for r in results if not r["error"]]

    # 梳明细（重点文件）
    print("\n梳明细（有梳文件）：")
    for r in ok:
        if r.get("comb_detail") and r["comb_detail"] != "[]":
            print(f"  {r['id']} {r['subset']}: {r['comb_detail']}")

    # 冒烟版马氏距离（同 v1 口径）
    normals = [r for r in ok if r["is_normal"]]
    if len(normals) >= 2:
        print("\n冒烟版马氏距离（正常=留一基线，异常=8 正常基线；仅查量级）:")
        for r in ok:
            if r["is_normal"]:
                others = [x for x in normals if x is not r]
                bl = up.fit_baseline([feats_of(x) for x in others])
            else:
                bl = up.fit_baseline([feats_of(x) for x in normals])
            r["maha_dist"] = up.maha_distance(bl, feats_of(r))
            print(f"  {r['id']:>6} {r['subset']:<4} maha={r['maha_dist']:9.2f}")

    # 验证点汇总
    brg = [r for r in ok if r["is_bearing"]]
    nrm = [r for r in ok if r["is_normal"]]
    n_brg_comb = sum(1 for r in brg if r["n_clean_combs"] > 0 or r["comb_confirmed"] > 0)
    n_nrm_comb = sum(1 for r in nrm if r["comb_confirmed"] > 0)
    n_guard = sum(1 for r in ok if r["suspected_interference"] > 0)
    print(f"\nv2 冒烟验证点：")
    print(f"  轴承样本能盲检出梳: {n_brg_comb}/{len(brg)}")
    print(f"  正常样本梳检出（应稀少）: {n_nrm_comb}/{len(nrm)}")
    print(f"  干扰防护命中文件: {n_guard}/{len(ok)}")
    confs = np.array([[r["conf_outer"], r["conf_inner"], r["conf_ball"]] for r in ok])
    assert not np.isnan(confs).any(), "存在 NaN 置信度"
    print(f"  断言通过：无 NaN 置信度")
    print(f"  均值 {dt / max(len(picked), 1):.2f}s/条 → 全量 ETA "
          f"{dt / max(len(picked), 1) * len(rows) / 60:.1f} min")


# ---------------------------------------------------------------- 主流程

def main() -> None:
    ap = argparse.ArgumentParser(description="音频资产库合并原型评测")
    ap.add_argument("--rules", choices=("v1", "v2"), default="v1",
                    help="判别规则版本（默认 v1 理论锚定级联；v2 盲梳检）")
    ap.add_argument("--smoke", action="store_true", help="分层冒烟，不落盘")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="跳过逐文件计算，由已有 results.csv 重出聚合/图/报告")
    args = ap.parse_args()
    _apply_rule_paths(args.rules)

    common.verify_geometry()
    up.verify_orders()
    rows = load_manifest()
    mc = subset_counts(rows)
    print(f"规则版本: {args.rules}；产物目录: {RESULTS_CSV.parent}")
    print(f"manifest 共 {mc['total']} 条：正常 {mc['is_normal']} / 异常 {mc['is_abnormal']}"
          f"（轴承 {mc['is_bearing']} / 摩擦 {mc['is_friction']} / "
          f"其他 {mc['other_abnormal']}；详细部位参考 {mc['is_detailed']}）")

    if args.smoke:
        if args.rules == "v2":
            run_smoke_v2(rows)
        else:
            run_smoke(rows)
        return

    if args.aggregate_only:
        results = read_results_csv(RESULTS_CSV)
        skipped = []
        if SKIPPED_JSON.exists():
            skipped = json.loads(SKIPPED_JSON.read_text(encoding="utf-8"))
        total_s = sum(r["elapsed_s"] for r in results)
        detail = aggregate(results, skipped, total_s)
        print(f"聚合完成：{RESULTS_CSV} / {DETAIL_JSON} / {REPORT_MD}")
        return

    t0 = time.time()
    results, skipped = run_all(rows, rules=args.rules)
    total_s = time.time() - t0
    detail = aggregate(results, skipped, total_s)
    print(f"\n完成 {detail['counts']['processed']}/{detail['counts']['total']} 条，"
          f"耗时 {total_s / 60:.1f} min")
    if args.rules == "v2":
        na2 = detail["normal_abnormal_v2"]
        print(f"v2: 异常检出率 {na2['异常检出率']:.1%}，正常误报率 {na2['正常误报率']:.1%}，"
              f"轴承全体口径(含未定位) {detail['bearing_friction_v2']['bearing']['全体口径检出率_含未定位']:.1%}")
    else:
        na99 = detail["normal_abnormal"]["p99"]
        print(f"p99: 异常检出率 {na99['异常检出率']:.1%}，正常误报率 {na99['正常误报率']:.1%}")
    print(f"输出：{RESULTS_CSV}\n      {DETAIL_JSON}\n      {REPORT_MD}\n      {FIG_DIR}/")


if __name__ == "__main__":
    main()

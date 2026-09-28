# -*- coding: utf-8 -*-
"""轴承故障二分类 ML 实验（需求：reports/bearing_ml_experiment/实验需求与计划.md）。

normal(564)+bearing(171) 训练与随机分层 5 折评估；abnormal/others(526) 仅作
阈值标定（工作点"修正参数"）；统一包 v2 DSP（diagnose）作零训练基线对照。
特征：统一包 36 维免转速特征（use_physical=False，含 T6 盲梳结构化）。

用法（在 analysis/ 下）：
    python run_ml_bearing_experiment.py --smoke   # 冒烟：三类各 8 条，不落盘
    python run_ml_bearing_experiment.py           # 全量，产物落 reports/bearing_ml_experiment/
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

import common  # analysis/common.py：import 时强制 Agg；提供中文字体与 save_json
import matplotlib.pyplot as plt

from bearing_unified.diagnose import diagnose  # 注意：必须从模块导入函数（顶层名是模块）
from bearing_unified.ml.features import extract_features, feature_names
from bearing_unified.ml.train import ABNORMAL, NORMAL, make_estimator

FS = 16000.0
BEARING_NAME = "FIELD7"
RPM_WINDOW = (1100.0, 1200.0)
RANDOM_STATE = 42
N_SPLITS = 5
OTHERS_FPR_TARGET = 0.30  # 决策3：others 误报 ≲30% → 标定阈值 = others 分数 70 分位
RECALL_TARGETS = (0.90, 0.95, 0.98)  # 决策2：多档工作点
SUPERVISED_MODELS = ("LinearSVM", "RBF-SVM", "RandomForest", "HistGB")
NOVELTY_NAME = "novelty:elliptic_envelope"
ALL_MODELS = SUPERVISED_MODELS + (NOVELTY_NAME,)
POS_A = ("BPFO", "BPFI", "BSF")  # DSP 口径A：仅定位判决
POS_B = POS_A + ("bearing_unlocalized",)  # 口径B（主口径）：含轴承-未定位

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "datasets" / "manifest.csv"
OUT_DIR = ROOT / "analysis" / "reports" / "bearing_ml_experiment"
CACHE_NPZ = OUT_DIR / "features.npz"

CATS = ("normal", "abnormal/bearing", "abnormal/others")
CAT_LABEL = {"normal": "normal", "abnormal/bearing": "bearing", "abnormal/others": "others"}


# ---------------------------------------------------------------- 数据加载
def read_wav_any(path: str) -> tuple[np.ndarray, float, float, str]:
    """读 WAV → (满幅归一信号, fs, 削波比例, 备注)（与 run_eval_unified 同规则）。"""
    from scipy import signal as sps
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
        x = sps.resample_poly(x, frac.numerator, frac.denominator)
        notes.append(f"重采样{sr}→{int(FS)}Hz")
        sr = int(FS)
    return x, float(sr), clip_ratio, "; ".join(notes)


def load_manifest_rows() -> list[dict]:
    """读 datasets/manifest.csv（UTF-8），跳过 excluded 行。"""
    with open(MANIFEST, encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["category"] != "excluded"]
    if not rows:
        raise RuntimeError(f"manifest 无数据行: {MANIFEST}")
    return rows


def collect_files(smoke: bool) -> list[dict]:
    """汇总待处理文件清单；smoke 时三类各分层抽 8 条。"""
    rows = load_manifest_rows()
    files = [
        {
            "file_name": r["file_name"],
            "category": r["category"],
            "description": r["description"],
            "path": str(ROOT / "datasets" / r["target_relpath"]),
        }
        for r in rows
    ]
    if not smoke:
        return files
    rng = np.random.default_rng(RANDOM_STATE)
    picked: list[dict] = []
    for cat in CATS:
        sub = [f for f in files if f["category"] == cat]
        idx = rng.permutation(len(sub))[:8]
        picked += [sub[i] for i in idx]
    return picked


# ---------------------------------------------------------------- 特征提取
def extract_all_features(files: list[dict], names36: list[str],
                         use_cache: bool) -> tuple[np.ndarray, list[str], list[str], list[dict]]:
    """逐文件提取 36 维特征（带 features.npz 缓存），返回 X / files / cats / skipped。"""
    fname_list = [f["file_name"] for f in files]
    if use_cache and CACHE_NPZ.exists():
        z = np.load(CACHE_NPZ, allow_pickle=False)
        if list(z["files"]) == fname_list:
            print(f"[特征] 命中缓存 {CACHE_NPZ.name}（{z['X'].shape[0]} 条），跳过重算")
            return z["X"], fname_list, list(z["cats"]), []
        print("[特征] 缓存与当前清单不一致，重新提取")

    X = np.zeros((len(files), len(names36)), dtype=np.float64)
    cats: list[str] = []
    skipped: list[dict] = []
    t0 = time.perf_counter()
    for i, f in enumerate(files):
        t1 = time.perf_counter()
        try:
            x, fs, _clip, note = read_wav_any(f["path"])
            fv = extract_features(x, fs, use_physical=False, bearing=BEARING_NAME)
            fv = np.asarray(fv, dtype=np.float64)
            fv = np.where(np.isfinite(fv), fv, 0.0)  # 包内同款：非有限值置 0
            if fv.shape != (len(names36),):
                raise ValueError(f"特征维度异常 {fv.shape}")
            X[i] = fv
            cats.append(f["category"])
            f["feat_time_s"] = time.perf_counter() - t1
            if note:
                f["note"] = note
        except Exception as exc:  # 单文件失败隔离，不中断整批
            skipped.append({"file": f["file_name"], "category": f["category"],
                            "stage": "特征提取", "error": repr(exc)})
        if (i + 1) % 50 == 0 or i + 1 == len(files):
            elapsed = time.perf_counter() - t0
            eta = elapsed / (i + 1) * (len(files) - i - 1)
            print(f"[特征] {i + 1}/{len(files)}  已用 {elapsed:.0f}s  ETA {eta:.0f}s", flush=True)
    # 剔除失败行，保持 X 与文件清单对齐
    if skipped:
        bad = {s["file"] for s in skipped}
        keep = [i for i, fn in enumerate(fname_list) if fn not in bad]
        X, cats = X[keep], [cats[i] for i in keep]
        files = [f for f in files if f["file_name"] not in bad]
        print(f"[特征] 失败 {len(bad)} 条，有效 {X.shape[0]} 条")
    if use_cache and X.shape[0]:
        CACHE_NPZ.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CACHE_NPZ, X=X, files=np.array(fname_list[:0] + [f["file_name"] for f in files]),
                            cats=np.array(cats))
        print(f"[特征] 缓存已写 {CACHE_NPZ}")
    return X, [f["file_name"] for f in files], cats, skipped


# ---------------------------------------------------------------- CV 与工作点
def positive_score(est, X: np.ndarray, pos_label) -> np.ndarray:
    """取 predict_proba 中正类列（列下标必须经 classes_ 查，novelty 顺序不保证）。"""
    proba = np.asarray(est.predict_proba(X))
    if proba.ndim == 1:
        return proba
    classes = [str(c) for c in est.classes_]
    return proba[:, classes.index(str(pos_label))]


def cv_oof_scores(model_name: str, X: np.ndarray, y01: np.ndarray, y_str: np.ndarray,
                  names36: list[str]) -> np.ndarray:
    """随机分层 5 折 OOF 正类（bearing/abnormal）分数。每折 make_estimator 重建。"""
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.full(len(y01), np.nan)
    for tr, te in skf.split(X, y01):
        est = make_estimator(model_name, random_state=RANDOM_STATE, feature_names=names36)
        if model_name.startswith("novelty:"):
            est.fit(X[tr], y_str[tr])  # NoveltyDetector 内部只取 y==normal 样本
            pos = ABNORMAL
        else:
            est.fit(X[tr], y01[tr])
            pos = 1
        oof[te] = positive_score(est, X[te], pos)
    return oof


def point_metrics(thr: float, oof: np.ndarray, y01: np.ndarray,
                  others_scores: np.ndarray) -> dict:
    """一个阈值下的三点指标：bearing recall / normal 误报（均 OOF 口径）/ others 误报（标定口径）。"""
    pos = oof >= thr
    return {
        "threshold": float(thr),
        "bearing_recall_oof": float(pos[y01 == 1].mean()),
        "normal_fpr_oof": float(pos[y01 == 0].mean()),
        "others_fpr_cal": float((others_scores >= thr).mean()),
    }


def fit_full(model_name: str, X: np.ndarray, y01: np.ndarray, y_str: np.ndarray,
             names36: list[str]):
    est = make_estimator(model_name, random_state=RANDOM_STATE, feature_names=names36)
    if model_name.startswith("novelty:"):
        est.fit(X, y_str)
    else:
        est.fit(X, y01)
    return est


# ---------------------------------------------------------------- DSP 基线
def run_dsp(files: list[dict], write: bool) -> list[dict]:
    """逐文件 v2 诊断（不传 rpm/calibrator），记录判决与耗时；checkpoint 每 200 条。"""
    results: list[dict] = []
    skipped: list[dict] = []
    t0 = time.perf_counter()
    for i, f in enumerate(files):
        t1 = time.perf_counter()
        try:
            x, fs, _clip, _note = read_wav_any(f["path"])
            r = diagnose(x, fs, bearing=BEARING_NAME, rpm_window=RPM_WINDOW)
            f["dsp_time_s"] = time.perf_counter() - t1
            results.append({
                "file_name": f["file_name"],
                "verdict": r.verdict,
                "n_clean_combs": int(r.n_clean_combs),
                "low_conf": bool(r.low_conf),
                "rpm_fallback": bool(r.rpm_fallback),
                "rpm_est": float(r.rpm_est),
                "pos_A": r.verdict in POS_A,
                "pos_B": r.verdict in POS_B,
            })
        except Exception as exc:
            skipped.append({"file": f["file_name"], "category": f["category"],
                            "stage": "DSP", "error": repr(exc)})
        if (i + 1) % 50 == 0 or i + 1 == len(files):
            elapsed = time.perf_counter() - t0
            eta = elapsed / (i + 1) * (len(files) - i - 1)
            print(f"[DSP] {i + 1}/{len(files)}  已用 {elapsed:.0f}s  ETA {eta:.0f}s", flush=True)
        if write and (i + 1) % 200 == 0:
            common.save_json({"done": i + 1, "results": results, "skipped": skipped},
                             OUT_DIR / "checkpoint.json")
    if skipped:
        print(f"[DSP] 失败 {len(skipped)} 条")
    return results + [{"file_name": s["file"], "verdict": "ERROR", "n_clean_combs": -1,
                       "low_conf": False, "rpm_fallback": False, "rpm_est": 0.0,
                       "pos_A": False, "pos_B": False} for s in skipped]


def dsp_subset_metrics(dsp: list[dict], files: list[dict]) -> dict:
    """三子集 × 两口径的检出/误报。"""
    by_name = {d["file_name"]: d for d in dsp}
    out: dict = {}
    for cat in CATS:
        sub = [f["file_name"] for f in files if f["category"] == cat]
        a = [by_name[n]["pos_A"] for n in sub if n in by_name]
        b = [by_name[n]["pos_B"] for n in sub if n in by_name]
        key = CAT_LABEL[cat]
        out[key] = {
            "n": len(sub),
            "posA_rate": float(np.mean(a)) if a else None,
            "posB_rate": float(np.mean(b)) if b else None,
        }
    return out


# ---------------------------------------------------------------- 图
def make_figures(oof_by_model: dict, y01: np.ndarray, best_name: str,
                 oof_best: np.ndarray, others_scores: np.ndarray, thr_cal: float,
                 dsp_metrics: dict, ml_point: dict, importances: dict,
                 names36: list[str], write: bool) -> None:
    common.setup_matplotlib()
    fig_dir = OUT_DIR / "figures"
    if write:
        fig_dir.mkdir(parents=True, exist_ok=True)

    # ① P-R 曲线（5 模型 OOF）
    from sklearn.metrics import average_precision_score, precision_recall_curve
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for name, oof in oof_by_model.items():
        p, r, _ = precision_recall_curve(y01, oof)
        ap = average_precision_score(y01, oof)
        label = name if not name.startswith("novelty:") else "novelty(马氏)"
        ax.plot(r, p, label=f"{label} (AP={ap:.3f})")
    ax.set_xlabel("召回率 (bearing)")
    ax.set_ylabel("精确率")
    ax.set_title(f"各模型 OOF P-R 曲线（normal+bearing，{N_SPLITS} 折）")
    ax.legend(loc="lower left", fontsize=9)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    if write:
        fig.savefig(fig_dir / "fig1_pr_curves.png")
    plt.close(fig)

    # ② 阈值扫描：bearing recall 与 others/normal 误报率
    lo = float(min(oof_best.min(), others_scores.min()))
    hi = float(max(oof_best.max(), others_scores.max()))
    grid = np.unique(np.linspace(lo, hi, 201))
    rec = [ (oof_best >= t)[y01 == 1].mean() for t in grid ]
    fpr_o = [ (others_scores >= t).mean() for t in grid ]
    fpr_n = [ (oof_best >= t)[y01 == 0].mean() for t in grid ]
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.plot(grid, rec, label="bearing 召回率 (OOF)")
    ax.plot(grid, fpr_n, label="normal 误报率 (OOF)")
    ax.plot(grid, fpr_o, label="others 误报率（标定口径）")
    ax.axvline(thr_cal, color="k", ls="--", lw=1, label=f"标定阈值 {thr_cal:.3f}")
    ax.set_xlabel("判定为 bearing 的分数阈值")
    ax.set_ylabel("比例")
    ax.set_title(f"阈值扫描（最优模型：{best_name}）")
    ax.legend(fontsize=9)
    fig.tight_layout()
    if write:
        fig.savefig(fig_dir / "fig2_threshold_sweep.png")
    plt.close(fig)

    # ③ 最优模型@标定阈值 三类混淆矩阵（normal/bearing 用 OOF，others 用全量 refit 分数）
    cm = np.zeros((3, 2), dtype=int)
    pred_nb = (oof_best >= thr_cal).astype(int)
    cm[0, 0] = int((pred_nb[y01 == 0] == 0).sum()); cm[0, 1] = int((pred_nb[y01 == 0] == 1).sum())
    cm[1, 0] = int((pred_nb[y01 == 1] == 0).sum()); cm[1, 1] = int((pred_nb[y01 == 1] == 1).sum())
    po = (others_scores >= thr_cal).astype(int)
    cm[2, 0] = int((po == 0).sum()); cm[2, 1] = int((po == 1).sum())
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    row_sum = cm.sum(axis=1, keepdims=True).clip(min=1)
    im = ax.imshow(cm / row_sum, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n({cm[i, j] / row_sum[i, 0]:.1%})",
                    ha="center", va="center", fontsize=10)
    ax.set_xticks([0, 1], ["判正常", "判轴承"])
    ax.set_yticks([0, 1, 2], ["正常", "轴承", "其他异常\n(标定口径)"])
    ax.set_title(f"最优模型 {best_name} @ 阈值 {thr_cal:.3f}")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    if write:
        fig.savefig(fig_dir / "fig3_confusion_at_cal.png")
    plt.close(fig)

    # ④ DSP vs ML 三子集对比柱状图
    groups = [("normal", "正常误报率"), ("bearing", "轴承检出率"), ("others", "其他异常误报率(标定)")]
    ml_vals = [ml_point["normal_fpr_oof"], ml_point["bearing_recall_oof"], ml_point["others_fpr_cal"]]
    dspa = [dsp_metrics[k]["posA_rate"] for k, _ in groups]
    dspb = [dsp_metrics[k]["posB_rate"] for k, _ in groups]
    xpos = np.arange(3)
    w = 0.26
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.bar(xpos - w, ml_vals, w, label=f"ML {best_name}@{thr_cal:.3f}")
    ax.bar(xpos, dspa, w, label="DSP 口径A(仅定位)")
    ax.bar(xpos + w, dspb, w, label="DSP 口径B(含未定位)")
    ax.set_xticks(xpos, [cn for _, cn in groups])
    ax.set_ylabel("比例")
    ax.set_title("ML 工作点 vs v2 DSP 基线")
    for x, v in zip(xpos - w, ml_vals):
        ax.text(x, v + 0.01, f"{v:.1%}", ha="center", fontsize=8)
    for x, v in zip(xpos, dspa):
        ax.text(x, v + 0.01, f"{v:.1%}", ha="center", fontsize=8)
    for x, v in zip(xpos + w, dspb):
        ax.text(x, v + 0.01, f"{v:.1%}", ha="center", fontsize=8)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.1)
    fig.tight_layout()
    if write:
        fig.savefig(fig_dir / "fig4_dsp_vs_ml.png")
    plt.close(fig)

    # ⑤ 特征重要性 top-15（RF / HistGB 双面板）
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    for ax, (name, imp) in zip(axes, importances.items()):
        order = np.argsort(imp)[::-1][:15]
        ax.barh(range(len(order)), imp[order][::-1])
        ax.set_yticks(range(len(order)), [names36[i] for i in order][::-1], fontsize=8)
        ax.set_title(name)
        ax.set_xlabel("feature importance")
    fig.suptitle("特征重要性 Top-15（全量 refit）")
    fig.tight_layout()
    if write:
        fig.savefig(fig_dir / "fig5_feature_importance.png")
    plt.close(fig)
    if not write:
        return
    print(f"[图] 已输出 {len(list(fig_dir.glob('*.png')))} 张 → {fig_dir}")


# ---------------------------------------------------------------- 报告
def write_report(meta: dict, model_rows: list[dict], best_name: str, thr_cal: float,
                 cal_point: dict, op_table: list[dict], dsp_metrics: dict,
                 importances: dict, names36: list[str], skipped: list[dict],
                 timing: dict) -> str:
    L: list[str] = []
    L.append("# 轴承故障二分类 ML 实验报告\n")
    L.append(f"生成时间：{meta['generated_at']}\n")
    L.append("## 口径声明\n")
    L.append("- 数据：`datasets/` normal 564 / abnormal-bearing 171 / abnormal-others 526（标签为工厂/标注员声称值，非拆解确认）。")
    L.append(f"- 训练与评估：normal+bearing 共 {meta['n_train']} 条，随机分层 {N_SPLITS} 折（random_state={RANDOM_STATE}）。"
             "**限制**：record_15/17 两个录音会话高度相关，随机折可能高估泛化，指标应作乐观参考。")
    L.append("- others 526 条**全部用于阈值标定**（标定阈值=others 分数 70 分位）；"
             "**others 误报率为标定集口径，不是独立泛化估计**。normal 误报为 CV-OOF 口径，两者口径不同，须分开解读。")
    L.append(f"- 特征：统一包 36 维免转速特征（`use_physical=False`，含 T6 盲梳结构化 8 维，FIELD7 几何 + 标称转速窗 {RPM_WINDOW} rpm）。")
    L.append("- DSP 基线：统一包 v2 `diagnose`（零训练），不传实测转速与马氏标定；口径A=仅定位判决，口径B=含轴承-未定位（主口径）。")
    L.append("- 诚实纪律：以下指标如实报告，不达标不粉饰；结论含归因。\n")
    L.append("## 模型对比（5 折 OOF）\n")
    L.append("| 模型 | AUC-PR | ROC-AUC | 备注 |")
    L.append("|---|---|---|---|")
    for r in model_rows:
        note = "最优" if r["model"] == best_name else ""
        L.append(f"| {r['model']} | {r['auc_pr']:.3f} | {r['roc_auc']:.3f} | {note} |")
    L.append(f"\n最优模型：**{best_name}**（按 AUC-PR）。\n")
    L.append("## 工作点\n")
    L.append("| 工作点 | 阈值 | bearing 召回(OOF) | normal 误报(OOF) | others 误报(标定口径) |")
    L.append("|---|---|---|---|---|")
    L.append(f"| **标定点(others 70 分位)** | {cal_point['threshold']:.4f} | "
             f"{cal_point['bearing_recall_oof']:.1%} | {cal_point['normal_fpr_oof']:.1%} | "
             f"{cal_point['others_fpr_cal']:.1%} |")
    for row in op_table:
        L.append(f"| recall@{row['target']:.2f} | {row['threshold']:.4f} | "
                 f"{row['bearing_recall_oof']:.1%} | {row['normal_fpr_oof']:.1%} | "
                 f"{row['others_fpr_cal']:.1%} |")
    L.append("\n## DSP 基线对照（三子集 × 两口径）\n")
    L.append("| 子集 | n | DSP口径A | DSP口径B(主) | ML@标定点 |")
    L.append("|---|---|---|---|---|")
    ml_by_cat = {"normal": cal_point["normal_fpr_oof"], "bearing": cal_point["bearing_recall_oof"],
                 "others": cal_point["others_fpr_cal"]}
    cat_cn = {"normal": "正常", "bearing": "轴承", "others": "其他异常(标定口径)"}
    for k in ("normal", "bearing", "others"):
        d = dsp_metrics[k]
        L.append(f"| {cat_cn[k]} | {d['n']} | {d['posA_rate']:.1%} | {d['posB_rate']:.1%} | "
                 f"{ml_by_cat[k]:.1%} |")
    L.append("\n## 特征重要性 Top-15\n")
    for name, imp in importances.items():
        order = np.argsort(imp)[::-1][:15]
        L.append(f"**{name}**：" + "、".join(f"`{names36[i]}`({imp[i]:.3f})" for i in order) + "\n")
    if skipped:
        L.append(f"## 失败记录（{len(skipped)} 条）\n")
        for s in skipped:
            L.append(f"- `{s['file']}` [{s['stage']}] {s['error']}")
        L.append("")
    L.append("## 结论\n")
    L.append(meta["conclusion"])
    L.append(f"\n## 耗时\n特征 {timing['feat_s']:.0f}s，DSP {timing['dsp_s']:.0f}s，"
             f"训练评估 {timing['ml_s']:.0f}s。\n")
    return "\n".join(L)


# ---------------------------------------------------------------- 主流程
def main() -> None:
    ap = argparse.ArgumentParser(description="轴承故障二分类 ML 实验（others 仅标定）")
    ap.add_argument("--smoke", action="store_true", help="三类各抽 8 条冒烟验证，不落盘")
    args = ap.parse_args()
    write = not args.smoke
    if write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = collect_files(args.smoke)
    n_by_cat = {c: sum(1 for f in files if f["category"] == c) for c in CATS}
    print(f"[数据] {n_by_cat}（smoke={args.smoke}）")
    names36 = feature_names(use_physical=False)

    # 1) 特征
    t_feat = time.perf_counter()
    X, fnames, cats, skipped = extract_all_features(files, names36, use_cache=write)
    files = [f for f in files if f["file_name"] in set(fnames)]
    feat_s = time.perf_counter() - t_feat

    # 2) 训练/标定/评估矩阵
    mask_nb = np.array([c in ("normal", "abnormal/bearing") for c in cats])
    X_nb = X[mask_nb]
    y01 = np.array([1 if c == "abnormal/bearing" else 0 for c in cats])[mask_nb]
    y_str = np.array([NORMAL if v == 0 else ABNORMAL for v in y01])
    X_others = X[~mask_nb]
    print(f"[ML] 训练评估 {X_nb.shape}，标定 {X_others.shape}")

    # 3) 5 模型 OOF CV
    t_ml = time.perf_counter()
    oof_by_model = {m: cv_oof_scores(m, X_nb, y01, y_str, names36) for m in ALL_MODELS}

    from sklearn.metrics import average_precision_score, roc_auc_score
    model_rows = []
    for m, oof in oof_by_model.items():
        model_rows.append({
            "model": m,
            "auc_pr": float(average_precision_score(y01, oof)),
            "roc_auc": float(roc_auc_score(y01, oof)),
        })
    model_rows.sort(key=lambda r: -r["auc_pr"])
    best_name = model_rows[0]["model"]
    oof_best = oof_by_model[best_name]
    print("[ML] 模型排序：" + "  ".join(f"{r['model']}={r['auc_pr']:.3f}" for r in model_rows))

    # 4) 最优模型全量 refit → others 标定
    best_est = fit_full(best_name, X_nb, y01, y_str, names36)
    pos_label = ABNORMAL if best_name.startswith("novelty:") else 1
    others_scores = positive_score(best_est, X_others, pos_label)
    thr_cal = float(np.quantile(others_scores, 1 - OTHERS_FPR_TARGET))
    cal_point = point_metrics(thr_cal, oof_best, y01, others_scores)

    # 5) 多档工作点（recall 反查阈值，OOF 分布）
    op_table = []
    for target in RECALL_TARGETS:
        thr_r = float(np.quantile(oof_best[y01 == 1], 1 - target))
        row = {"target": target, **point_metrics(thr_r, oof_best, y01, others_scores)}
        op_table.append(row)

    # 6) 特征重要性（RF 原生；HistGB 无 feature_importances_，用 permutation importance）
    importances: dict[str, np.ndarray] = {}
    for name in ("RandomForest", "HistGB"):
        est = fit_full(name, X_nb, y01, y_str, names36)
        if hasattr(est, "feature_importances_"):
            importances[name] = np.asarray(est.feature_importances_)
        else:
            from sklearn.inspection import permutation_importance

            pi = permutation_importance(est, X_nb, y01, n_repeats=5,
                                        random_state=RANDOM_STATE, n_jobs=-1)
            importances[name] = np.asarray(pi.importances_mean)
    ml_s = time.perf_counter() - t_ml

    # 7) DSP 基线（带结果缓存：文件清单一致即复用，重跑免重算）
    t_dsp = time.perf_counter()
    dsp_cache = OUT_DIR / "dsp_results.json"
    dsp = None
    if write and dsp_cache.exists():
        z = json.loads(dsp_cache.read_text(encoding="utf-8"))
        if [d["file_name"] for d in z] == [f["file_name"] for f in files]:
            dsp = z
            print(f"[DSP] 命中缓存（{len(z)} 条），跳过重算")
    if dsp is None:
        dsp = run_dsp(files, write)
        if write:
            common.save_json(dsp, dsp_cache)
    dsp_metrics = dsp_subset_metrics(dsp, files)
    dsp_s = time.perf_counter() - t_dsp

    # 8) 结果打印
    print(f"[标定] 阈值={thr_cal:.4f}  recall={cal_point['bearing_recall_oof']:.1%}  "
          f"normal误报={cal_point['normal_fpr_oof']:.1%}  others误报(标定)={cal_point['others_fpr_cal']:.1%}")
    print(f"[DSP] 轴承检出 B口径={dsp_metrics['bearing']['posB_rate']:.1%}  "
          f"正常误报 B口径={dsp_metrics['normal']['posB_rate']:.1%}")

    if not write:
        print("[冒烟] 全流程跑通，未落盘。")
        return

    # 9) DSP vs ML 结论素材与产物落盘
    gain = cal_point["bearing_recall_oof"] - dsp_metrics["bearing"]["posB_rate"]
    conclusion = (
        f"按 AUC-PR 最优模型为 {best_name}。在标定阈值 {thr_cal:.4f} 下，bearing 召回 "
        f"{cal_point['bearing_recall_oof']:.1%}（OOF 口径），较 v2 DSP 主口径（含未定位）检出 "
        f"{dsp_metrics['bearing']['posB_rate']:.1%} {'高' if gain >= 0 else '低'} {abs(gain):.1%}；"
        f"normal 误报 {cal_point['normal_fpr_oof']:.1%}（DSP {dsp_metrics['normal']['posB_rate']:.1%}）。"
        f"others 误报 {cal_point['others_fpr_cal']:.1%} 为标定集口径（按 ≲30% 约束设计）。"
        "注意：随机分层 5 折受会话相关性影响偏乐观，标签为声称值，结论需现场数据复验。"
    )
    meta = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_train": int(len(y01)),
        "conclusion": conclusion,
    }
    timing = {"feat_s": feat_s, "dsp_s": dsp_s, "ml_s": ml_s}
    report_md = write_report(meta, model_rows, best_name, thr_cal, cal_point, op_table,
                             dsp_metrics, importances, names36, skipped, timing)

    # results.csv（逐文件）
    by_name_dsp = {d["file_name"]: d for d in dsp}
    short_of_model = {"LinearSVM": "linear_svm", "RBF-SVM": "rbf_svm",
                      "RandomForest": "random_forest", "HistGB": "histgb",
                      NOVELTY_NAME: "novelty"}
    cat_of = {f["file_name"]: f["category"] for f in files}
    desc_of = {f["file_name"]: f["description"] for f in files}
    with open(OUT_DIR / "results.csv", "w", encoding="utf-8", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["file_name", "category", "description"] +
                   [f"prob_{short_of_model[m]}" for m in ALL_MODELS] +
                   ["ml_score_best", "ml_pred_at_cal", "dsp_verdict", "dsp_pos_A",
                    "dsp_pos_B", "dsp_n_clean_combs", "dsp_low_conf",
                    "feat_time_s", "dsp_time_s", "note"])
        others_score_of = {ff["file_name"]: float(s)
                           for ff, s in zip((f for f in files if f["category"] == "abnormal/others"),
                                            others_scores)}
        # OOF 数组行号与 nb 子集文件的对齐映射（不能用全量文件下标直接索引 735 行数组）
        nb_row_of = {files[i]["file_name"]: int(r)
                     for r, i in enumerate(np.where(mask_nb)[0])}
        for f in files:
            fn = f["file_name"]
            in_nb = cat_of[fn] in ("normal", "abnormal/bearing")
            score = float(oof_best[nb_row_of[fn]]) if in_nb else others_score_of[fn]
            w.writerow([
                fn, cat_of[fn], desc_of[fn],
                *[f"{oof_by_model[m][nb_row_of[fn]]:.6f}" if in_nb else "" for m in ALL_MODELS],
                f"{score:.6f}",
                int(score >= thr_cal),
                by_name_dsp.get(fn, {}).get("verdict", ""),
                int(bool(by_name_dsp.get(fn, {}).get("pos_A", False))),
                int(bool(by_name_dsp.get(fn, {}).get("pos_B", False))),
                by_name_dsp.get(fn, {}).get("n_clean_combs", ""),
                int(by_name_dsp.get(fn, {}).get("low_conf", False)),
                f.get("feat_time_s", ""), f.get("dsp_time_s", ""),
                f.get("note", ""),
            ])

    # eval_detail.json
    detail = {
        "口径声明": {
            "数据": "datasets/（manifest UTF-8；标签为工厂/标注员声称值，非拆解确认）",
            "训练评估": f"normal+bearing 共 {len(y01)} 条，随机分层 {N_SPLITS} 折 random_state={RANDOM_STATE}；"
                       "限制：record_15/17 会话相关性使随机折偏乐观",
            "others用法": "526 条全部用于阈值标定（70 分位）；其误报率为标定集口径，非独立泛化估计",
            "特征": "统一包 36 维免转速（use_physical=False，T6 盲梳结构化含 FIELD7 几何）",
            "DSP基线": f"v2 diagnose，bearing={BEARING_NAME}，rpm_window={RPM_WINDOW}，无实测转速/无马氏标定",
            "决策": "8 项用户确认决策见 实验需求与计划.md 1.3",
        },
        "counts": {CAT_LABEL[c]: n_by_cat[c] for c in CATS},
        "models": model_rows,
        "best_model": best_name,
        "calibrated_threshold": thr_cal,
        "operating_point_calibrated": cal_point,
        "operating_points_recall": op_table,
        "dsp_baseline": dsp_metrics,
        "dsp_rpm_fallback_rate": float(np.mean([d["rpm_fallback"] for d in dsp])),
        "feature_importance": {k: dict(zip(names36, map(float, v))) for k, v in importances.items()},
        "skipped": skipped,
        "timing_s": timing,
        "conclusion": conclusion,
    }
    common.save_json(detail, OUT_DIR / "eval_detail.json")
    (OUT_DIR / "实验报告.md").write_text(report_md, encoding="utf-8")

    make_figures(oof_by_model, y01, best_name, oof_best, others_scores, thr_cal,
                 dsp_metrics, cal_point, importances, names36, write=True)
    print(f"[完成] 产物 → {OUT_DIR}")


if __name__ == "__main__":
    main()

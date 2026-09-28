# -*- coding: utf-8 -*-
"""评测结果统计：分层准确率、混淆矩阵、两方案一致性、错误结构归因。

输出：reports/stats_summary.json、figures/confusion_*.png、figures/agreement.png
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict

import numpy as np

import common

common.setup_matplotlib()
import matplotlib.pyplot as plt

CLASS_ORDER = ["正常", "外圈", "内圈", "滚珠"]
ALL_LABELS = CLASS_ORDER + [common.OTHER_CN]


def load_results() -> list[dict]:
    with open(common.REPORTS_DIR / "results.csv", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def confusion(sub: list[dict]) -> np.ndarray:
    cm = np.zeros((len(CLASS_ORDER), len(ALL_LABELS)), dtype=int)
    for r in sub:
        i = CLASS_ORDER.index(r["class"])
        j = ALL_LABELS.index(r["verdict_cn"]) if r["verdict_cn"] in ALL_LABELS else None
        if j is None:  # 理论不会发生（映射已收敛到五值）
            continue
        cm[i, j] += 1
    return cm


def plot_confusion(cm: np.ndarray, title: str, path, acc: float) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    im = ax.imshow(cm, cmap="Blues", vmin=0)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j]:
                color = "white" if cm[i, j] > cm.max() * 0.6 else "black"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)
    ax.set_xticks(range(len(ALL_LABELS)), ALL_LABELS, rotation=20)
    ax.set_yticks(range(len(CLASS_ORDER)), CLASS_ORDER)
    ax.set_xlabel("判决")
    ax.set_ylabel("真值（工厂标签）")
    ax.set_title(f"{title}  acc={acc:.1%}")
    fig.colorbar(im, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def cohens_kappa(a: list[str], b: list[str]) -> float:
    labels = sorted(set(a) | set(b))
    idx = {l: i for i, l in enumerate(labels)}
    n = len(a)
    obs = sum(1 for x, y in zip(a, b) if x == y) / n
    pa = sum(a.count(l) for l in labels if False)  # placeholder
    from collections import Counter as C
    ca, cb = C(a), C(b)
    exp = sum(ca[l] / n * cb[l] / n for l in labels)
    return (obs - exp) / (1 - exp) if exp < 1 else 1.0


def main() -> None:
    rows = load_results()
    stats: dict = {}

    for round_name in ("A_main_1150", "B_fr_est"):
        rd = {}
        for scheme in ("glm", "kimi"):
            sub = [r for r in rows if r["round"] == round_name and r["scheme"] == scheme]
            cm = confusion(sub)
            acc = float(np.trace(cm[:, :4]) / cm.sum())
            # 分层
            by_class = {
                c: float(cm[i, i] / cm[i].sum()) for i, c in enumerate(CLASS_ORDER)
            }
            by_session = {}
            for sess in ("morning", "afternoon"):
                s = [r for r in sub if r["session"] == sess]
                by_session[sess] = {
                    "n": len(s),
                    "acc": sum(int(r["correct"]) for r in s) / len(s),
                }
            by_batch = {}
            for b in ("9-16", "9-17", "9-18"):
                s = [r for r in sub if r["batch"] == b]
                by_batch[b] = {
                    "n": len(s),
                    "acc": sum(int(r["correct"]) for r in s) / len(s),
                }
            # 错误结构
            n_normal_verdict = sum(1 for r in sub if r["verdict_cn"] == "正常")
            n_other = sum(1 for r in sub if r["verdict_cn"] == common.OTHER_CN)
            faults = [r for r in sub if r["class"] != "正常"]
            miss = sum(1 for r in faults if r["verdict_cn"] == "正常")
            healthy = [r for r in sub if r["class"] == "正常"]
            fa = sum(1 for r in healthy if r["verdict_cn"] != "正常")
            wrong_type = sum(
                1 for r in sub
                if r["class"] != "正常" and r["verdict_cn"] != "正常"
                and r["verdict_cn"] != r["class"]
            )
            rd[scheme] = {
                "accuracy": acc,
                "confusion": cm.tolist(),
                "confusion_labels": {"rows": CLASS_ORDER, "cols": ALL_LABELS},
                "by_class": by_class,
                "by_session": by_session,
                "by_batch": by_batch,
                "n_verdict_normal": n_normal_verdict,
                "n_verdict_other_FTF_cage": n_other,
                "fault_missed_as_normal": miss,
                "healthy_false_alarmed": fa,
                "wrong_fault_type": wrong_type,
            }
            plot_confusion(
                cm,
                f"{scheme} 轮次{'A(1150rpm)' if round_name.startswith('A') else 'B(fr_est)'}",
                common.FIG_DIR / f"confusion_{scheme}_{'A' if round_name.startswith('A') else 'B'}.png",
                acc,
            )
        # 两方案一致性（同轮次）
        ga = {r["file_id"]: r["verdict_cn"] for r in rows
              if r["round"] == round_name and r["scheme"] == "glm"}
        ka = {r["file_id"]: r["verdict_cn"] for r in rows
              if r["round"] == round_name and r["scheme"] == "kimi"}
        agree_n = sum(1 for fid in ga if ga[fid] == ka[fid])
        rd["agreement"] = {
            "n_agree": agree_n,
            "n_total": len(ga),
            "rate": agree_n / len(ga),
            "kappa": cohens_kappa(list(ga.values()), list(ka.values())),
            "confusion_glm_vs_kimi": None,  # 用矩阵图呈现
        }
        stats[round_name] = rd

    # 一致性矩阵图（轮次 A：glm × kimi）
    ga = {r["file_id"]: r["verdict_cn"] for r in rows
          if r["round"] == "A_main_1150" and r["scheme"] == "glm"}
    ka = {r["file_id"]: r["verdict_cn"] for r in rows
          if r["round"] == "A_main_1150" and r["scheme"] == "kimi"}
    labels = ALL_LABELS
    cm2 = np.zeros((len(labels), len(labels)), dtype=int)
    for fid in ga:
        cm2[labels.index(ga[fid]), labels.index(ka[fid])] += 1
    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    im = ax.imshow(cm2, cmap="Purples", vmin=0)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm2[i, j]:
                color = "white" if cm2[i, j] > cm2.max() * 0.6 else "black"
                ax.text(j, i, str(cm2[i, j]), ha="center", va="center", color=color)
    ax.set_xticks(range(len(labels)), labels, rotation=20)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("kimi 判决")
    ax.set_ylabel("glm 判决")
    agree = np.trace(cm2) / cm2.sum()
    ax.set_title(f"轮次 A 两方案判决交叉（一致率 {agree:.1%}）")
    fig.colorbar(im, shrink=0.8)
    fig.tight_layout()
    fig.savefig(common.FIG_DIR / "agreement_A.png")
    plt.close(fig)

    # kimi 得分尺度失配证据（正常类 top 分 vs 阈值 15）
    k_healthy = [float(r["confidence"]) for r in rows
                 if r["round"] == "A_main_1150" and r["scheme"] == "kimi" and r["class"] == "正常"]
    stats["kimi_score_scale"] = {
        "healthy_top_scores_A": {
            "min": min(k_healthy), "median": float(np.median(k_healthy)), "max": max(k_healthy),
        },
        "health_threshold_calibrated": 15.0,
        "sim_calibrated_healthy_range": [5.97, 12.4],
    }

    # glm 置信度：对/错分布
    for round_name in ("A_main_1150", "B_fr_est"):
        g = [r for r in rows if r["round"] == round_name and r["scheme"] == "glm"]
        conf_ok = [float(r["confidence"]) for r in g if r["correct"] == "1"]
        conf_err = [float(r["confidence"]) for r in g if r["correct"] == "0"]
        stats[round_name]["glm_confidence"] = {
            "correct_median": float(np.median(conf_ok)) if conf_ok else None,
            "error_median": float(np.median(conf_err)) if conf_err else None,
        }

    # 两方案一致且与标签不符 / 一致且正确（轮次 A）
    cls = {r["file_id"]: r["class"] for r in rows if r["round"] == "A_main_1150" and r["scheme"] == "glm"}
    agree_wrong, agree_right = [], []
    for fid in ga:
        if ga[fid] == ka[fid]:
            (agree_right if ga[fid] == cls[fid] else agree_wrong).append(
                {"file_id": fid, "class": cls[fid], "verdict": ga[fid]}
            )
    stats["A_main_1150"]["agree_both_wrong_vs_label"] = agree_wrong
    stats["A_main_1150"]["agree_both_correct"] = len(agree_right)

    common.save_json(stats, common.REPORTS_DIR / "stats_summary.json")

    # 控制台摘要
    for round_name, rd in stats.items():
        if not round_name.startswith(("A_", "B_")):
            continue
        print(f"\n=== {round_name} ===")
        for scheme in ("glm", "kimi"):
            s = rd[scheme]
            print(f"{scheme}: acc={s['accuracy']:.3f} 分类={ {k: round(v,2) for k,v in s['by_class'].items()} }")
            print(f"   漏检(故障判正常)={s['fault_missed_as_normal']} 误报(正常判故障)={s['healthy_false_alarmed']} "
                  f"错型={s['wrong_fault_type']} 判其他={s['n_verdict_other_FTF_cage']}")
        print(f"一致性: {rd['agreement']['n_agree']}/{rd['agreement']['n_total']} "
              f"({rd['agreement']['rate']:.1%}) kappa={rd['agreement']['kappa']:.3f}")
    print("\nkimi 正常类 top 分（A 轮）:", stats["kimi_score_scale"]["healthy_top_scores_A"])
    print("两方案一致但与标签不符（A 轮）:", len(stats["A_main_1150"]["agree_both_wrong_vs_label"]))
    for it in stats["A_main_1150"]["agree_both_wrong_vs_label"][:10]:
        print("   ", it)


if __name__ == "__main__":
    main()

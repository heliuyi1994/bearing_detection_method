"""命令行入口：bearing-fault simulate / diagnose。

- ``simulate``：生成一段仿真信号，保存 signal.npz 与波形图；
- ``diagnose``：对新仿真信号或 simulate 保存的 npz 执行完整诊断，
  输出波形图、包络谱图、得分图与 summary.json，并在终端打印报告。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # CLI 强制无头后端，避免无显示环境报错

import matplotlib.pyplot as plt
import numpy as np

from .bearing import Bearing
from .pipeline import DiagnosisPipeline
from .plotting import (
    FAULT_LABELS,
    plot_envelope_report,
    plot_scores,
    plot_waveform,
)
from .simulator import FAULT_TYPES, BearingSimulator


def _add_sim_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--fault", default="outer", choices=FAULT_TYPES, help="故障类型")
    p.add_argument("--fs", type=float, default=12000.0, help="采样频率 Hz")
    p.add_argument("--duration", type=float, default=5.0, help="信号时长 s")
    p.add_argument("--rpm", type=float, default=1797.0, help="轴转速 rpm")
    p.add_argument("--snr", type=float, default=-6.0, help="信噪比 dB")
    p.add_argument("--seed", type=int, default=42, help="随机种子")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bearing-fault",
        description="基于传统信号处理的轴承故障诊断：仿真 + 共振解调包络分析",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_sim = sub.add_parser("simulate", help="生成仿真信号并保存")
    _add_sim_args(p_sim)
    p_sim.add_argument("--out", default="out_simulate", help="输出目录")

    p_diag = sub.add_parser("diagnose", help="执行完整诊断流程")
    _add_sim_args(p_diag)
    p_diag.add_argument("--input", help="simulate 保存的 signal.npz；给出后忽略仿真参数")
    p_diag.add_argument("--band", nargs=2, type=float, metavar=("LOW", "HIGH"),
                        help="共振解调带通频带 Hz，缺省自动")
    p_diag.add_argument("--out", default="out_diagnose", help="输出目录")

    p_ds = sub.add_parser("ml-dataset", help="批量仿真并生成 ML 特征数据集")
    p_ds.add_argument("--out", default="data/ml_dataset.npz", help="输出 npz 路径")
    p_ds.add_argument("--fs", type=float, default=12000.0, help="采样频率 Hz")
    p_ds.add_argument("--duration", type=float, default=3.0, help="单样本时长 s")
    p_ds.add_argument("--seeds", type=int, default=8, help="每网格点的种子数")

    p_tr = sub.add_parser("ml-train", help="训练两级 ML 模型并评估")
    p_tr.add_argument("--dataset", default="data/ml_dataset.npz", help="数据集 npz")
    p_tr.add_argument("--out", default="models", help="模型与指标输出目录")
    p_tr.add_argument("--test-size", type=float, default=0.3, help="测试集比例")
    p_tr.add_argument("--seed", type=int, default=42, help="随机种子")
    p_tr.add_argument("--stage1", default="elliptic_envelope",
                      choices=["elliptic_envelope", "isolation_forest"],
                      help="检测级算法（默认 elliptic_envelope）")

    p_pr = sub.add_parser("ml-predict", help="加载模型对信号做两级推理")
    p_pr.add_argument("--model", default="models/two_stage.joblib", help="模型文件")
    p_pr.add_argument("--input", help="simulate 保存的 signal.npz；给出后忽略仿真参数")
    _add_sim_args(p_pr)
    return parser


def _simulate(args: argparse.Namespace) -> tuple[np.ndarray, BearingSimulator]:
    sim = BearingSimulator(
        fs=args.fs, duration=args.duration, rpm=args.rpm, seed=args.seed,
    )
    x = sim.signal(fault=args.fault, snr_db=args.snr)
    return x, sim


def _cmd_simulate(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    x, sim = _simulate(args)
    t = np.arange(x.size) / sim.fs
    np.savez(out / "signal.npz", signal=x, fs=sim.fs, rpm=sim.rpm, fault=args.fault)

    fig, _ = plot_waveform(t, x, title=f"仿真信号：{FAULT_LABELS[args.fault]}")
    fig.savefig(out / "waveform.png", dpi=150)
    plt.close(fig)

    fr = sim.rpm / 60.0
    freqs = sim.bearing.characteristic_frequencies(fr)
    print(f"已保存: {out/'signal.npz'} 与 {out/'waveform.png'}")
    print(f"转频 fr = {fr:.2f} Hz，理论特征频率:")
    for name, f in freqs.items():
        print(f"  {FAULT_LABELS[name]:<8} {f:8.2f} Hz")
    return 0


def _cmd_diagnose(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    true_fault = None
    if args.input:
        data = np.load(args.input)
        x, fs, rpm = data["signal"], float(data["fs"]), float(data["rpm"])
        true_fault = str(data["fault"])
    else:
        x, sim = _simulate(args)
        fs, rpm, true_fault = sim.fs, sim.rpm, args.fault

    band = tuple(args.band) if args.band else None
    pipe = DiagnosisPipeline(fs=fs, band=band)
    report = pipe.run(x, rpm)

    t = np.arange(x.size) / fs
    fig, _ = plot_waveform(t, x, title="待诊断信号波形")
    fig.savefig(out / "waveform.png", dpi=150)
    plt.close(fig)

    fig, _ = plot_envelope_report(report)
    fig.savefig(out / "envelope_spectrum.png", dpi=150)
    plt.close(fig)

    fig, _ = plot_scores(report.scores, report.verdict)
    fig.savefig(out / "scores.png", dpi=150)
    plt.close(fig)

    summary = report.summary()
    summary["true_fault"] = true_fault
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"采样频率 {fs:.0f} Hz，转速 {rpm:.0f} rpm（转频 {rpm/60:.2f} Hz）")
    print(f"解调频带 {report.band[0]:.0f}–{report.band[1]:.0f} Hz")
    print("理论特征频率: " + ", ".join(
        f"{FAULT_LABELS[k]} {v:.2f} Hz" for k, v in report.char_freqs.items()
    ))
    print("谱峰匹配得分: " + ", ".join(
        f"{FAULT_LABELS[k]} {v:.4f}" for k, v in report.scores.items()
    ))
    err = report.peak_error()
    err_txt = f"{err*100:.2f}%" if err is not None else "无命中"
    print(f"诊断结论: {FAULT_LABELS.get(report.verdict, report.verdict)}"
          f"（最强命中峰误差 {err_txt}）")
    if true_fault:
        ok = "✓ 正确" if report.verdict == true_fault else "✗ 与真实类型不符"
        print(f"真实类型: {FAULT_LABELS[true_fault]} → {ok}")
    print(f"结果已保存至 {out}/")
    return 0


def _cmd_ml_dataset(args: argparse.Namespace) -> int:
    from .ml import DatasetConfig, generate, save_npz

    cfg = DatasetConfig(fs=args.fs, duration=args.duration, n_seeds=args.seeds)
    print(f"参数网格: {len(cfg.faults)} 类 × {len(cfg.rpms)} 转速 × "
          f"{len(cfg.snrs)} SNR × {cfg.n_seeds} 种子 = {cfg.grid_size()} 样本")
    dataset = generate(cfg, verbose=True)
    path = save_npz(args.out, dataset)
    counts = {c: int((dataset.y == i).sum()) for i, c in enumerate(dataset.classes)}
    print(f"已保存 {path}，X 形状 {dataset.X.shape}，各类样本数 {counts}")
    return 0


def _cmd_ml_train(args: argparse.Namespace) -> int:
    from .ml import evaluate, load_npz, train_two_stage

    dataset = load_npz(args.dataset)
    result = train_two_stage(
        dataset, test_size=args.test_size, seed=args.seed, stage1=args.stage1
    )
    print(f"检测级算法: {args.stage1}")
    print("分类级交叉验证（5 折平均准确率）:")
    for name, score in result.cv_scores.items():
        mark = "  ← 择优" if name == result.best_stage2 else ""
        print(f"  {name:<14} {score:.4f}{mark}")
    metrics = evaluate(result, out_dir=args.out)
    model_path = result.model.save(Path(args.out) / "two_stage.joblib")
    det = metrics["detection"]
    print(f"测试集（{len(result.y_test)} 样本）级联 5 类准确率: {metrics['accuracy']:.4f}")
    print(f"检测级: 故障检出率 {det['detection_rate']:.4f}，"
          f"健康误报率 {det['false_alarm_rate']:.4f}")
    print(f"模型已保存至 {model_path}，指标与混淆矩阵见 {args.out}/")
    return 0


def _cmd_ml_predict(args: argparse.Namespace) -> int:
    from .ml import TwoStageModel

    model = TwoStageModel.load(args.model)
    true_fault = None
    if args.input:
        data = np.load(args.input)
        x, rpm = data["signal"], float(data["rpm"])
        source = args.input
        if "fault" in data:
            true_fault = str(data["fault"])
    else:
        x, sim = _simulate(args)
        rpm, source, true_fault = sim.rpm, f"仿真（{args.fault}）", args.fault

    res = model.predict_one(x, rpm)
    print(f"输入: {source}")
    print(f"检测级: {'异常' if res['anomaly'] else '正常'}")
    print(f"诊断结论: {FAULT_LABELS.get(res['verdict'], res['verdict'])}")
    if res["fault_proba"]:
        probs = ", ".join(
            f"{FAULT_LABELS.get(k, k)} {v:.3f}"
            for k, v in sorted(res["fault_proba"].items(), key=lambda kv: -kv[1])
        )
        print(f"分类概率: {probs}")
    if true_fault:
        ok = "✓ 正确" if res["verdict"] == true_fault else "✗ 不符"
        print(f"真实类型: {FAULT_LABELS[true_fault]} → {ok}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "simulate":
        return _cmd_simulate(args)
    if args.command == "diagnose":
        return _cmd_diagnose(args)
    if args.command == "ml-dataset":
        return _cmd_ml_dataset(args)
    if args.command == "ml-train":
        return _cmd_ml_train(args)
    if args.command == "ml-predict":
        return _cmd_ml_predict(args)
    return 2  # pragma: no cover


if __name__ == "__main__":
    sys.exit(main())

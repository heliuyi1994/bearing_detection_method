"""命令行入口：bearing-diag <analyze|simulate|bearings>。

用法示例：

    # 分析实测/仿真数据（需采样率与转速）
    bearing-diag analyze vibration.csv --fs 12000 --rpm 1797 \
        --bearing SKF6205 --out report/

    # 已知特征频率时跳过轴承几何参数
    bearing-diag analyze data.npy --fs 48000 --shaft-freq 29.95 \
        --fault-freqs '{"BPFO": 107.4, "BPFI": 162.2, "BSF": 70.6, "FTF": 11.9}'

    # 生成仿真故障信号并直接诊断（快速体验）
    bearing-diag simulate --fault BPFO --snr 5 --out sim_bpfo.csv
    bearing-diag analyze sim_bpfo.csv --fs 12000 --rpm 1797 --bearing SKF6205

    # 查看内置轴承
    bearing-diag bearings [--rpm 1797]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .fault_freqs import (
    BUILTIN_BEARINGS,
    FAULT_NAMES_CN,
    get_bearing,
    load_bearing_json,
)


def _build_fault_freqs(args) -> tuple[dict | None, object | None]:
    """从 CLI 参数解析特征频率：--fault-freqs > --bearing-json > --bearing。"""
    if args.fault_freqs:
        try:
            ff = json.loads(args.fault_freqs)
        except json.JSONDecodeError as e:
            raise ValueError(f"--fault-freqs JSON 解析失败: {e}") from e
        if not isinstance(ff, dict) or not ff:
            raise ValueError("--fault-freqs 须为非空 JSON 对象，"
                             '如 \'{"BPFO": 107.4, "BPFI": 162.2}\'')
        return {str(k).upper(): float(v) for k, v in ff.items()}, None
    if getattr(args, "bearing_json", None):
        bearing = load_bearing_json(args.bearing_json)
        return None, bearing
    if getattr(args, "bearing", None):
        bearing = get_bearing(args.bearing)  # 未知名称抛 KeyError，由 main 统一处理
        return None, bearing
    return None, None


def _resolve_shaft_freq(args) -> float:
    if args.shaft_freq:
        return args.shaft_freq
    if args.rpm:
        return args.rpm / 60.0
    raise ValueError("必须提供 --rpm 或 --shaft-freq 之一（轴转频率）")


def cmd_analyze(args) -> int:
    from .diagnose import diagnose
    from .loaders import load_signal
    from .report import write_reports

    shaft_freq = _resolve_shaft_freq(args)
    fault_freqs, bearing = _build_fault_freqs(args)
    if fault_freqs is None and bearing is None:
        raise ValueError(
            "缺少特征频率来源：请提供 --bearing <名称>、--bearing-json <文件> "
            "或 --fault-freqs '<JSON>'"
        )

    x, fs, meta = load_signal(args.input, fs=args.fs, column=args.column)
    if args.fs is None and fs:
        pass  # load_signal 已从时间列推断

    manual_band = None
    if args.band:
        parts = args.band.split(":")
        if len(parts) != 2:
            raise SystemExit("--band 格式为 LOW:HIGH，如 2000:4000")
        manual_band = (float(parts[0]), float(parts[1]))

    result = diagnose(
        x, fs=fs, shaft_freq=shaft_freq,
        fault_freqs=fault_freqs, bearing=bearing,
        use_kurtogram=not args.no_kurtogram,
        manual_band=manual_band,
        kurtogram_level=args.kurtogram_level,
        n_harmonics=args.harmonics,
    )

    from .report import build_text_report

    print(build_text_report(result, source_desc=args.input))
    out_dir = args.out or "report"
    paths = write_reports(result, out_dir, source_desc=args.input)
    print(f"\n[报告已输出] {paths['png']}\n            {paths['txt']}\n"
          f"            {paths['json']}", file=sys.stderr)
    return 0


def cmd_simulate(args) -> int:
    from .loaders import save_signal_csv
    from .simulate import simulate

    sig, info = simulate(
        fault=args.fault, fs=args.fs, duration=args.duration,
        rpm=args.rpm, bearing=get_bearing(args.bearing), snr_db=args.snr,
        jitter=args.jitter, resonance_hz=args.resonance, seed=args.seed,
    )
    out = args.out or f"sim_{args.fault}.csv"
    save_signal_csv(out, sig, info["fs"])
    print(f"已生成仿真信号: {out}")
    print(f"  故障类型      : {args.fault} ({FAULT_NAMES_CN.get(args.fault, '正常')})")
    print(f"  采样率/时长   : {info['fs']:.0f} Hz / {info['duration']:.2f} s")
    print(f"  转速          : {info['rpm']:.0f} rpm (fr={info['shaft_freq']:.3f} Hz)")
    print(f"  真实特征频率  : "
          + ", ".join(f"{k}={v:.2f}" for k, v in info["true_fault_freqs"].items()))
    print(f"  SNR / 共振频率: {info['snr_db']} dB / {info['resonance_hz']:.0f} Hz")
    print(f"  下一步        : bearing-diag analyze {out} --fs {info['fs']:.0f} "
          f"--rpm {info['rpm']:.0f} --bearing {args.bearing}")
    return 0


def cmd_ml_train(args) -> int:
    from .ml.dataset import make_sim_dataset
    from .ml.evaluate import (
        compare_with_traditional,
        hierarchical_cv_report,
        loso_evaluate,
        loso_evaluate_two_stage,
    )
    from .ml.train import save_model, train_pipeline, train_pipeline_two_stage

    print(f"生成仿真数据集: 每类 {args.n_base} 条基录音 × {args.windows} 窗 "
          f"(架构: {args.arch}, 物理特征: {'开' if not args.no_physical else '关'})…")
    ds = make_sim_dataset(
        n_base_per_class=args.n_base, windows_per_base=args.windows,
        window_s=args.window_s, fs=args.fs, bearing=args.bearing,
        seed=args.seed, use_physical=not args.no_physical,
    )
    print(f"数据集: {ds.n_samples} 样本 × {ds.n_features} 特征, "
          f"{len(set(ds.groups))} 个防泄漏分组, 类别分布 {ds.label_counts()}")

    metrics: dict = {}
    if args.arch == "two_stage":
        m1, m2, report = train_pipeline_two_stage(ds)
        s1_spec = {"model": report["stage1"]["best_model"],
                   "best_params": report["stage1"]["best_params"]}
        s2_spec = {"model": report["stage2"]["best_model"],
                   "best_params": report["stage2"]["best_params"]}

        print("\n两级架构分级评估（GroupKFold 防泄漏）:")
        hier = hierarchical_cv_report(ds, s1_spec, s2_spec)
        print("  Stage1 检测器阈值工作点（漏报=故障判正常, 误报=正常判异常）:")
        for thr, m in hier["stage1"].items():
            print(f"    阈值 {thr}: acc={m['accuracy']:.4f} f1={m['f1']:.4f} "
                  f"漏报率={m['miss_rate']:.3f} 误报率={m['false_alarm_rate']:.3f}")
        print(f"  Stage2 四类故障: acc={hier['stage2']['accuracy']:.4f} "
              f"macroF1={hier['stage2']['macro_f1']:.4f}")
        print("  端到端层次准确率: "
              + "  ".join(f"thr{t}={a:.4f}" for t, a in
                          hier["end_to_end_accuracy"].items()))

        print("\n留一转速验证（LOSO，跨工况泛化）:")
        loso = loso_evaluate_two_stage(ds, s1_spec, s2_spec)

        n_cmp = max(3, min(8, args.n_base))
        print("\n独立测试集上与传统方法对比:")
        bundle = {"arch": "two_stage", "model": None,
                  "stage1": {"model": m1}, "stage2": {"model": m2},
                  "threshold": 0.5,
                  "feature_names": ds.feature_names_,
                  "classes": list(ds.label_counts()), "use_physical": ds.use_physical}
        cmp_trad = compare_with_traditional(
            bundle, fs=args.fs, window_s=2.0, n_per_class=n_cmp,
            bearing_name=args.bearing, seed=args.seed + 1)

        metrics = {
            "arch": "two_stage",
            "stage1": report["stage1"], "stage2": report["stage2"],
            "hierarchical_cv": hier,
            "loso": loso,
            "traditional_compare": {k: cmp_trad[k] for k in
                                    ("ml_accuracy", "traditional_accuracy",
                                     "agreement", "n")},
        }
        out_dir = Path(args.out)
        model_path = save_model(
            out_dir / "model.joblib",
            feature_names_=ds.feature_names_,
            classes=list(ds.label_counts()),
            use_physical=ds.use_physical,
            config={"fs": args.fs, "bearing": args.bearing,
                    "window_s": args.window_s, "dataset_seed": args.seed,
                    "arch": "two_stage"},
            metrics=metrics,
            stage1=m1, stage2=m2, stage1_spec=s1_spec, stage2_spec=s2_spec,
        )
    else:  # flat（单级，A/B 对比用）
        model, report = train_pipeline(ds)
        print("\n留一转速验证（LOSO，跨工况泛化）:")
        loso = loso_evaluate(ds, model)
        n_cmp = max(3, min(8, args.n_base))
        print("\n独立测试集上与传统方法对比:")
        cmp_trad = compare_with_traditional(
            {"model": model, "classes": list(model.classes_),
             "feature_names": ds.feature_names_,
             "use_physical": ds.use_physical},
            fs=args.fs, window_s=2.0, n_per_class=n_cmp,
            bearing_name=args.bearing, seed=args.seed + 1)
        metrics = {"arch": "flat",
                   "groupkfold_comparison": report["comparison"],
                   "best_model": report["best_model"],
                   "best_params": report["best_params"],
                   "cv_macro_f1": report["cv_macro_f1"],
                   "loso": loso, "traditional_compare": {
                       k: cmp_trad[k] for k in ("ml_accuracy", "traditional_accuracy",
                                                "agreement", "n")}}
        out_dir = Path(args.out)
        model_path = save_model(
            out_dir / "model.joblib", model,
            feature_names_=ds.feature_names_,
            classes=list(model.classes_), use_physical=ds.use_physical,
            config={"fs": args.fs, "bearing": args.bearing,
                    "window_s": args.window_s, "dataset_seed": args.seed,
                    "arch": "flat"},
            metrics=metrics,
        )

    (out_dir / "training_report.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[模型已保存] {model_path}")
    print(f"[评估报告]   {out_dir / 'training_report.json'}")
    pred_extra = (f" --rpm <rpm> --bearing {args.bearing}"
                  if not args.no_physical else "")
    print(f"预测示例: bearing-diag predict <数据> --fs {args.fs:.0f} "
          f"--model {model_path}{pred_extra}"
          + (" [--threshold 0.3]" if args.arch == "two_stage" else ""))
    return 0


def cmd_predict(args) -> int:
    from .ml.infer import format_prediction, predict_file
    from .ml.train import load_model

    bundle = load_model(args.model)
    shaft_freq = None
    if args.rpm or args.shaft_freq:
        shaft_freq = args.shaft_freq or (args.rpm / 60.0)
    result = predict_file(
        args.input, bundle, fs=args.fs, shaft_freq=shaft_freq,
        bearing_name=args.bearing, column=args.column,
        compare_traditional=args.compare,
        threshold=args.threshold,
    )
    print(format_prediction(result))
    return 0


def cmd_bearings(args) -> int:
    rpm = args.rpm
    print("内置轴承参数库（长度单位 mm）：")
    for b in BUILTIN_BEARINGS.values():
        print(f"\n  {b.name}: 滚动体数 n={b.n_balls}, 节圆直径 D={b.pitch_diameter}, "
              f"滚动体直径 d={b.ball_diameter}, 接触角 α={b.contact_angle_deg}°")
        if rpm:
            ff = b.fault_frequencies(rpm / 60.0)
            print("    特征频率 @ "
                  + ", ".join(f"{k}={v:.2f} Hz" for k, v in ff.items()))
    if not rpm:
        print("\n  提示: 加 --rpm 1797 可查看该转速下的特征频率")
    print("\n自定义轴承: 用 --bearing-json 提供 "
          '{"name","n_balls","pitch_diameter","ball_diameter","contact_angle_deg"}')
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bearing-diag",
        description="滚动轴承故障诊断（传统信号处理：时域指标/包络谱/Kurtogram/特征频率匹配）",
    )
    sub = p.add_subparsers(dest="command", required=True)

    pa = sub.add_parser("analyze", help="分析振动信号并输出诊断报告")
    pa.add_argument("input", help="数据文件 (csv/txt/dat/npy/npz/mat)")
    pa.add_argument("--fs", type=float, help="采样率 Hz（数据含时间列时可省略）")
    pa.add_argument("--rpm", type=float, help="转速 rpm（与 --shaft-freq 二选一）")
    pa.add_argument("--shaft-freq", type=float, dest="shaft_freq",
                    help="轴频 Hz（与 --rpm 二选一）")
    pa.add_argument("--bearing", default="SKF6205",
                    help="内置轴承名（默认 SKF6205；可选 %s）"
                    % "/".join(BUILTIN_BEARINGS))
    pa.add_argument("--bearing-json", dest="bearing_json",
                    help="自定义轴承参数 JSON 文件（优先于 --bearing）")
    pa.add_argument("--fault-freqs", dest="fault_freqs",
                    help='直接给定特征频率 JSON（优先于轴承参数）')
    pa.add_argument("--column", type=int, help="多列数据取第几列（默认自动/第0列）")
    pa.add_argument("--band", help="手动指定解调频带 LOW:HIGH（默认 Kurtogram 自动）")
    pa.add_argument("--no-kurtogram", action="store_true",
                    help="禁用 Kurtogram，对全带做包络（不推荐）")
    pa.add_argument("--kurtogram-level", type=int, default=4,
                    dest="kurtogram_level", help="Kurtogram 分解层数（默认 4）")
    pa.add_argument("--harmonics", type=int, default=5,
                    help="谐波匹配次数（默认 5）")
    pa.add_argument("--out", default="report", help="报告输出目录（默认 ./report）")
    pa.set_defaults(func=cmd_analyze)

    ps = sub.add_parser("simulate", help="生成轴承故障仿真信号（CSV）")
    ps.add_argument("--fault", default="BPFO",
                    choices=["normal", "BPFO", "BPFI", "BSF", "FTF"],
                    help="故障类型（默认 BPFO；normal 为健康信号）")
    ps.add_argument("--fs", type=float, default=12000.0)
    ps.add_argument("--duration", type=float, default=1.0)
    ps.add_argument("--rpm", type=float, default=1797.0)
    ps.add_argument("--bearing", default="SKF6205")
    ps.add_argument("--snr", type=float, default=5.0, help="信噪比 dB（默认 5）")
    ps.add_argument("--jitter", type=float, default=0.01, help="冲击滑移比例")
    ps.add_argument("--resonance", type=float, default=3000.0,
                    help="共振频率 Hz（默认 3000）")
    ps.add_argument("--seed", type=int, default=None)
    ps.add_argument("--out", default=None, help="输出 CSV 路径")
    ps.set_defaults(func=cmd_simulate)

    pm = sub.add_parser("ml-train", help="训练机器学习故障分类模型（仿真数据）")
    pm.add_argument("--arch", choices=["two_stage", "flat"], default="two_stage",
                    help="分类架构: two_stage 两级(默认, 先正常/异常再四类故障) "
                         "/ flat 单级 5 分类(A/B 对比)")
    pm.add_argument("--n-base", type=int, default=15,
                    help="每类基录音数（默认 15，防泄漏分组单位）")
    pm.add_argument("--windows", type=int, default=8,
                    help="每条基录音切窗数（默认 8）")
    pm.add_argument("--window-s", type=float, default=1.0, dest="window_s",
                    help="每窗时长秒（默认 1.0）")
    pm.add_argument("--fs", type=float, default=12000.0)
    pm.add_argument("--bearing", default="SKF6205")
    pm.add_argument("--no-physical", action="store_true", dest="no_physical",
                    help="消融：剔除特征频率物理特征（无需转速信息的盲特征路线）")
    pm.add_argument("--seed", type=int, default=2026)
    pm.add_argument("--out", default="models", help="模型输出目录（默认 ./models）")
    pm.set_defaults(func=cmd_ml_train)

    pp = sub.add_parser("predict", help="用训练好的 ML 模型预测故障类别")
    pp.add_argument("input", help="数据文件 (csv/txt/dat/npy/npz/mat)")
    pp.add_argument("--model", default="models/model.joblib", help="模型文件")
    pp.add_argument("--fs", type=float, help="采样率 Hz")
    pp.add_argument("--rpm", type=float, help="转速（物理特征模型必需）")
    pp.add_argument("--shaft-freq", type=float, dest="shaft_freq",
                    help="轴频 Hz（与 --rpm 二选一）")
    pp.add_argument("--bearing", default="SKF6205",
                    help="轴承型号（物理特征模型必需）")
    pp.add_argument("--column", type=int, help="多列数据取第几列")
    pp.add_argument("--threshold", type=float, default=None,
                    help="两级架构 Stage1 异常判定阈值（默认用模型包内 0.5；"
                         "调低→更灵敏报警, 调高→更保守）")
    pp.add_argument("--compare", action="store_true",
                    help="同时运行传统方法并对比两法结论")
    pp.set_defaults(func=cmd_predict)

    pb = sub.add_parser("bearings", help="列出内置轴承参数与特征频率")
    pb.add_argument("--rpm", type=float, default=None)
    pb.set_defaults(func=cmd_bearings)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"错误: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

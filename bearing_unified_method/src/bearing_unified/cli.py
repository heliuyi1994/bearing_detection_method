# -*- coding: utf-8 -*-
"""命令行入口：bearing-unified <analyze|simulate|bearings|bands>。

用法示例：

    # 仿真内圈故障并用现场台架轴承分析（v2 规则，默认轴承 FIELD7）
    bearing-unified simulate --fault BPFI --snr 5 --seed 42 --out sim_bpfi.csv
    bearing-unified analyze sim_bpfi.csv --fs 12000 --bearing FIELD7 --out report/

    # 自定义转速窗（定位频带随之重算）
    bearing-unified analyze data.wav --fs 16000 --bearing FIELD7 --rpm-window 1050:1250

    # 查看内置轴承与定位频带
    bearing-unified bearings --rpm 1150
    bearing-unified bands --bearing FIELD7 --rpm-window 1100:1200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .fault_freqs import BUILTIN_BEARINGS, FAULT_NAMES_CN, get_bearing
from .localize import RPM_WINDOW_DEFAULT, class_bands


def _parse_rpm_window(s: str) -> tuple[float, float]:
    parts = s.split(":")
    if len(parts) != 2:
        raise ValueError("--rpm-window 格式为 LO:HI，如 1100:1200")
    lo, hi = float(parts[0]), float(parts[1])
    if not (0 < lo < hi):
        raise ValueError(f"--rpm-window 需要 0 < LO < HI，得到 {s}")
    return (lo, hi)


def cmd_analyze(args) -> int:
    from .diagnose import diagnose
    from .loaders import load_signal
    from .report import write_reports

    bearing = get_bearing(args.bearing)
    rpm_window = _parse_rpm_window(args.rpm_window) if args.rpm_window else RPM_WINDOW_DEFAULT

    x, fs, meta = load_signal(args.input, fs=args.fs, column=args.column)

    manual_band = None
    if args.band:
        parts = args.band.split(":")
        if len(parts) != 2:
            raise ValueError("--band 格式为 LOW:HIGH，如 2000:4000")
        manual_band = (float(parts[0]), float(parts[1]))

    result = diagnose(
        x, fs=fs, bearing=bearing, rpm=args.rpm, rpm_window=rpm_window,
        manual_band=manual_band,
        kurtogram_level=args.kurtogram_level,
        n_harmonics=args.harmonics,
        conf_threshold=args.conf_threshold,
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
    print(f"  下一步        : bearing-unified analyze {out} --fs {info['fs']:.0f} "
          f"--bearing {args.bearing}")
    return 0


def cmd_bearings(args) -> int:
    rpm = args.rpm
    print("内置轴承参数库（长度单位 mm；FIELD7 为现场台架注册轴承）：")
    for b in BUILTIN_BEARINGS.values():
        print(f"\n  {b.name}: 滚动体数 n={b.n_balls}, 节圆直径 D={b.pitch_diameter}, "
              f"滚动体直径 d={b.ball_diameter}, 接触角 α={b.contact_angle_deg}°")
        if rpm:
            ff = b.fault_frequencies(rpm / 60.0)
            print("    特征频率 @ "
                  + ", ".join(f"{k}={v:.2f} Hz" for k, v in ff.items()))
    if not rpm:
        print("\n  提示: 加 --rpm 1150 可查看该转速下的特征频率")
    print("\n自定义轴承: diagnose() 可直接接受 Bearing 对象；"
          "现场轴承注册见 fault_freqs.register_field_bearing")
    return 0


def cmd_bands(args) -> int:
    bearing = get_bearing(args.bearing)
    rpm_window = _parse_rpm_window(args.rpm_window)
    bands = class_bands(bearing, rpm_window)
    print(f"定位频带（{bearing.name} @ {rpm_window[0]:.0f}–{rpm_window[1]:.0f} rpm 现算）：")
    for b in bands:
        print(f"  {b['label']:16s}: [{b['lo']:7.3f}, {b['hi']:7.3f}] Hz"
              f"（阶次 {b['order']:.4f}×fr，k={b['k']}）")
    print("\n判别仅 3 类：外圈 BPFO / 内圈 BPFI / 滚珠 BSF（2×BSF 带归入滚珠）；"
          "频带间空档走 fr_hyp 补救，仍无解判「轴承-未定位」。")
    return 0


def cmd_ml_dataset(args) -> int:
    from .ml.dataset import make_sim_dataset, save_dataset_npz

    print(f"生成仿真数据集（FIELD7 恒速窗 1100–1200 rpm 四点网格；"
          f"物理特征:{'开' if not args.no_physical else '关'} "
          f"梳特征:{'开' if not args.no_comb else '关'}）…")
    ds = make_sim_dataset(
        n_base_per_class=args.n_base, windows_per_base=args.windows,
        window_s=args.window_s, fs=args.fs, bearing=args.bearing,
        seed=args.seed, use_physical=not args.no_physical,
        use_comb=not args.no_comb, verbose=False,
    )
    print(f"数据集: {ds.n_samples} 样本 × {ds.n_features} 特征, "
          f"{len(set(ds.groups))} 个防泄漏分组, 类别分布 {ds.label_counts()}")
    p = save_dataset_npz(ds, args.out)
    print(f"[数据集已保存] {p}（+ {p.stem}.json）")
    return 0


def _load_or_make_dataset(args, rpm_window):
    from .ml.dataset import load_dataset_npz, make_sim_dataset

    if getattr(args, "dataset", None):
        if args.no_physical or args.no_comb:
            print("注意：--dataset 的特征构成在 ml-dataset 生成时已固定，"
                  "--no-physical/--no-comb 对已有数据集不生效"
                  "（如需消融，请用对应开关重新 ml-dataset 生成）")
        print(f"加载数据集: {args.dataset}")
        return load_dataset_npz(args.dataset)
    print(f"生成仿真数据集: 每类 {args.n_base} 条基录音 × {args.windows} 窗 "
          f"(架构: {args.arch}, 物理特征: {'开' if not args.no_physical else '关'}, "
          f"梳特征: {'开' if not args.no_comb else '关'})…")
    return make_sim_dataset(
        n_base_per_class=args.n_base, windows_per_base=args.windows,
        window_s=args.window_s, fs=args.fs, bearing=args.bearing,
        seed=args.seed, use_physical=not args.no_physical,
        use_comb=not args.no_comb, rpm_window=rpm_window,
    )


def cmd_ml_train(args) -> int:
    import json

    from .localize import RPM_WINDOW_DEFAULT
    from .ml.evaluate import (
        compare_with_traditional,
        hierarchical_cv_report,
        loso_evaluate,
        loso_evaluate_two_stage,
    )
    from .ml.train import save_model, train_pipeline, train_pipeline_two_stage

    rpm_window = (_parse_rpm_window(args.rpm_window) if args.rpm_window
                  else RPM_WINDOW_DEFAULT)
    ds = _load_or_make_dataset(args, rpm_window)
    print(f"数据集: {ds.n_samples} 样本 × {ds.n_features} 特征, "
          f"{len(set(ds.groups))} 个防泄漏分组, 类别分布 {ds.label_counts()}")

    metrics: dict = {}
    if args.arch == "two_stage":
        m1, m2, report = train_pipeline_two_stage(
            ds, stage1=args.stage1, novelty_backend=args.novelty_backend)
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
        print(f"  Stage2 三类故障: acc={hier['stage2']['accuracy']:.4f} "
              f"macroF1={hier['stage2']['macro_f1']:.4f}")
        print("  端到端层次准确率: "
              + "  ".join(f"thr{t}={a:.4f}" for t, a in
                          hier["end_to_end_accuracy"].items()))

        print("\n留一转速验证（LOSO，跨工况泛化）:")
        loso = loso_evaluate_two_stage(ds, s1_spec, s2_spec)

        n_cmp = max(3, min(8, args.n_base))
        print("\n独立测试集上与 v2 DSP 对比:")
        bundle = {"arch": "two_stage", "model": None,
                  "stage1": {"model": m1}, "stage2": {"model": m2},
                  "threshold": 0.5,
                  "feature_names": ds.feature_names_,
                  "classes": list(ds.label_counts()),
                  "use_physical": ds.use_physical, "use_comb": ds.use_comb}
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
            use_physical=ds.use_physical, use_comb=ds.use_comb,
            config={"fs": args.fs, "bearing": args.bearing,
                    "window_s": args.window_s, "dataset_seed": args.seed,
                    "arch": "two_stage", "stage1_backend": args.stage1,
                    "rpm_window": list(rpm_window)},
            metrics=metrics,
            stage1=m1, stage2=m2, stage1_spec=s1_spec, stage2_spec=s2_spec,
        )
    else:  # flat（单级，A/B 对比用）
        model, report = train_pipeline(ds)
        print("\n留一转速验证（LOSO，跨工况泛化）:")
        loso = loso_evaluate(ds, model)
        n_cmp = max(3, min(8, args.n_base))
        print("\n独立测试集上与 v2 DSP 对比:")
        cmp_trad = compare_with_traditional(
            {"model": model, "classes": list(model.classes_),
             "feature_names": ds.feature_names_,
             "use_physical": ds.use_physical, "use_comb": ds.use_comb},
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
            classes=list(model.classes_),
            use_physical=ds.use_physical, use_comb=ds.use_comb,
            config={"fs": args.fs, "bearing": args.bearing,
                    "window_s": args.window_s, "dataset_seed": args.seed,
                    "arch": "flat",
                    "rpm_window": list(rpm_window)},
            metrics=metrics,
        )

    (out_dir / "training_report.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[模型已保存] {model_path}")
    print(f"[评估报告]   {out_dir / 'training_report.json'}")
    pred_extra = (f" --rpm <rpm> --bearing {args.bearing}"
                  if not args.no_physical else "")
    print(f"预测示例: bearing-unified predict <数据> --fs {args.fs:.0f} "
          f"--model {model_path}{pred_extra}"
          + (" [--threshold 0.3]" if args.arch == "two_stage" else ""))
    return 0


def cmd_predict(args) -> int:
    from .ml.infer import format_prediction, predict_file
    from .ml.train import load_model

    bundle = load_model(args.model)
    shaft_freq = args.shaft_freq or (args.rpm / 60.0 if args.rpm else None)
    result = predict_file(
        args.input, bundle, fs=args.fs, shaft_freq=shaft_freq,
        bearing_name=args.bearing, column=args.column,
        compare_traditional=args.compare,
        threshold=args.threshold,
    )
    print(format_prediction(result))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bearing-unified",
        description="统一轴承故障诊断（v2 盲梳检：Kurtogram/零相位解调/梳存在性/频带落入定位）",
    )
    sub = p.add_subparsers(dest="command", required=True)

    pa = sub.add_parser("analyze", help="分析振动信号并输出诊断报告（v2 规则）")
    pa.add_argument("input", help="数据文件 (csv/txt/dat/npy/npz/mat/wav)")
    pa.add_argument("--fs", type=float, help="采样率 Hz（数据含时间列时可省略）")
    pa.add_argument("--rpm", type=float, default=None,
                    help="真实转速 rpm（可选，仅显示参考；v2 匹配不依赖）")
    pa.add_argument("--rpm-window", dest="rpm_window", default=None,
                    help="标称转速窗 LO:HI（默认 1100:1200，定位频带随之重算）")
    pa.add_argument("--bearing", default="FIELD7",
                    help="内置轴承名（默认 FIELD7 现场台架；可选 %s）"
                    % "/".join(BUILTIN_BEARINGS))
    pa.add_argument("--column", type=int, help="多列数据取第几列（默认自动/第0列）")
    pa.add_argument("--band", help="手动指定解调频带 LOW:HIGH（默认 Kurtogram 自动）")
    pa.add_argument("--kurtogram-level", type=int, default=4,
                    dest="kurtogram_level", help="Kurtogram 分解层数（默认 4）")
    pa.add_argument("--harmonics", type=int, default=5,
                    help="谐波匹配次数（默认 5）")
    pa.add_argument("--conf-threshold", type=float, default=0.2,
                    dest="conf_threshold",
                    help="低置信标记的定位置信度门槛（默认 0.2 平衡档；"
                         "保守 0.3 / 灵敏 0.1）")
    pa.add_argument("--out", default="report", help="报告输出目录（默认 ./report）")
    pa.set_defaults(func=cmd_analyze)

    ps = sub.add_parser("simulate", help="生成轴承故障仿真信号（CSV）")
    ps.add_argument("--fault", default="BPFO",
                    choices=["normal", "BPFO", "BPFI", "BSF", "FTF"],
                    help="故障类型（默认 BPFO；normal 为健康信号；FTF 仅仿真，判别不参与）")
    ps.add_argument("--fs", type=float, default=16000.0)
    ps.add_argument("--duration", type=float, default=3.0)
    ps.add_argument("--rpm", type=float, default=1150.0,
                    help="转速 rpm（默认 1150，与 FIELD7 标称窗一致）")
    ps.add_argument("--bearing", default="FIELD7")
    ps.add_argument("--snr", type=float, default=5.0, help="信噪比 dB（默认 5）")
    ps.add_argument("--jitter", type=float, default=0.01, help="冲击滑移比例")
    ps.add_argument("--resonance", type=float, default=2200.0,
                    help="共振频率 Hz（默认 2200，本台架共振区）")
    ps.add_argument("--seed", type=int, default=None)
    ps.add_argument("--out", default=None, help="输出 CSV 路径")
    ps.set_defaults(func=cmd_simulate)

    pb = sub.add_parser("bearings", help="列出内置轴承参数与特征频率")
    pb.add_argument("--rpm", type=float, default=None)
    pb.set_defaults(func=cmd_bearings)

    pbd = sub.add_parser("bands", help="列出定位频带（转速窗×几何现算）")
    pbd.add_argument("--bearing", default="FIELD7")
    pbd.add_argument("--rpm-window", dest="rpm_window", default="1100:1200")
    pbd.set_defaults(func=cmd_bands)

    pd = sub.add_parser("ml-dataset", help="生成 ML 仿真数据集（npz，供 ml-train --dataset 复用）")
    pd.add_argument("--n-base", type=int, default=15,
                    help="每类基录音数（默认 15，防泄漏分组单位）")
    pd.add_argument("--windows", type=int, default=8,
                    help="每条基录音切窗数（默认 8）")
    pd.add_argument("--window-s", type=float, default=1.0, dest="window_s",
                    help="每窗时长秒（默认 1.0）")
    pd.add_argument("--fs", type=float, default=16000.0)
    pd.add_argument("--bearing", default="FIELD7")
    pd.add_argument("--no-physical", action="store_true", dest="no_physical",
                    help="消融：剔除 T2 物理谐波突出度（无需转速的盲特征路线）")
    pd.add_argument("--no-comb", action="store_true", dest="no_comb",
                    help="消融：剔除 T6 盲梳检结构化特征")
    pd.add_argument("--seed", type=int, default=2026)
    pd.add_argument("--out", required=True, help="输出 npz 路径（同时写同名 .json）")
    pd.set_defaults(func=cmd_ml_dataset)

    pm = sub.add_parser("ml-train", help="训练 ML 故障分类模型（仿真数据，两阶段 3 类）")
    pm.add_argument("--arch", choices=["two_stage", "flat"], default="two_stage",
                    help="分类架构: two_stage 两级(默认, 先正常/异常再三类故障) "
                         "/ flat 单级 4 分类(A/B 对比)")
    pm.add_argument("--stage1", choices=["supervised", "novelty"], default="supervised",
                    help="Stage1 后端: supervised 监督式(默认) / novelty 新奇检测(只训健康样本)")
    pm.add_argument("--novelty-backend", choices=["elliptic_envelope", "isolation_forest"],
                    default="elliptic_envelope", dest="novelty_backend",
                    help="新奇检测后端（默认 EllipticEnvelope；IF 有塌缩教训，仅对照）")
    pm.add_argument("--dataset", default=None,
                    help="已有数据集 npz（缺省则按下列参数现场生成）")
    pm.add_argument("--n-base", type=int, default=15,
                    help="每类基录音数（默认 15，防泄漏分组单位）")
    pm.add_argument("--windows", type=int, default=8,
                    help="每条基录音切窗数（默认 8）")
    pm.add_argument("--window-s", type=float, default=1.0, dest="window_s",
                    help="每窗时长秒（默认 1.0）")
    pm.add_argument("--fs", type=float, default=16000.0)
    pm.add_argument("--bearing", default="FIELD7")
    pm.add_argument("--rpm-window", dest="rpm_window", default=None,
                    help="标称转速窗 LO:HI（默认 1100:1200，LOSO 网格与 T6 频带随之）")
    pm.add_argument("--no-physical", action="store_true", dest="no_physical",
                    help="消融：剔除 T2 物理谐波突出度（无需转速的盲特征路线）")
    pm.add_argument("--no-comb", action="store_true", dest="no_comb",
                    help="消融：剔除 T6 盲梳检结构化特征")
    pm.add_argument("--seed", type=int, default=2026)
    pm.add_argument("--out", default="models", help="模型输出目录（默认 ./models）")
    pm.set_defaults(func=cmd_ml_train)

    pp = sub.add_parser("predict", help="用训练好的 ML 模型预测故障类别（3 类 + normal）")
    pp.add_argument("input", help="数据文件 (csv/txt/dat/npy/npz/mat)")
    pp.add_argument("--model", default="models/model.joblib", help="模型文件")
    pp.add_argument("--fs", type=float, help="采样率 Hz")
    pp.add_argument("--rpm", type=float, help="转速 rpm（T2 物理特征用；缺省则转速自估，回退 1150）")
    pp.add_argument("--shaft-freq", type=float, dest="shaft_freq",
                    help="轴频 Hz（与 --rpm 二选一，优先）")
    pp.add_argument("--bearing", default=None,
                    help="轴承型号（T6/T2 需要；缺省取模型包训练时记录值，通常 FIELD7）")
    pp.add_argument("--column", type=int, help="多列数据取第几列")
    pp.add_argument("--threshold", type=float, default=None,
                    help="两级架构 Stage1 异常判定阈值（默认用模型包内 0.5；"
                         "调低→更灵敏报警, 调高→更保守）")
    pp.add_argument("--compare", action="store_true",
                    help="同时运行 v2 DSP 盲梳检流水线并对比两法结论")
    pp.set_defaults(func=cmd_predict)
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

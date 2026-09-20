"""一键演示：仿真 → 诊断 → 出报告。

运行方式::

    python examples/run_demo.py            # 全部五种信号（含正常）
    python examples/run_demo.py BPFO BPFI  # 只演示指定故障

产物输出到 examples/demo_output/<故障类型>/ 下（PNG + TXT + JSON）。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 允许直接 python examples/run_demo.py 运行（无需安装）
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bearing_diag.diagnose import diagnose  # noqa: E402
from bearing_diag.report import write_reports  # noqa: E402
from bearing_diag.simulate import simulate  # noqa: E402

FAULTS = ["normal", "BPFO", "BPFI", "BSF", "FTF"]


def main(argv: list[str]) -> None:
    faults = [a.upper() for a in argv if a.upper() in FAULTS] or FAULTS
    out_root = Path(__file__).parent / "demo_output"
    for fault in faults:
        dur = 4.0 if fault == "FTF" else 2.0
        sig, info = simulate(
            fault, fs=12000.0, duration=dur, rpm=1797.0,
            bearing="SKF6205", snr_db=5.0, seed=2024,
        )
        result = diagnose(sig, fs=info["fs"], shaft_freq=info["shaft_freq"],
                          bearing="SKF6205")
        out_dir = out_root / fault.lower()
        paths = write_reports(result, out_dir, source_desc=f"simulate({fault})")
        ok = "✓" if result.verdict == fault else "✗"
        print(f"{ok} {fault:6s} → 判定 {result.verdict:6s} "
              f"(置信度 {result.confidence:.2f})  报告: {paths['png']}")
    print(f"\n全部报告位于: {out_root}")
    print("提示: 打开 PNG 人工核对包络谱上的谐波族是诊断流程的最后一步。")


if __name__ == "__main__":
    main(sys.argv[1:])

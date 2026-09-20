"""轴承几何参数与故障特征频率计算。

特征频率公式（接触角 alpha，滚动体数 n，节圆直径 D，滚动体直径 d，轴频 fr）：

    BPFO = (n/2) * fr * (1 - (d/D) * cos(alpha))     外圈故障（滚动体经过外圈缺陷）
    BPFI = (n/2) * fr * (1 + (d/D) * cos(alpha))     内圈故障（滚动体经过内圈缺陷）
    BSF  = (D/(2d)) * fr * (1 - ((d/D) * cos(alpha))**2)   滚动体自转频率（缺陷单侧碰拍减半为 2xBSF）
    FTF  = (fr/2) * (1 - (d/D) * cos(alpha))         保持架频率

公式推导见 docs/principles.md 第 3 章。注意实际轴承存在 1~2% 的滚动体滑移，
实测谱峰通常略低于理论值。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

FAULT_TYPES = ("BPFO", "BPFI", "BSF", "FTF")
FAULT_NAMES_CN = {
    "BPFO": "外圈故障",
    "BPFI": "内圈故障",
    "BSF": "滚动体故障",
    "FTF": "保持架故障",
}


@dataclass(frozen=True)
class Bearing:
    """滚动轴承几何参数。长度单位只需自洽（通常 mm）。"""

    name: str
    n_balls: int          # 滚动体数量
    pitch_diameter: float  # D: 节圆直径（滚动体中心所在圆直径）
    ball_diameter: float   # d: 滚动体直径
    contact_angle_deg: float = 0.0  # alpha: 接触角（度），深沟球轴承为 0

    def fault_frequencies(self, shaft_freq: float) -> dict[str, float]:
        """给定轴频 fr (Hz)，返回四种故障特征频率 (Hz)。"""
        import math

        if shaft_freq <= 0:
            raise ValueError("shaft_freq 必须为正数（单位 Hz）")
        c = math.cos(math.radians(self.contact_angle_deg))
        ratio = (self.ball_diameter / self.pitch_diameter) * c
        fr = shaft_freq
        return {
            "BPFO": self.n_balls / 2.0 * fr * (1.0 - ratio),
            "BPFI": self.n_balls / 2.0 * fr * (1.0 + ratio),
            "BSF": self.pitch_diameter / (2.0 * self.ball_diameter) * fr * (1.0 - ratio**2),
            "FTF": fr / 2.0 * (1.0 - ratio),
        }


# 内置常见轴承参数库（长度单位 mm）。
# SKF 6205 与 ER-16K 为凯斯西储大学（CWRU）轴承数据中心所用驱动端轴承，
# 便于用户自备 CWRU 数据后直接分析（见 loaders.py）。
BUILTIN_BEARINGS: dict[str, Bearing] = {
    "SKF6205": Bearing("SKF6205", n_balls=9, pitch_diameter=39.04, ball_diameter=7.94,
                       contact_angle_deg=0.0),
    "ER16K": Bearing("ER16K", n_balls=8, pitch_diameter=38.53, ball_diameter=7.94,
                     contact_angle_deg=0.0),
    "SKF6208": Bearing("SKF6208", n_balls=9, pitch_diameter=59.5, ball_diameter=11.9,
                       contact_angle_deg=0.0),
    "SKF22216": Bearing("SKF22216", n_balls=16, pitch_diameter=106.5, ball_diameter=14.0,
                        contact_angle_deg=10.0),
}


def get_bearing(name: str) -> Bearing:
    """按名称取内置轴承；支持 '6205'/'SKF6205' 等宽松写法。"""
    key = name.upper().replace("-", "").replace(" ", "")
    for k, v in BUILTIN_BEARINGS.items():
        if k in key or key in k:
            return v
    raise KeyError(
        f"未知轴承 '{name}'。内置：{', '.join(BUILTIN_BEARINGS)}；"
        "或用 --bearing-json 提供自定义参数文件"
    )


def load_bearing_json(path: str | Path) -> Bearing:
    """从 JSON 文件加载自定义轴承。

    文件格式：{"name": "...", "n_balls": 9, "pitch_diameter": 39.04,
              "ball_diameter": 7.94, "contact_angle_deg": 0.0}
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    fields = {"name", "n_balls", "pitch_diameter", "ball_diameter", "contact_angle_deg"}
    unknown = set(data) - fields
    if unknown:
        raise ValueError(f"轴承 JSON 含未知字段: {sorted(unknown)}")
    return Bearing(
        name=str(data["name"]),
        n_balls=int(data["n_balls"]),
        pitch_diameter=float(data["pitch_diameter"]),
        ball_diameter=float(data["ball_diameter"]),
        contact_angle_deg=float(data.get("contact_angle_deg", 0.0)),
    )


def bearing_to_dict(bearing: Bearing) -> dict:
    return asdict(bearing)

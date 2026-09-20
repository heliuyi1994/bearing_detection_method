"""滚动轴承几何参数与故障特征频率计算。

特征频率（defect frequencies）是滚动轴承故障诊断的理论基准：
滚道上的局部缺陷每被滚动体碾过一次就产生一次冲击，冲击的重复频率
完全由轴承几何尺寸与轴转频决定，与载荷、转速漂移无关（忽略滑差时）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Bearing:
    """滚动轴承几何参数。

    参数
    ----
    n : int
        滚动体个数。
    d : float
        滚动体直径 (mm)。
    pitch_d : float
        节径（滚动体中心圆直径）D (mm)。
    contact_angle : float
        接触角 α（度），深沟球轴承为 0。
    """

    n: int
    d: float
    pitch_d: float
    contact_angle: float = 0.0

    @classmethod
    def sk6205(cls) -> "Bearing":
        """SKF 6205-2RS 深沟球轴承（CWRU 数据集所用型号）。"""
        return cls(n=9, d=7.94, pitch_d=39.04, contact_angle=0.0)

    @property
    def geometry_ratio(self) -> float:
        """几何因子 (d/D)·cosα，四个特征频率公式共用的无量纲量。"""
        return (self.d / self.pitch_d) * math.cos(math.radians(self.contact_angle))

    def ftf(self, shaft_freq: float) -> float:
        """保持架故障频率 FTF = (1/2)·fr·(1 − (d/D)·cosα)。"""
        return 0.5 * shaft_freq * (1.0 - self.geometry_ratio)

    def bpfo(self, shaft_freq: float) -> float:
        """外圈故障频率 BPFO = (n/2)·fr·(1 − (d/D)·cosα)。"""
        return 0.5 * self.n * shaft_freq * (1.0 - self.geometry_ratio)

    def bpfi(self, shaft_freq: float) -> float:
        """内圈故障频率 BPFI = (n/2)·fr·(1 + (d/D)·cosα)。"""
        return 0.5 * self.n * shaft_freq * (1.0 + self.geometry_ratio)

    def bsf(self, shaft_freq: float) -> float:
        """滚动体故障频率 BSF = (D/2d)·fr·(1 − ((d/D)·cosα)²)。"""
        r = self.geometry_ratio
        return (self.pitch_d / (2.0 * self.d)) * shaft_freq * (1.0 - r * r)

    def characteristic_frequencies(self, shaft_freq: float) -> dict[str, float]:
        """返回四类故障的特征频率字典，键与仿真器/诊断模块的故障名一致。"""
        return {
            "cage": self.ftf(shaft_freq),
            "ball": self.bsf(shaft_freq),
            "inner": self.bpfi(shaft_freq),
            "outer": self.bpfo(shaft_freq),
        }

"""特征频率公式测试：SKF 6205 文献系数 + 公式内在恒等式。"""

import pytest

from bearing_fault import Bearing


class TestSKF6205:
    """CWRU 数据集轴承 SKF 6205-2RS：n=9, d=7.94mm, D=39.04mm, α=0°。

    文献中的阶次系数（特征频率/转频）：
    BPFO≈3.5848, BPFI≈5.4152, BSF≈2.3570, FTF≈0.3983。
    """

    b = Bearing.sk6205()
    fr = 29.95  # 1797 rpm

    def test_bpfo(self):
        assert self.b.bpfo(self.fr) == pytest.approx(3.5848 * self.fr, rel=1e-3)

    def test_bpfi(self):
        assert self.b.bpfi(self.fr) == pytest.approx(5.4152 * self.fr, rel=1e-3)

    def test_bsf(self):
        assert self.b.bsf(self.fr) == pytest.approx(2.3570 * self.fr, rel=1e-3)

    def test_ftf(self):
        assert self.b.ftf(self.fr) == pytest.approx(0.3983 * self.fr, rel=1e-3)


class TestIdentities:
    """特征频率公式的代数恒等式，对任意几何参数成立。"""

    @pytest.mark.parametrize("n,d,D,alpha", [
        (9, 7.94, 39.04, 0.0),
        (8, 12.0, 60.0, 15.0),
        (12, 6.0, 40.0, 25.0),
    ])
    def test_bpfo_plus_bpfi(self, n, d, D, alpha):
        """BPFO + BPFI = n·fr（每个滚动体通过内、外圈各计一次）。"""
        b = Bearing(n=n, d=d, pitch_d=D, contact_angle=alpha)
        fr = 25.0
        assert b.bpfo(fr) + b.bpfi(fr) == pytest.approx(n * fr, rel=1e-12)

    def test_characteristic_frequencies_keys(self):
        b = Bearing.sk6205()
        freqs = b.characteristic_frequencies(30.0)
        assert set(freqs) == {"cage", "ball", "inner", "outer"}
        assert freqs["cage"] < freqs["ball"] < freqs["outer"] < freqs["inner"]

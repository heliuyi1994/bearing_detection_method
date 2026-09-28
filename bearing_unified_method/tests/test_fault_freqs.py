"""fault_freqs 单元测试：CWRU 官方公布值对照。"""

import json

import pytest

from bearing_unified.fault_freqs import (
    BUILTIN_BEARINGS,
    Bearing,
    get_bearing,
    load_bearing_json,
)

# CWRU 官方 SKF 6205-2RS JEM 驱动端轴承 @ 1797 rpm 公布值（Hz）
CWRU_1797 = {"BPFO": 107.36, "BPFI": 162.19, "BSF": 70.59, "FTF": 11.92}


class TestFaultFrequencies:
    def test_skf6205_matches_cwru(self):
        ff = BUILTIN_BEARINGS["SKF6205"].fault_frequencies(1797 / 60)
        for k, expect in CWRU_1797.items():
            assert ff[k] == pytest.approx(expect, abs=0.06), k

    def test_bpfo_equals_n_times_ftf(self):
        # 恒等关系 BPFO = n·FTF（文档第 3 章）
        for b in BUILTIN_BEARINGS.values():
            ff = b.fault_frequencies(25.0)
            assert ff["BPFO"] == pytest.approx(b.n_balls * ff["FTF"], rel=1e-12)

    def test_bpfi_bpfo_ordering(self):
        # 内圈频率恒高于外圈（cos 项符号）
        for b in BUILTIN_BEARINGS.values():
            ff = b.fault_frequencies(30.0)
            assert ff["BPFI"] > ff["BPFO"] > ff["FTF"]
            assert ff["BPFO"] > ff["BSF"] * 0  # 仅验证键存在

    def test_invalid_shaft_freq(self):
        with pytest.raises(ValueError):
            BUILTIN_BEARINGS["SKF6205"].fault_frequencies(0)

    def test_get_bearing_loose_name(self):
        assert get_bearing("6205") is BUILTIN_BEARINGS["SKF6205"]
        assert get_bearing("skf-6205").name == "SKF6205"
        with pytest.raises(KeyError):
            get_bearing("NOPE999")

    def test_bearing_json_roundtrip(self, tmp_path):
        data = {"name": "T1", "n_balls": 10, "pitch_diameter": 40.0,
                "ball_diameter": 8.0, "contact_angle_deg": 15.0}
        p = tmp_path / "b.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        b = load_bearing_json(p)
        assert b.name == "T1" and b.n_balls == 10
        ff = b.fault_frequencies(16.0)
        assert ff["BPFO"] > 0 and ff["FTF"] > 0
        # 未知字段应报错
        p.write_text(json.dumps({**data, "bad": 1}), encoding="utf-8")
        with pytest.raises(ValueError):
            load_bearing_json(p)

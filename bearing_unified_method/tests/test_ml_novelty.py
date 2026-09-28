# -*- coding: utf-8 -*-
"""Stage1 新奇检测后端（NoveltyDetector）单元测试。

覆盖：只训健康样本的协议要求、异常分排序、与监督后端同接口（predict_proba/
classes_/make_estimator/统一评估协议）、IF 后端可构造但默认 EE。
"""

import json

import numpy as np
import pytest

from bearing_unified.ml.dataset import make_sim_dataset
from bearing_unified.ml.evaluate import hierarchical_cv_report, loso_evaluate_two_stage
from bearing_unified.ml.train import (
    ABNORMAL,
    NORMAL,
    NoveltyDetector,
    make_estimator,
    train_pipeline_two_stage,
)
from bearing_unified.simulate import simulate


@pytest.fixture(scope="module")
def dataset():
    return make_sim_dataset(n_base_per_class=4, windows_per_base=4,
                            window_s=0.8, seed=777)


@pytest.fixture(scope="module")
def novelty_trained(dataset):
    m1, m2, report = train_pipeline_two_stage(dataset, verbose=False,
                                              stage1="novelty")
    return m1, m2, report


class TestNoveltyDetector:
    def test_fit_uses_healthy_only(self, dataset):
        """协议要求：fit 只取 y==normal 样本拟合（故障标签不参与）。"""
        det = NoveltyDetector(feature_names=dataset.feature_names_)
        y1 = np.where(dataset.y == NORMAL, NORMAL, ABNORMAL)
        det.fit(dataset.X, y1)
        # 检测子集 4 维（小样本下 EE 稳健），classes_ 按 y 的唯一值排序
        assert det.pipe_.named_steps["det"].n_features_in_ == len(det.DEFAULT_SUBSET)
        assert set(det.classes_) == {NORMAL, ABNORMAL}

    def test_fault_scores_higher(self, dataset):
        """仿真故障样本的异常分应显著高于健康样本。"""
        det = NoveltyDetector(feature_names=dataset.feature_names_)
        y1 = np.where(dataset.y == NORMAL, NORMAL, ABNORMAL)
        det.fit(dataset.X, y1)
        scores = det.decision_function(dataset.X)
        s_norm = scores[dataset.y == NORMAL]
        s_fault = scores[dataset.y != NORMAL]
        assert np.median(s_fault) > np.median(s_norm)
        # 伪概率接口：故障的 p_abnormal 中位数应 > 0.5
        # （列位置按 classes_ 中 abnormal 的索引取，勿硬编码列号）
        i_abn = int(np.where(np.asarray(det.classes_) == ABNORMAL)[0][0])
        p_abn = det.predict_proba(dataset.X)[:, i_abn]
        assert np.median(p_abn[dataset.y != NORMAL]) > 0.5
        assert np.median(p_abn[dataset.y == NORMAL]) < 0.5

    def test_same_interface_as_supervised(self, dataset):
        """与监督后端同接口：predict_proba 形状/classes_/clone 兼容。"""
        det = make_estimator("novelty:elliptic_envelope", {},
                             feature_names=dataset.feature_names_)
        y1 = np.where(dataset.y == NORMAL, NORMAL, ABNORMAL)
        det.fit(dataset.X, y1)
        p = det.predict_proba(dataset.X[:5])
        assert p.shape == (5, 2)
        assert np.allclose(p.sum(axis=1), 1.0)
        pred = det.predict(dataset.X[:5])
        assert set(pred) <= {NORMAL, ABNORMAL}

    def test_isolation_forest_backend_constructible(self, dataset):
        """IF 后端仅对照保留（有检出率塌缩教训，见 train.py docstring），
        默认可构造但不做默认。"""
        det = NoveltyDetector(backend="isolation_forest",
                              feature_names=dataset.feature_names_)
        y1 = np.where(dataset.y == NORMAL, NORMAL, ABNORMAL)
        det.fit(dataset.X, y1)
        assert det.predict_proba(dataset.X[:3]).shape == (3, 2)

    def test_too_few_healthy_raises(self, dataset):
        det = NoveltyDetector(feature_names=dataset.feature_names_)
        X = np.vstack([dataset.X[dataset.y == NORMAL][:3],
                       dataset.X[dataset.y != NORMAL][:3]])
        y = np.array([NORMAL] * 3 + [ABNORMAL] * 3)
        with pytest.raises(ValueError, match="健康样本过少"):
            det.fit(X, y)

    def test_unknown_backend_raises(self):
        det = NoveltyDetector(backend="nope")
        with pytest.raises(ValueError, match="未知新奇检测后端"):
            det.fit(np.random.default_rng(0).normal(size=(10, 4)),
                    np.array([NORMAL] * 10))


class TestNoveltyTwoStageIntegration:
    def test_stage_structure(self, novelty_trained):
        m1, m2, report = novelty_trained
        assert report["stage1"]["best_model"] == "novelty:elliptic_envelope"
        assert set(m1.classes_) == {NORMAL, ABNORMAL}
        assert set(m2.classes_) == {"BPFO", "BPFI", "BSF"}

    def test_hierarchical_cv_with_novelty(self, dataset, novelty_trained):
        """新奇检测后端进入统一评估协议（GroupKFold + 工作点扫描）。"""
        _, _, report = novelty_trained
        s1 = {"model": report["stage1"]["best_model"],
              "best_params": report["stage1"]["best_params"]}
        s2 = {"model": report["stage2"]["best_model"],
              "best_params": report["stage2"]["best_params"]}
        hier = hierarchical_cv_report(dataset, s1, s2)
        assert set(hier["stage1"]) == {"0.30", "0.50", "0.70"}
        # 仿真数据 + 冲击特征可分时，EE 后端漏报率应很低
        assert hier["stage1"]["0.50"]["miss_rate"] <= 0.05
        assert hier["stage2"]["accuracy"] >= 0.9

    def test_loso_with_novelty(self, dataset, novelty_trained):
        _, _, report = novelty_trained
        s1 = {"model": report["stage1"]["best_model"],
              "best_params": report["stage1"]["best_params"]}
        s2 = {"model": report["stage2"]["best_model"],
              "best_params": report["stage2"]["best_params"]}
        out = loso_evaluate_two_stage(dataset, s1, s2, verbose=False)
        assert len(out["per_speed"]) == 4
        # 小规模数据集上新奇检测后端的端到端下限（监督后端同数据集 ≥0.9，
        # 见 test_ml_train；新奇检测的价值在无故障标注场景而非此指标）
        assert out["summary"]["mean_end_to_end"] >= 0.75


class TestCLI:
    def test_ml_train_and_predict(self, tmp_path, capsys, monkeypatch):
        from bearing_unified.cli import main as cli_main

        monkeypatch.chdir(tmp_path)
        rc = cli_main(["ml-train", "--n-base", "3", "--windows", "3",
                       "--window-s", "0.6", "--seed", "9",
                       "--out", "mdl"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "分类器对比" in out and "LOSO" in out
        assert "两级架构分级评估" in out and "阈值工作点" in out
        assert (tmp_path / "mdl" / "model.joblib").exists()
        report = json.loads((tmp_path / "mdl" / "training_report.json")
                            .read_text(encoding="utf-8"))
        assert report["arch"] == "two_stage"
        assert set(report["hierarchical_cv"]["stage1"]) == {"0.30", "0.50", "0.70"}
        assert report["loso"]["summary"]["mean_end_to_end"] >= 0.9

        x, info = simulate("BPFI", fs=16000, duration=2.0, rpm=1150,
                           bearing="FIELD7", snr_db=8, resonance_hz=2200, seed=41)
        from bearing_unified.loaders import save_signal_csv
        f = tmp_path / "q.csv"
        save_signal_csv(f, x, 16000)
        rc = cli_main(["predict", str(f), "--fs", "16000",
                       "--model", "mdl/model.joblib", "--rpm", "1150",
                       "--compare"])
        assert rc == 0
        out2 = capsys.readouterr().out
        assert "判定类别 : BPFI" in out2
        assert "v2 DSP" in out2  # --compare 输出含 v2 DSP 对照

    def test_ml_train_novelty_backend(self, tmp_path, capsys, monkeypatch):
        from bearing_unified.cli import main as cli_main

        monkeypatch.chdir(tmp_path)
        rc = cli_main(["ml-train", "--stage1", "novelty",
                       "--n-base", "3", "--windows", "3",
                       "--window-s", "0.6", "--seed", "9",
                       "--out", "mdl_nov"])
        assert rc == 0
        report = json.loads((tmp_path / "mdl_nov" / "training_report.json")
                            .read_text(encoding="utf-8"))
        assert report["stage1"]["best_model"] == "novelty:elliptic_envelope"
        assert "只训健康样本" in capsys.readouterr().out

    def test_ml_train_flat_arch(self, tmp_path, capsys, monkeypatch):
        from bearing_unified.cli import main as cli_main

        monkeypatch.chdir(tmp_path)
        rc = cli_main(["ml-train", "--arch", "flat", "--n-base", "3",
                       "--windows", "3", "--window-s", "0.6", "--seed", "9",
                       "--out", "mdl_flat"])
        assert rc == 0
        report = json.loads((tmp_path / "mdl_flat" / "training_report.json")
                            .read_text(encoding="utf-8"))
        assert report["arch"] == "flat"
        assert report["cv_macro_f1"] >= 0.9

    def test_ml_train_no_comb(self, tmp_path, capsys, monkeypatch):
        from bearing_unified.cli import main as cli_main

        monkeypatch.chdir(tmp_path)
        rc = cli_main(["ml-train", "--no-comb", "--n-base", "3",
                       "--windows", "3", "--window-s", "0.6", "--seed", "9",
                       "--out", "mdl_nc"])
        assert rc == 0
        import joblib
        bundle = joblib.load(tmp_path / "mdl_nc" / "model.joblib")
        assert bundle["use_comb"] is False
        assert len(bundle["feature_names"]) == 43

    def test_ml_dataset_then_train(self, tmp_path, capsys, monkeypatch):
        from bearing_unified.cli import main as cli_main

        monkeypatch.chdir(tmp_path)
        rc = cli_main(["ml-dataset", "--n-base", "3", "--windows", "3",
                       "--window-s", "0.6", "--seed", "5", "--out", "data/ds.npz"])
        assert rc == 0
        assert (tmp_path / "data" / "ds.npz").exists()
        rc = cli_main(["ml-train", "--dataset", "data/ds.npz", "--out", "mdl_ds"])
        assert rc == 0
        assert (tmp_path / "mdl_ds" / "model.joblib").exists()

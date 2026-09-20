"""ML 子系统测试：特征提取、数据集 round-trip、训练冒烟、推理正确性、CLI。

为控制测试时长，均使用缩小网格（少样本、短信号）；准确率类断言只在
易分条件下做功能性验证，完整网格的准确率基准由 CLI 端到端验证承担。
"""

import json
from pathlib import Path

import numpy as np
import pytest

from bearing_fault import BearingSimulator
from bearing_fault.ml import (
    CLASS_NAMES,
    FEATURE_NAMES,
    DatasetConfig,
    FeatureExtractor,
    TwoStageModel,
    evaluate,
    generate,
    load_npz,
    save_npz,
    train_two_stage,
)

FS = 12000.0


@pytest.fixture(scope="module")
def small_dataset():
    """5 类 × 1 转速 × 2 SNR × 8 种子 = 80 样本的小数据集。

    健康样本 16 个：IsolationForest 检测级需要数十个健康样本标定阈值，
    样本过少（<10）时对极端点的判定不可靠（实测教训）。
    """
    cfg = DatasetConfig(duration=1.5, rpms=(1797.0,), snrs=(0.0, 6.0), n_seeds=8)
    return generate(cfg)


@pytest.fixture(scope="module")
def trained(small_dataset):
    return train_two_stage(small_dataset, test_size=0.3, seed=42)


class TestFeatureExtractor:
    def test_dimension_and_names(self):
        sim = BearingSimulator(fs=FS, duration=1.0, rpm=1797, seed=0)
        feats = FeatureExtractor(fs=FS).extract(sim.signal("outer"), 1797)
        assert len(FEATURE_NAMES) == 26
        assert set(feats) == set(FEATURE_NAMES)
        assert all(np.isfinite(list(feats.values())))

    def test_fault_signal_has_stronger_envelope_features(self):
        sim = BearingSimulator(fs=FS, duration=2.0, rpm=1797, seed=1)
        ex = FeatureExtractor(fs=FS)
        f_healthy = ex.extract(sim.signal("healthy"), 1797)
        f_outer = ex.extract(sim.signal("outer", snr_db=0.0), 1797)
        assert f_outer["env_outer_h1"] > 10 * max(f_healthy["env_outer_h1"], 1e-9)
        assert f_outer["env_max_ratio"] > f_healthy["env_max_ratio"]


class TestDataset:
    def test_shape_and_meta(self, small_dataset):
        assert small_dataset.X.shape == (80, 26)
        assert small_dataset.y.shape == (80,)
        assert set(np.unique(small_dataset.y)) == set(range(5))
        assert len(small_dataset.meta) == 80

    def test_save_load_roundtrip(self, small_dataset, tmp_path: Path):
        path = save_npz(tmp_path / "ds.npz", small_dataset)
        loaded = load_npz(path)
        np.testing.assert_array_equal(loaded.X, small_dataset.X)
        np.testing.assert_array_equal(loaded.y, small_dataset.y)
        np.testing.assert_array_equal(loaded.meta, small_dataset.meta)
        assert loaded.feature_names == small_dataset.feature_names
        assert loaded.fs == small_dataset.fs

    def test_grid_size(self):
        assert DatasetConfig().grid_size() == 360


class TestTraining:
    def test_cv_scores_recorded(self, trained):
        assert set(trained.cv_scores) == {"svm_rbf", "random_forest", "knn"}
        assert trained.best_stage2 in trained.cv_scores
        assert trained.cv_scores[trained.best_stage2] == max(
            trained.cv_scores.values()
        )

    def test_cascade_metrics_structure(self, trained):
        metrics = evaluate(trained)
        assert 0.0 <= metrics["accuracy"] <= 1.0
        det = metrics["detection"]
        assert 0.0 <= det["false_alarm_rate"] <= 1.0
        assert 0.0 <= det["detection_rate"] <= 1.0
        assert len(metrics["confusion_matrix"]) == len(CLASS_NAMES)

    def test_evaluate_writes_files(self, trained, tmp_path: Path):
        evaluate(trained, out_dir=tmp_path)
        assert (tmp_path / "metrics.json").exists()
        assert (tmp_path / "confusion_matrix.png").exists()
        metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
        assert "accuracy" in metrics


class TestPredict:
    def test_easy_cases(self, trained):
        """易分条件（高 SNR、训练覆盖的转速与时长）下推理应正确。"""
        sim = BearingSimulator(fs=FS, duration=1.5, rpm=1797, seed=999)
        model = trained.model
        assert model.predict_one(sim.signal("healthy"), 1797)["verdict"] == "healthy"
        assert model.predict_one(sim.signal("outer", snr_db=6.0), 1797)["verdict"] == "outer"

    def test_elliptic_envelope_stage1(self, small_dataset):
        """默认检测级 EllipticEnvelope 应检出故障并正确分类。"""
        result = train_two_stage(small_dataset, test_size=0.3, seed=42,
                                 stage1="elliptic_envelope")
        sim = BearingSimulator(fs=FS, duration=1.5, rpm=1797, seed=999)
        res = result.model.predict_one(sim.signal("outer", snr_db=6.0), 1797)
        assert res["anomaly"] is True
        assert res["verdict"] == "outer"

    def test_isolation_forest_stage1(self, small_dataset):
        """备选检测级 IsolationForest 在本夹具（16 健康样本）下可用。"""
        result = train_two_stage(small_dataset, test_size=0.3, seed=42,
                                 stage1="isolation_forest")
        sim = BearingSimulator(fs=FS, duration=1.5, rpm=1797, seed=999)
        res = result.model.predict_one(sim.signal("outer", snr_db=6.0), 1797)
        assert res["verdict"] == "outer"

    def test_unknown_stage1_raises(self, small_dataset):
        with pytest.raises(ValueError):
            train_two_stage(small_dataset, stage1="xgboost")

    def test_save_load_roundtrip(self, trained, tmp_path: Path):
        path = trained.model.save(tmp_path / "m.joblib")
        loaded = TwoStageModel.load(path)
        X = np.random.default_rng(0).normal(size=(3, 26))
        np.testing.assert_array_equal(
            loaded.predict_batch(X), trained.model.predict_batch(X)
        )


class TestCli:
    def test_ml_pipeline_end_to_end(self, tmp_path: Path):
        from bearing_fault.cli import main

        ds_path = tmp_path / "ds.npz"
        rc = main(["ml-dataset", "--out", str(ds_path), "--duration", "1.0",
                   "--seeds", "2"])
        assert rc == 0 and ds_path.exists()

        model_dir = tmp_path / "models"
        rc = main(["ml-train", "--dataset", str(ds_path), "--out", str(model_dir)])
        assert rc == 0
        assert (model_dir / "two_stage.joblib").exists()
        assert (model_dir / "metrics.json").exists()
        assert (model_dir / "confusion_matrix.png").exists()

        rc = main(["ml-predict", "--model", str(model_dir / "two_stage.joblib"),
                   "--fault", "outer", "--snr", "6"])
        assert rc == 0

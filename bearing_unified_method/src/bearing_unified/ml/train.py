"""训练管线：分类器对比选优 + 小网格调参 + 模型序列化。

支持两种架构（docs/principles.md 第 11.3 节）：

- flat（单级）：一个 4 分类器直接输出 normal/BPFO/BPFI/BSF（统一包 3 类故障）；
- two_stage（两级，默认落地架构）：
    Stage 1 检测器  normal vs abnormal 二分类（报警决策，阈值可调）
    Stage 2 分类器  三类故障分类（仅故障样本训练，输出条件概率）

Stage 1 两种后端：
- supervised（默认）：glm 监督式二分类，四分类器 GroupKFold 对比选优 +
  小网格 GridSearch，SVM 经概率校准；
- novelty：kimi 式新奇检测（NoveltyDetector/EllipticEnvelope，只用健康样本
  拟合）——覆盖"现场无故障标注"场景（合并方案移植项 3）。

流程：每级独立执行"四分类器 GroupKFold 对比（防泄漏）→ macro-F1 选优 →
小网格 GridSearch → 全量重拟合"。所有分类器支持 predict_proba
（LinearSVM/RBF-SVM 经 CalibratedClassifierCV 校准）。
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    IsolationForest,
    RandomForestClassifier,
)
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

MODEL_FILENAME = "model.joblib"
NORMAL = "normal"
ABNORMAL = "abnormal"
THRESHOLD_DEFAULT = 0.5


class NoveltyDetector(BaseEstimator, ClassifierMixin):
    """Stage 1 新奇检测后端（kimi 移植）：只用健康样本拟合的异常检测器。

    接口与监督后端一致（fit/predict/predict_proba/classes_），因此可直接
    进入统一评估协议（GroupKFold/LOSO/阈值工作点）——fit(X, y) 内部只取
    y==normal 的样本拟合检测器，各折训练集的健康样本之外的信息（含故障
    标签）一律不用，满足"只训健康样本"的协议要求。

    后端选择（kimi train.py 注释记录的实测教训）：IsolationForest 的分裂界
    由训练数据 min/max 决定、不随极端程度外推，健康样本数或网格配置稍变
    即可能把全部故障判为正常（**检出率塌缩为 0**）；EllipticEnvelope 基于
    稳健马氏距离，同批实验全配置检出率 1.0。故默认且推荐
    elliptic_envelope；isolation_forest 仅作对照保留、默认不用。

    特征子集：kimi 的又一教训——全特征维下健康样本有近零方差方向，标准化
    后高维稀疏、小样本协方差病态（EE 的 MCD 要求 n≫p），检测器在检测级应
    只用少量"冲击性"特征。默认 4 维（t1_kurtosis / t1_crest_factor /
    t1_clearance_factor / t5_env_peak_prom），与统一包 T1/T5 组同源。

    predict_proba 为启发式伪概率：以健康样本 decision_function 的中位数与
    标准差做 z 化后过 logistic（不是统计意义上的校准概率；阈值工作点在此
    单调轴上仍然可操作）。
    """

    #: 检测级冲击性特征子集（小样本下 EE 稳健的维度上限，见 docstring 教训）
    DEFAULT_SUBSET = ("t1_kurtosis", "t1_crest_factor",
                      "t1_clearance_factor", "t5_env_peak_prom")

    def __init__(self, backend: str = "elliptic_envelope",
                 contamination: float = 0.01, random_state: int = 0,
                 feature_names: list[str] | None = None,
                 feature_subset: tuple[str, ...] = DEFAULT_SUBSET) -> None:
        self.backend = backend
        self.contamination = contamination
        self.random_state = random_state
        self.feature_names = feature_names
        self.feature_subset = feature_subset

    def _make_base(self):
        if self.backend == "isolation_forest":
            return IsolationForest(n_estimators=200,
                                   contamination=self.contamination,
                                   random_state=self.random_state)
        if self.backend == "elliptic_envelope":
            return EllipticEnvelope(contamination=self.contamination,
                                    random_state=self.random_state)
        raise ValueError(f"未知新奇检测后端 {self.backend!r}")

    def _cols(self) -> np.ndarray:
        if self.feature_names is None:
            return np.arange(len(self.feature_subset))
        idx = [i for i, n in enumerate(self.feature_names) if n in self.feature_subset]
        if len(idx) != len(self.feature_subset):
            missing = set(self.feature_subset) - {self.feature_names[i] for i in idx}
            raise ValueError(f"特征名列表缺少检测子集字段: {sorted(missing)}")
        return np.asarray(idx)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        uniq = np.unique(y)
        if (uniq == NORMAL).any():
            normal_label, abn_label = NORMAL, ABNORMAL
        elif set(uniq.tolist()) == {0, 1}:
            # sklearn cross_val_predict 的 LabelEncoder 路径：sorted({abnormal,
            # normal}) → abnormal=0、normal=1（确定性映射，见 evaluate 调用处）
            normal_label, abn_label = 1, 0
        else:
            raise ValueError(
                f"无法从标签 {uniq.tolist()} 识别健康类（期望 normal/abnormal 或其编码）")
        self.classes_ = uniq
        self._normal_label, self._abn_label = normal_label, abn_label
        Xh = X[y == normal_label][:, self._cols()]
        if Xh.shape[0] < 5:
            raise ValueError(f"健康样本过少（{Xh.shape[0]} < 5），无法拟合新奇检测器")
        self.pipe_ = Pipeline([("scaler", StandardScaler()), ("det", self._make_base())])
        self.pipe_.fit(Xh)
        df = self.pipe_.decision_function(Xh)
        self.df_mu_ = float(np.median(df))
        self.df_sd_ = float(np.std(df)) or 1.0
        return self

    def decision_function(self, X) -> np.ndarray:
        """异常分（越大越异常）。"""
        X = np.asarray(X, dtype=float)
        return -self.pipe_.decision_function(X[:, self._cols()])

    def predict_proba(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        z = (self.pipe_.decision_function(X[:, self._cols()]) - self.df_mu_) / self.df_sd_
        p_abn = 1.0 / (1.0 + np.exp(z))
        out = np.empty((X.shape[0], len(self.classes_)))
        for i, c in enumerate(self.classes_):
            out[:, i] = p_abn if c == self._abn_label else 1.0 - p_abn
        return out

    def predict(self, X) -> np.ndarray:
        p = self.predict_proba(X)
        return self.classes_[np.argmax(p, axis=1)]


def make_novelty_detector(backend: str = "elliptic_envelope",
                          feature_names: list[str] | None = None,
                          contamination: float = 0.01,
                          random_state: int = 0) -> NoveltyDetector:
    """构造 Stage 1 新奇检测器（详见 NoveltyDetector docstring 的教训记录）。"""
    return NoveltyDetector(backend=backend, contamination=contamination,
                           random_state=random_state, feature_names=feature_names)


def build_model_zoo(random_state: int = 0) -> dict[str, dict]:
    """候选分类器及其小网格。返回 {名称: {"estimator", "param_grid"}}。"""
    scaler = StandardScaler()
    return {
        "LinearSVM": {
            "estimator": Pipeline([
                ("scaler", scaler),
                ("clf", CalibratedClassifierCV(
                    LinearSVC(C=1.0, dual="auto", max_iter=20000),
                    method="sigmoid", cv=3)),
            ]),
            "param_grid": {"clf__estimator__C": [0.1, 1.0, 10.0]},
        },
        "RBF-SVM": {
            "estimator": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", CalibratedClassifierCV(
                    SVC(C=10.0, gamma="scale", random_state=random_state),
                    method="sigmoid", cv=3, ensemble=False)),
            ]),
            "param_grid": {"clf__estimator__C": [1.0, 10.0, 100.0],
                           "clf__estimator__gamma": ["scale", 0.01]},
        },
        "RandomForest": {
            "estimator": RandomForestClassifier(
                n_estimators=300, random_state=random_state, n_jobs=-1),
            "param_grid": {"max_depth": [None, 12]},
        },
        "HistGB": {
            "estimator": HistGradientBoostingClassifier(
                max_iter=200, random_state=random_state),
            "param_grid": {"max_iter": [100, 200]},
        },
    }


def compare_classifiers(dataset, n_splits: int = 5) -> list[dict]:
    """四分类器 GroupKFold 对比。返回按 macro-F1 降序的结果表。"""
    from sklearn.metrics import accuracy_score, f1_score

    if dataset.n_samples < n_splits * 4:
        n_splits = max(2, dataset.n_samples // 8)
    cv = GroupKFold(n_splits=n_splits)
    rows: list[dict] = []
    for name, spec in build_model_zoo().items():
        est = spec["estimator"]
        pred = cross_val_predict(est, dataset.X, dataset.y,
                                 groups=dataset.groups, cv=cv)
        rows.append({
            "model": name,
            "accuracy": float(accuracy_score(dataset.y, pred)),
            "macro_f1": float(f1_score(dataset.y, pred, average="macro")),
        })
    rows.sort(key=lambda r: r["macro_f1"], reverse=True)
    return rows


def tune_and_fit(dataset, model_name: str, random_state: int = 0, n_splits: int = 5):
    """对指定分类器做小网格搜索并返回最优拟合（refit 于全量数据）。"""
    spec = build_model_zoo(random_state)[model_name]
    cv = GroupKFold(n_splits=min(n_splits, max(2, len(np.unique(dataset.groups)) // 4)))
    gs = GridSearchCV(
        spec["estimator"], spec["param_grid"], scoring="f1_macro",
        cv=cv, refit=True, n_jobs=-1,
    )
    gs.fit(dataset.X, dataset.y, groups=dataset.groups)
    return gs


def train_pipeline(dataset, random_state: int = 0, verbose: bool = True):
    """完整训练：对比 → 选优 → 调参 → 重拟合。返回 (model, report)。"""
    cmp_rows = compare_classifiers(dataset)
    best_name = cmp_rows[0]["model"]
    if verbose:
        print("分类器对比（GroupKFold 防泄漏交叉验证）：")
        for r in cmp_rows:
            print(f"  {r['model']:12s} acc={r['accuracy']:.4f}  macroF1={r['macro_f1']:.4f}")
        print(f"选优: {best_name}，进入小网格调参…")
    gs = tune_and_fit(dataset, best_name, random_state)
    if verbose:
        print(f"最优超参: {gs.best_params_}  CV macroF1={gs.best_score_:.4f}")
    report = {
        "comparison": cmp_rows,
        "best_model": best_name,
        "best_params": {k: v for k, v in gs.best_params_.items()},
        "cv_macro_f1": float(gs.best_score_),
    }
    return gs.best_estimator_, report


def make_estimator(name: str, params: dict | None = None, random_state: int = 0,
                   feature_names: list[str] | None = None):
    """按名称与超参重建未拟合的估计器（供评估端做交叉验证，避免用全量
    重拟合模型评自己训练的数据）。

    name 为 "novelty:<backend>" 时重建新奇检测器（feature_names 用于定位
    检测子集列）；否则走监督 zoo 路径。
    """
    if name.startswith("novelty:"):
        p = dict(params or {})
        backend = p.pop("backend", name.split(":", 1)[1])
        return NoveltyDetector(backend=backend, random_state=random_state,
                               feature_names=feature_names, **p)
    est = clone(build_model_zoo(random_state)[name]["estimator"])
    if params:
        est.set_params(**params)
    return est


def train_pipeline_two_stage(dataset, random_state: int = 0, verbose: bool = True,
                             stage1: str = "supervised",
                             novelty_backend: str = "elliptic_envelope"):
    """两级层次分类训练。

    Stage 1: normal vs abnormal。supervised（默认）= 监督 zoo 对比选优；
        novelty = 新奇检测（只训健康样本，backend 见 NoveltyDetector）。
    Stage 2: 三类故障（仅故障样本，监督 zoo 对比选优）。

    返回 (stage1_model, stage2_model, report)，report 含各级选型与最优超参，
    供评估端用 make_estimator 重建未拟合估计器做诚实交叉验证。
    """
    y1 = np.where(dataset.y == NORMAL, NORMAL, ABNORMAL)
    from dataclasses import replace

    ds1 = replace(dataset, y=y1)
    if stage1 == "novelty":
        if verbose:
            print(f"── Stage 1 检测器（新奇检测 {novelty_backend}，只训健康样本）──")
        m1 = make_novelty_detector(backend=novelty_backend,
                                   feature_names=dataset.feature_names_,
                                   random_state=random_state)
        m1.fit(dataset.X, y1)
        r1 = {"best_model": f"novelty:{novelty_backend}", "best_params": {},
              "cv_macro_f1": None,
              "note": "新奇检测：仅用健康样本拟合（合并方案移植项 3）"}
        if verbose:
            print("  已拟合（协议评估见分级交叉验证与 LOSO）")
    elif stage1 == "supervised":
        if verbose:
            print("── Stage 1 检测器（normal vs abnormal，监督式）──")
        m1, r1 = train_pipeline(ds1, random_state, verbose)
    else:
        raise ValueError(f"未知 stage1 后端 {stage1!r}（可选 supervised|novelty）")
    if verbose:
        print("── Stage 2 分类器（三类故障）──")
    ds2 = dataset.subset(dataset.y != NORMAL)
    m2, r2 = train_pipeline(ds2, random_state, verbose)
    report = {"stage1": r1, "stage2": r2, "arch": "two_stage"}
    return m1, m2, report


def save_model(
    path: str | Path,
    model=None,
    *,
    feature_names_: list[str] | None = None,
    classes: list[str] | None = None,
    use_physical: bool = True,
    use_comb: bool = True,
    config: dict | None = None,
    metrics: dict | None = None,
    stage1=None,
    stage2=None,
    stage1_spec: dict | None = None,
    stage2_spec: dict | None = None,
    threshold: float = THRESHOLD_DEFAULT,
) -> Path:
    """序列化模型包。

    单级：model；两级：stage1/stage2（模型）+ stage*_spec（名称与超参，供评估
    端重建）+ threshold。两类包都带 arch 字段，推理端按 arch 分派。
    """
    if model is None and stage1 is None:
        raise ValueError("save_model 需要 model（单级）或 stage1/stage2（两级）")
    if feature_names_ is None or classes is None:
        raise ValueError("feature_names_ 与 classes 为必填")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "arch": "two_stage" if stage1 is not None else "flat",
        "model": model,
        "stage1": {"model": stage1, "spec": stage1_spec} if stage1 is not None else None,
        "stage2": {"model": stage2, "spec": stage2_spec} if stage2 is not None else None,
        "threshold": float(threshold),
        "feature_names": feature_names_,
        "classes": classes,
        "use_physical": use_physical,
        "use_comb": use_comb,
        "config": config or {},
        "metrics": metrics or {},
        "format_version": 2,
    }
    joblib.dump(bundle, p)
    return p


def load_model(path: str | Path) -> dict:
    """加载模型包并做基本校验（旧版无 arch 字段的包按 flat 兼容）。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"模型文件不存在: {p}")
    bundle = joblib.load(p)
    if bundle.get("arch") == "two_stage":
        for stage_key in ("stage1", "stage2"):
            stage = bundle.get(stage_key)
            if not stage or "model" not in stage:
                raise ValueError(f"两级模型包缺少 {stage_key}.model（文件损坏？）")
    elif "model" not in bundle:
        raise ValueError("模型包缺少 model 字段（文件可能损坏或版本不兼容）")
    for key in ("feature_names", "classes", "use_physical"):
        if key not in bundle:
            raise ValueError(f"模型包缺少字段 {key}（文件可能损坏或版本不兼容）")
    bundle.setdefault("use_comb", True)
    return bundle

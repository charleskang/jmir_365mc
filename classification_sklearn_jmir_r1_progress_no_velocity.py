
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import platform
import re
import shutil
import sys
import tempfile
import time
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize, brentq
from scipy.special import expit, logit
from scipy.stats import spearmanr, kruskal
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix,
    precision_recall_curve, roc_auc_score, roc_curve,
)
from sklearn.model_selection import (
    LeaveOneGroupOut, StratifiedKFold, StratifiedGroupKFold, train_test_split,
)
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from imblearn.over_sampling import SMOTE, SMOTENC
from threadpoolctl import threadpool_limits

VERSION = "JMIR70655-R1.1-NO-VELOCITY"
LOG = logging.getLogger("jmir_r1")
DEFAULT_CSV = "all_data_set_processed_final-extreme_out_v1.csv"
LABEL_MAP = {
    "Rad_2week": "Seroma",
    "Rad_4week": "Bruising",
    "Rad_2mon_w": "Surface irregularity",
    "Rad_2mon_m": "Induration",
}
LEGACY_THRESHOLDS = {"Rad_2week": 0.1, "Rad_4week": 0.3,
                     "Rad_2mon_w": 0.3, "Rad_2mon_m": 0.4}

ALIASES = {
    "HEIGHT": "HIGHT", "height": "HIGHT",
    "total_stroke_count": "totalStrokeCount", "tatalStrokeCount": "totalStrokeCount",
    "total_stroke_velocity": "totalStrokeVelocity",  
    "cohesion_out": "Cohesion_out", "cohesion_min": "Cohesion_min",
    "separation_average": "Separation_average", "separation_max": "Separation_max",
    "separation_min": "Separation_min", "separation_out": "Separation_out",
    "separation_count": "Separation_count",
    "supersonic_waves_size": "Supersonic_Waves_Size",
    "supersonic_Waves_Size": "Supersonic_Waves_Size",
}
PATIENT_CORE = ["age", "sex", "fat_ext_volume", "BEFORE_SIZE", "BEFORE_WEIGHT",
                "HIGHT", "operation_time"]
MOTION_CORE = ["totalStrokeCount", "cohesion_average",
               "cohesion_max", "Cohesion_out", "Separation_average", "Separation_max",
               "Separation_min", "Separation_out"]
OPTIONAL_MOTION = ["cohesion_count", "Separation_count", "Cohesion_min"]
LOG_COLUMNS = ["cohesion_average", "cohesion_max", "Cohesion_min"]
NOISE_COLUMNS = ["age", "BEFORE_SIZE", "BEFORE_WEIGHT", "HIGHT",
                 "Supersonic_Waves_Size", "operation_time",
                 "cohesion_average", "cohesion_max", "Separation_average"]
QUALITY_COLUMNS = ["detection_rate", "fallback_rate", "interpolated_fraction", "rejected_fraction"]
ENGINEERED = ["BMI", "Size_Weight_Ratio",
              "Operation_Time_Stroke_Count", "Separation_Range"]
FAMILIES = ("Patient", "Motion", "Combined", "Augmented")
STRATEGIES = ("none", "weight", "smote", "stack")
METRICS_FOR_CI = ("auroc", "auprc_ap", "brier", "accuracy", "precision", "recall",
                  "specificity", "f1", "flagged_fraction", "flags_per_detected_case")




def jsonable(x: Any) -> Any:
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, (tuple, set)):
        return list(x)
    raise TypeError(type(x).__name__)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=jsonable), encoding="utf-8")


def write_csv(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = obj if isinstance(obj, pd.DataFrame) else pd.DataFrame(obj)
    table.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.10g")


def safe_name(text: Any) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣_.-]+", "_", str(text))


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def seed_for(seed: int, *parts: Any) -> int:

    raw = "|".join(map(str, (seed,) + parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "little") % (2**31 - 1)


def finite_mean(values: Any) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def format_duration(seconds: float) -> str:
  
    if not np.isfinite(seconds) or seconds < 0:
        return "계산 중"
    seconds = int(round(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}시간 {minutes:02d}분 {secs:02d}초"
    if minutes:
        return f"{minutes}분 {secs:02d}초"
    return f"{secs}초"


class ProgressETA:
    
    def __init__(self, label: str, total: int, log_every: Optional[int] = None):
        self.label = label
        self.total = max(int(total), 0)
        self.done = 0
        self.started = time.perf_counter()
        self.log_every = max(1, int(log_every or max(1, self.total // 20))) if self.total else 1
        self._last_logged = 0
        if self.total:
            LOG.info("%s 시작: 총 %s단계", self.label, self.total)

    def update(self, step: int = 1, detail: str = "", force: bool = False) -> None:
        if not self.total:
            return
        self.done = min(self.total, self.done + int(step))
        should_log = force or self.done == 1 or self.done == self.total or (self.done - self._last_logged) >= self.log_every
        if not should_log:
            return
        elapsed = max(time.perf_counter() - self.started, 1e-9)
        rate = self.done / elapsed
        remaining = (self.total - self.done) / rate if rate > 0 else float("nan")
        pct = 100.0 * self.done / self.total
        suffix = f" | {detail}" if detail else ""
        LOG.info(
            "%s 진행: %s/%s (%.1f%%) | 경과 %s | 예상 남은 시간 %s%s",
            self.label, self.done, self.total, pct, format_duration(elapsed),
            format_duration(remaining), suffix,
        )
        self._last_logged = self.done


@dataclass
class Config:
    csv: str = DEFAULT_CSV
    outdir: Optional[str] = None
    id_col: Optional[str] = None
    site_col: Optional[str] = None
    branch_col: Optional[str] = None
    surgeon_col: Optional[str] = None
    schemes: list[str] = field(default_factory=lambda: ["holdout"])
    families: list[str] = field(default_factory=lambda: list(FAMILIES))
    strategies: list[str] = field(default_factory=lambda: list(STRATEGIES))
    outcomes: list[str] = field(default_factory=lambda: list(LABEL_MAP))
    folds: int = 5
    repeats: int = 10
    inner_folds: int = 3
    test_size: float = 0.3
    seed: int = 42
    n_estimators: int = 400
    max_depth: int = 15
    min_samples_leaf: int = 1
    n_jobs: int = -1
    tune: bool = False
    tune_metric: str = "auroc"
    n_bootstrap: int = 2000
    confidence: float = 0.95
    bootstrap_unit: str = "auto"
    min_bootstrap_groups: int = 10
    threshold_mode: str = "recall"
    target_recall: float = 0.8
    fixed_threshold: float = 0.2
    ultrasound_timing: str = "unverified"
    extra_patient_cols: list[str] = field(default_factory=list)
    extra_categorical_cols: list[str] = field(default_factory=list)
    include_optional_motion: bool = False
    log_cohesion: bool = True
    imputation: str = "median"
    quality_csv: Optional[str] = None
    min_detection_rate: Optional[float] = None
    max_interpolated_fraction: Optional[float] = None
    max_rejected_fraction: Optional[float] = None
    shap: bool = True
    shap_family: str = "Combined"
    shap_strategy: str = "none"
    save_models: bool = False
    plots: bool = True
    pilot: bool = False
    confirm_reviewed_features: bool = False


def normalize_columns(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    data = data.copy()
    data.columns = [str(c).strip() for c in data.columns]
    if data.columns.duplicated().any():
        raise ValueError("CSV에 중복 열 이름이 있습니다.")
    changes = {}
    for old, new in ALIASES.items():
        if old in data:
            if new in data and old != new:
                raise ValueError(f"{old}와 {new}가 함께 있습니다. 같은 변수인지 먼저 확인하세요.")
            changes[old] = new
    return data.rename(columns=changes), changes


def build_feature_sets(data: pd.DataFrame, cfg: Config) -> tuple[dict[str, list[str]], list[str]]:

    patient = list(PATIENT_CORE)
    absent = [c for c in PATIENT_CORE + MOTION_CORE if c not in data]
    if absent:
        raise ValueError(f"필수 열이 없습니다: {absent}. 열 이름 또는 자료 버전을 확인하세요.")
    if cfg.ultrasound_timing == "confirmed":
        if "Supersonic_Waves_Size" not in data:
            raise ValueError("초음파 변수를 포함하도록 지정했지만 해당 열이 없습니다.")
        patient.append("Supersonic_Waves_Size")
    else:
        LOG.warning("[AW4] Supersonic_Waves_Size: 측정 시점 미확인/이용불가로 제외합니다.")
    categorical = ["sex"]
    if cfg.site_col:
        if cfg.site_col in set(LABEL_MAP) | {cfg.id_col, cfg.branch_col, cfg.surgeon_col}:
            raise ValueError("수술 부위 열에 결과/환자 ID/그룹 ID를 지정할 수 없습니다.")
        if cfg.site_col not in data:
            raise ValueError(f"수술 부위 열을 찾을 수 없습니다: {cfg.site_col}")
        patient.append(cfg.site_col)
        categorical.append(cfg.site_col)
    else:
        LOG.warning("[AW3] --site-col 미지정. 수술 부위 보완은 아직 완료되지 않았습니다.")
    protected = set(LABEL_MAP) | {cfg.id_col, cfg.surgeon_col, cfg.branch_col,
                                "_case", "_branch", "_surgeon", "_source_row"}
    for c in cfg.extra_patient_cols:
        if c in protected:
            raise ValueError(f"ID/결과 열은 predictor로 추가할 수 없습니다: {c}")
        if c not in data:
            raise ValueError(f"추가 환자/수술 변수 열이 없습니다: {c}")
        if c not in patient:
            patient.append(c)
    categorical.extend(cfg.extra_categorical_cols)
    if not set(categorical).issubset(patient):
        raise ValueError("추가 범주형 변수는 --extra-patient-cols에도 지정해야 합니다.")
    motion = list(MOTION_CORE)

    if cfg.include_optional_motion:
        motion.extend(c for c in OPTIONAL_MOTION if c in data)
    combined = list(dict.fromkeys(patient + motion))
    return {"Patient": patient, "Motion": motion, "Combined": combined,
            "Augmented": combined + list(ENGINEERED)}, list(dict.fromkeys(categorical))


def make_derived_frame(X: pd.DataFrame, features: list[str], log_cohesion: bool) -> pd.DataFrame:

    out = X.copy()
    if "BMI" in features:
        weight = pd.to_numeric(out["BEFORE_WEIGHT"], errors="raise")
        height = pd.to_numeric(out["HIGHT"], errors="raise")
        out["BMI"] = weight.where(weight > 0) / (height.where(height > 0) / 100.0) ** 2
        out["Size_Weight_Ratio"] = out["BEFORE_SIZE"] / weight.where(weight > 0)
        out["Operation_Time_Stroke_Count"] = out["operation_time"] * out["totalStrokeCount"]
        out["Separation_Range"] = out["Separation_max"] - out["Separation_min"]
    out = out[features].copy()

    if log_cohesion:
        for c in LOG_COLUMNS:
            if c in out:
                values = pd.to_numeric(out[c], errors="raise")
                out[c] = np.log(values.where(values > 0))
    return out.replace([np.inf, -np.inf], np.nan)


def category_text(value: Any) -> str:
    
    if pd.isna(value):
        return "__MISSING__"
    if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
        if float(value).is_integer():
            return str(int(value))
    return str(value)


class FoldPreprocessor:
  
    def __init__(self, features: list[str], categorical: list[str], cfg: Config, seed: int):
        self.features = features
        self.categorical = [c for c in features if c in categorical]
        self.numeric = [c for c in features if c not in categorical]
        self.log_cohesion = cfg.log_cohesion
        self.imputation = cfg.imputation
        self.seed = seed

    def _frame(self, X: pd.DataFrame) -> pd.DataFrame:
        out = make_derived_frame(X, self.features, self.log_cohesion)
        for c in self.numeric:
            out[c] = pd.to_numeric(out[c], errors="raise")
        for c in self.categorical:

            out[c] = out[c].map(category_text)
        return out

    def fit(self, X: pd.DataFrame) -> "FoldPreprocessor":
        frame = self._frame(X)
        self.all_missing_ = [c for c in self.numeric if frame[c].isna().all()]
        if self.imputation == "iterative":

            from sklearn.experimental import enable_iterative_imputer  
            from sklearn.impute import IterativeImputer
            self.imputer_ = IterativeImputer(random_state=self.seed, max_iter=15,
                                            sample_posterior=False, keep_empty_features=True)
        else:
            self.imputer_ = SimpleImputer(strategy="median", keep_empty_features=True)
        num = self.imputer_.fit_transform(frame[self.numeric])
        self.scaler_ = StandardScaler().fit(num)
        if self.categorical:
            self.ordinal_ = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            cat = self.ordinal_.fit_transform(frame[self.categorical])
            self.onehot_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            self.onehot_.fit(cat)
            catnames = self.onehot_.get_feature_names_out(self.categorical).tolist()
            parents = [c for c, levels in zip(self.categorical, self.onehot_.categories_)
                       for _ in levels]
        else:
            catnames, parents = [], []
        self.encoded_names_ = self.numeric + catnames
        self.encoded_parents_ = self.numeric + parents
        self.is_fitted_ = True
        return self

    def to_mixed(self, X: pd.DataFrame) -> np.ndarray:
        frame = self._frame(X)
        num = self.scaler_.transform(self.imputer_.transform(frame[self.numeric]))
        if not self.categorical:
            return num
        cat = self.ordinal_.transform(frame[self.categorical])
        return np.column_stack([num, cat])

    def encode_mixed(self, matrix: np.ndarray) -> np.ndarray:
        n = len(self.numeric)
        if not self.categorical:
            return np.asarray(matrix, dtype=float)
        return np.column_stack([matrix[:, :n], self.onehot_.transform(matrix[:, n:])])

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.encode_mixed(self.to_mixed(X))

    def audit(self) -> dict[str, Any]:
        return {"numeric": self.numeric, "categorical": self.categorical,
                "all_missing_in_fit": self.all_missing_,
                "numeric_center": self.scaler_.mean_, "numeric_scale": self.scaler_.scale_}


@dataclass
class BaseModel:
    processor: FoldPreprocessor
    forest: RandomForestClassifier
    fit_info: dict[str, Any]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:

        return self.forest.predict_proba(self.processor.transform(X))[:, 1]


@dataclass
class ModelBundle:
    base: BaseModel
    calibrator: LogisticRegression
    raw_threshold: float
    calibrated_threshold: float
    fit_info: dict[str, Any]

    def predict(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        p_raw = self.base.predict_proba(X)
        p_cal = self.calibrator.predict_proba(p_raw.reshape(-1, 1))[:, 1]
        return p_raw, p_cal


def augment_training_only(X: pd.DataFrame, y: np.ndarray, seed: int) -> tuple[pd.DataFrame, np.ndarray]:

    pos = X.loc[np.asarray(y) == 1].copy()
    rng = np.random.default_rng(seed)
    for c in NOISE_COLUMNS:
        if c in pos:
            values = pd.to_numeric(pos[c], errors="raise")
            factors = np.maximum(1 + rng.normal(0, 0.05, len(pos)), 1e-6)
            pos[c] = values * factors
    return pd.concat([X, pos], ignore_index=True), np.concatenate([y, np.ones(len(pos), dtype=int)])


def fit_base(X: pd.DataFrame, y: np.ndarray, features: list[str], categorical: list[str],
             strategy: str, params: dict[str, int], cfg: Config, seed: int) -> BaseModel:
    y = np.asarray(y, dtype=int)
    counts = np.bincount(y, minlength=2)
    if np.min(counts) == 0:
        raise AnalysisUnavailable("학습 자료에 한 클래스만 있습니다.")

    processor = FoldPreprocessor(features, categorical, cfg, seed).fit(X)
    X_fit, y_fit = X, y
    if strategy == "stack":
        X_fit, y_fit = augment_training_only(X, y, seed_for(seed, "noise"))
    matrix = processor.to_mixed(X_fit)
    before_smote = len(y_fit)
    smote_name, k = "none", None
    if strategy in {"smote", "stack"}:
        if np.min(counts) < 2:
            raise AnalysisUnavailable("실제 학습 환자 중 소수 클래스가 2명 미만이므로 SMOTE 불가.")

        k = int(min(5, np.min(counts) - 1))
        if processor.categorical:
            cat_indices = list(range(len(processor.numeric), matrix.shape[1]))
            sampler = SMOTENC(categorical_features=cat_indices, k_neighbors=k,
                              random_state=seed_for(seed, "smote"))
            smote_name = "SMOTENC"
        else:
            sampler = SMOTE(k_neighbors=k, random_state=seed_for(seed, "smote"))
            smote_name = "SMOTE"
        matrix, y_fit = sampler.fit_resample(matrix, y_fit)
    encoded = processor.encode_mixed(matrix)
    weight = {0: 1, 1: 2} if strategy in {"weight", "stack"} else None
    forest = RandomForestClassifier(
        n_estimators=cfg.n_estimators, max_depth=params["max_depth"],
        min_samples_split=3, min_samples_leaf=params["min_samples_leaf"],
        class_weight=weight, random_state=seed, n_jobs=cfg.n_jobs,
    )
    forest.fit(encoded, y_fit)
    return BaseModel(processor, forest, {
        "seed": seed, "params": params, "strategy": strategy,
        "real_n": len(y), "real_negative": int(counts[0]), "real_positive": int(counts[1]),
        "after_noise_n": before_smote, "after_resampling_n": len(y_fit),
        "sampler": smote_name, "k_neighbors": k, "class_weight": weight,
        "processor": processor.audit(),
    })


def inner_splits(y: np.ndarray, groups: Optional[np.ndarray], n_splits: int,
                 seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
 
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        raise AnalysisUnavailable("내부 검증에 두 클래스가 필요합니다.")
    upper = min(n_splits, int(np.bincount(y, minlength=2).min()))
    if groups is not None:
        upper = min(upper, len(np.unique(groups)))
    for k in range(upper, 1, -1):

        for attempt in range(8 if groups is not None else 1):
            actual_seed = seed_for(seed, "feasible_split", k, attempt)
            if groups is None:
                splitter = StratifiedKFold(k, shuffle=True, random_state=actual_seed)
                splits = list(splitter.split(np.zeros(len(y)), y))
            else:
                splitter = StratifiedGroupKFold(k, shuffle=True, random_state=actual_seed)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    splits = list(splitter.split(np.zeros(len(y)), y, groups))
            good = all(len(te) > 0 and np.bincount(y[tr], minlength=2).min() >= 2
                       for tr, te in splits)
            if good:
                seen = np.concatenate([te for _, te in splits])
                assert np.array_equal(np.sort(seen), np.arange(len(y)))
                for tr, te in splits:
                    assert not set(tr) & set(te)
                    if groups is not None:
                        assert not set(groups[tr]) & set(groups[te])
                return splits
    raise AnalysisUnavailable("그룹/사건 수 때문에 내부 학습 fold당 실제 두 클래스 각 2명 이상을 확보할 수 없습니다.")


def split_record(X: pd.DataFrame, splits: list[tuple[np.ndarray, np.ndarray]]) -> list[dict[str, Any]]:
    return [{"fold": f, "train_cases": X.iloc[tr]["_case"].tolist(),
             "validation_cases": X.iloc[te]["_case"].tolist()}
            for f, (tr, te) in enumerate(splits)]


def select_parameters(X: pd.DataFrame, y: np.ndarray, groups: Optional[np.ndarray],
                      features: list[str], categorical: list[str], strategy: str,
                      cfg: Config, seed: int) -> tuple[dict[str, int], dict[str, Any]]:

    default = {"max_depth": cfg.max_depth, "min_samples_leaf": cfg.min_samples_leaf}
    if not cfg.tune:
        return default, {"status": "fixed_original_settings", "selected": default}
    candidates = [default, {"max_depth": 10, "min_samples_leaf": 3},
                  {"max_depth": 8, "min_samples_leaf": 5}]
    candidates = [dict(t) for t in dict.fromkeys(tuple(c.items()) for c in candidates)]
    try:
        splits = inner_splits(y, groups, cfg.inner_folds, seed_for(seed, "tune"))
    except AnalysisUnavailable as exc:

        return default, {"status": "tuning_unavailable_fixed_settings", "reason": str(exc),
                         "selected": default}
    trials = []
    for candidate in candidates:
        oof = np.full(len(y), np.nan)
        for f, (tr, va) in enumerate(splits):
            model = fit_base(X.iloc[tr], y[tr], features, categorical, strategy, candidate,
                             cfg, seed_for(seed, "tune_fit", f))
            oof[va] = model.predict_proba(X.iloc[va])
        assert np.isfinite(oof).all()
        score = (roc_auc_score(y, oof) if cfg.tune_metric == "auroc"
                 else average_precision_score(y, oof))
        trials.append({"params": candidate, "inner_oof_score": float(score)})

    chosen = sorted(trials, key=lambda t: (-t["inner_oof_score"], t["params"]["max_depth"],
                                          -t["params"]["min_samples_leaf"]))[0]["params"]
    return chosen, {"status": "training_only_inner_cv", "selected": chosen,
                    "scoring": cfg.tune_metric, "trials": trials,
                    "splits": split_record(X, splits)}


def choose_threshold(y: np.ndarray, probability: np.ndarray, cfg: Config) -> float:

    if cfg.threshold_mode == "fixed":
        return float(cfg.fixed_threshold)
    fpr, tpr, thresholds = roc_curve(y, probability, drop_intermediate=False)
    finite = np.isfinite(thresholds) & (thresholds >= 0) & (thresholds <= 1)
    if cfg.threshold_mode == "recall":
        candidates = thresholds[finite & (tpr >= cfg.target_recall - 1e-12)]
        if not len(candidates):
            raise AnalysisUnavailable("학습 OOF에서 지정된 recall 목표 임계값을 찾을 수 없습니다.")
        return float(np.max(candidates))
    scores = np.where(finite, tpr - fpr, -np.inf)
    return float(thresholds[np.argmax(scores)])


def fit_bundle(X: pd.DataFrame, y: np.ndarray, groups: Optional[np.ndarray],
               features: list[str], categorical: list[str], strategy: str,
               cfg: Config, seed: int, audit_dir: Optional[Path] = None) -> ModelBundle:

    splits = inner_splits(y, groups, cfg.inner_folds, seed_for(seed, "calibration"))
    raw_oof = np.full(len(y), np.nan)
    records = []
    for f, (tr, va) in enumerate(splits):
        subgroups = None if groups is None else groups[tr]
        params, tuning = select_parameters(
            X.iloc[tr], y[tr], subgroups, features, categorical, strategy,
            cfg, seed_for(seed, "inner", f))
        base = fit_base(X.iloc[tr], y[tr], features, categorical, strategy, params,
                        cfg, seed_for(seed, "inner_final", f))
        raw_oof[va] = base.predict_proba(X.iloc[va])
        records.append({"fold": f, "tuning": tuning, "fit": base.fit_info})
    assert np.isfinite(raw_oof).all(), "모든 학습 환자에게 OOF 예측이 필요합니다."

    calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000, tol=1e-8)
    calibrator.fit(raw_oof.reshape(-1, 1), y)
    cal_oof = calibrator.predict_proba(raw_oof.reshape(-1, 1))[:, 1]
    if float(calibrator.coef_[0, 0]) < 0:
        LOG.warning("학습 OOF의 보정 기울기가 음수입니다. raw/calibrated 순위 변화와 불안정성을 확인하세요.")
    raw_threshold = choose_threshold(y, raw_oof, cfg)
    cal_threshold = choose_threshold(y, cal_oof, cfg)
    params, final_tuning = select_parameters(
        X, y, groups, features, categorical, strategy, cfg, seed_for(seed, "final_tuning"))
    final_base = fit_base(X, y, features, categorical, strategy, params,
                         cfg, seed_for(seed, "final_base"))
    info = {"version": VERSION, "training_n": len(y), "training_events": int(y.sum()),
            "seed": seed, "calibration_method": "sigmoid_on_raw_probability_C_1e6",
            "calibration_a": float(calibrator.coef_[0, 0]),
            "calibration_b": float(calibrator.intercept_[0]),
            "threshold_mode": cfg.threshold_mode, "target_recall": cfg.target_recall,
            "raw_threshold": raw_threshold, "calibrated_threshold": cal_threshold,
            "calibration_splits": split_record(X, splits), "inner_fits": records,
            "final_tuning": final_tuning, "final_fit": final_base.fit_info}
    if audit_dir is not None:
        write_json(audit_dir / "fit_audit.json", info)
        write_csv(audit_dir / "training_oof_for_fitting_only.csv", pd.DataFrame({
            "case": X["_case"].to_numpy(), "y_true": y,
            "raw_oof": raw_oof, "calibrator_fitted_oof": cal_oof,
            "role": "training_only_not_test_performance"}))
    return ModelBundle(final_base, calibrator, raw_threshold, cal_threshold, info)


def metrics(y: np.ndarray, p: np.ndarray, decisions: np.ndarray) -> dict[str, float]:

    y, p, decisions = np.asarray(y, int), np.asarray(p, float), np.asarray(decisions, int)
    if not (len(y) == len(p) == len(decisions)) or not len(y):
        raise ValueError("metric 입력 길이가 맞지 않습니다.")
    tn, fp, fn, tp = [int(v) for v in confusion_matrix(y, decisions, labels=[0, 1]).ravel()]
    n, positives, flags = len(y), tp + fn, tp + fp

    two_classes = len(np.unique(y)) == 2
    return {
        "n": n, "events": positives, "nonevents": tn + fp,
        "prevalence": positives / n, "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "accuracy": (tp + tn) / n,
        "precision": tp / flags if flags else np.nan,
        "recall": tp / positives if positives else np.nan,
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan,
        "auroc": float(roc_auc_score(y, p)) if two_classes else np.nan,

        "auprc_ap": float(average_precision_score(y, p)) if two_classes else np.nan,
        "brier": float(brier_score_loss(y, p)),
        "flagged_n": flags, "flagged_fraction": flags / n,
        "flags_per_detected_case": flags / tp if tp else np.nan,
    }


def calibration_diagnostics(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
   
    if len(np.unique(y)) < 2:
        return {"calibration_intercept": np.nan, "calibration_slope": np.nan,
                "calibration_in_the_large": np.nan, "diagnostic_status": "one_class"}
    z = logit(np.clip(p, 1e-6, 1 - 1e-6))
    if np.std(z) < 1e-10:
        return {"calibration_intercept": np.nan, "calibration_slope": np.nan,
                "calibration_in_the_large": np.nan, "diagnostic_status": "constant_score"}
    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        score = theta[0] + theta[1] * z
        residual = expit(score) - y
        loss = np.mean(np.logaddexp(0, score) - y * score)
        return float(loss), np.array([residual.mean(), np.mean(residual * z)])
    result = minimize(objective, [0., 1.], method="BFGS", jac=True,
                      options={"gtol": 1e-7, "maxiter": 500})
    ok = np.isfinite(result.x).all() and np.linalg.norm(objective(result.x)[1]) < 1e-5
    ok = ok and np.max(np.abs(result.x)) < 100
    citl = brentq(lambda a: float(np.mean(expit(a + z)) - np.mean(y)), -60, 60)
    return {"calibration_intercept": float(result.x[0]) if ok else np.nan,
            "calibration_slope": float(result.x[1]) if ok else np.nan,
            "calibration_in_the_large": float(citl),
            "diagnostic_status": ("descriptive_sparse_events" if min(y.sum(), len(y)-y.sum()) < 20
                                  else "descriptive") if ok else "unstable_fit"}


def wilson_interval(events: int, n: int, z: float = 1.95996398454) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    p = events / n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    width = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return center - width, center + width


def calibration_bins(y: np.ndarray, p: np.ndarray, bins: int = 8) -> pd.DataFrame:
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins+1)))
    if len(edges) == 1:
        assignment = np.zeros(len(p), dtype=int)
    else:
        assignment = np.searchsorted(edges[1:-1], p, side="right")
    rows = []
    for group in np.unique(assignment):
        mask = assignment == group
        n, events = int(mask.sum()), int(y[mask].sum())
        lo, hi = wilson_interval(events, n)
        rows.append({"bin": int(group), "n": n, "events": events,
                     "predicted_mean": float(p[mask].mean()), "observed_fraction": events/n,
                     "observed_wilson_low": lo, "observed_wilson_high": hi})
    return pd.DataFrame(rows)


def clinical_table(y: np.ndarray, p: np.ndarray, thresholds: list[float]) -> pd.DataFrame:

    rows = []
    for t in thresholds:
        m = metrics(y, p, (p >= t).astype(int))
        odds = t/(1-t)
        all_nb = m["prevalence"] - (1-m["prevalence"]) * odds
        nb = m["tp"]/m["n"] - m["fp"]/m["n"] * odds
        rows.append({"threshold": t, **m, "net_benefit": nb,
                     "treat_all_net_benefit": all_nb, "treat_none_net_benefit": 0.,
                     "difference_vs_treat_all": nb-all_nb, "difference_vs_treat_none": nb})
    return pd.DataFrame(rows)


def audit_and_load(cfg: Config, run: Path) -> tuple[pd.DataFrame, dict[str, list[str]], list[str]]:

    csv_path = Path(cfg.csv).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV가 없습니다: {csv_path}\n--csv로 실제 파일 경로를 지정하세요.")
    data, aliases = normalize_columns(pd.read_csv(csv_path))
    if data.empty:
        raise ValueError("CSV가 비어 있습니다.")
    reserved = [c for c in ("_case", "_source_row", "_branch", "_surgeon") if c in data]
    if reserved:
        raise ValueError(f"내부 예약 열 이름과 겹칩니다: {reserved}")
    missing_rows = [{"variable": c, "csv_n": len(data), "missing_n": int(data[c].isna().sum()),
                     "missing_fraction": float(data[c].isna().mean()),
                     "stage": "input_csv_before_any_new_imputation"} for c in data]
    write_csv(run / "missingness.csv", missing_rows)
    data["_source_row"] = np.arange(len(data))
    if cfg.id_col:
        if cfg.id_col not in data or data[cfg.id_col].isna().any():
            raise ValueError("--id-col이 없거나 환자 ID에 결측이 있습니다.")
        if data[cfg.id_col].duplicated().any():
            raise ValueError("중복 환자 ID가 있습니다. 이 버전은 1환자 1행을 전제로 합니다. "
                             "동일 환자의 여러 수술을 서로 다른 환자처럼 분할하지 마세요.")
        data["_case"] = data[cfg.id_col].map(
            lambda v: "case_" + hashlib.sha256(str(v).encode()).hexdigest()[:16])
    else:
        LOG.warning("환자 ID 미지정: 행 단위 ID를 사용합니다. 동일 환자의 중복 수술 여부는 확인할 수 없습니다.")
        data["_case"] = [f"row_{i:07d}" for i in range(len(data))]
    if data["_case"].duplicated().any():
        raise ValueError("내부 case ID가 중복되었습니다.")
    for option in [cfg.branch_col, cfg.surgeon_col]:
        if option in LABEL_MAP:
            raise ValueError("그룹 ID에 결과 열을 지정할 수 없습니다.")
    for option, internal, prefix in [(cfg.branch_col, "_branch", "branch"),
                                     (cfg.surgeon_col, "_surgeon", "surgeon")]:
        if option:
            if option not in data:
                raise ValueError(f"그룹 열이 없습니다: {option}")
            keys = sorted(data[option].dropna().astype(str).unique())
            mapping = {key: f"{prefix}_{i+1:04d}" for i, key in enumerate(keys)}
            data[internal] = data[option].map(lambda v: mapping[str(v)] if pd.notna(v) else None)
        else:
            data[internal] = None
    if cfg.quality_csv:
        if not cfg.id_col:
            raise ValueError("quality.csv를 합치려면 --id-col로 동일 환자 키를 지정해야 합니다.")
        q = pd.read_csv(cfg.quality_csv)
        needed = [c for c in QUALITY_COLUMNS if c in q]
        if not needed or cfg.id_col not in q:
            raise ValueError("quality.csv에 ID와 기록 품질 비율 열이 필요합니다.")
        if q[cfg.id_col].isna().any() or q[cfg.id_col].duplicated().any():
            raise ValueError("quality.csv의 ID에 결측 또는 중복이 있습니다.")
        if any(c in data for c in needed):
            raise ValueError("기존 CSV와 quality.csv에 같은 품질 열이 있습니다. 사용할 자료를 먼저 결정하세요.")
        data = data.merge(q[[cfg.id_col] + needed], on=cfg.id_col, how="left", validate="one_to_one")
        data = data.sort_values("_source_row").reset_index(drop=True)
    features, categorical = build_feature_sets(data, cfg)
    for outcome in cfg.outcomes:
        if outcome not in data:
            raise ValueError(f"결과 열이 없습니다: {outcome}")
        data[outcome] = pd.to_numeric(data[outcome], errors="raise")
        observed = set(data[outcome].dropna().unique())
        if not observed.issubset({0, 1}):
            raise ValueError(f"{outcome}은 0/1/결측이어야 합니다. 임의로 이분화하지 않습니다.")
    numeric = [c for c in features["Combined"] if c not in categorical]
    for c in numeric:
        data[c] = pd.to_numeric(data[c], errors="raise")

    rows = []
    for c in numeric + [q for q in QUALITY_COLUMNS if q in data]:
        v = pd.to_numeric(data[c], errors="raise")
        arr = v.to_numpy(float)
        finite = arr[np.isfinite(arr)]
        rows.append({"variable": c, "n_total": len(arr), "missing_n": int(np.isnan(arr).sum()),
                     "infinite_n": int(np.isinf(arr).sum()), "finite_n": len(finite),
                     "min": float(np.min(finite)) if len(finite) else np.nan,
                     "p01": float(np.quantile(finite, .01)) if len(finite) else np.nan,
                     "median": float(np.median(finite)) if len(finite) else np.nan,
                     "p99": float(np.quantile(finite, .99)) if len(finite) else np.nan,
                     "max": float(np.max(finite)) if len(finite) else np.nan,
                     "mean": float(np.mean(finite)) if len(finite) else np.nan,
                     "sd": float(np.std(finite, ddof=1)) if len(finite) > 1 else np.nan,
                     "nonpositive_n": int((finite <= 0).sum()),
                     "unit_verified_by_code": False})
        data[c] = v.replace([np.inf, -np.inf], np.nan)
        if c in QUALITY_COLUMNS and ((data[c] < 0) | (data[c] > 1)).any():
            raise ValueError(f"{c}는 0~1 비율이어야 합니다. 퍼센트 단위를 자동 추정하지 않습니다.")
    write_csv(run / "numeric_audit.csv", rows)
    for c in LOG_COLUMNS:
        if cfg.log_cohesion and c in data and (data[c] <= 0).any():
            LOG.warning("%s의 0 이하 %s건은 log 적용 때 결측 처리됩니다.", c, int((data[c] <= 0).sum()))
    derived = make_derived_frame(data, features["Augmented"], cfg.log_cohesion)
    write_csv(run / "derived_missingness.csv", [
        {"variable": c, "n": len(derived), "missing_n": int(derived[c].isna().sum()),
         "missing_fraction": float(derived[c].isna().mean()), "stage": "before_fold_imputation"}
        for c in derived])
    write_json(run / "feature_sets.json", features)
    write_json(run / "column_aliases.json", aliases)
    feature_dict = []
    for c in features["Augmented"]:
        feature_dict.append({"variable": c, "type": "categorical" if c in categorical else "numeric",
                             "included_in": ";".join(k for k, v in features.items() if c in v),
                             "measurement_time": "requires_clinical_confirmation",
                             "definition_and_unit": "requires_source_confirmation",
                             "log_applied": cfg.log_cohesion and c in LOG_COLUMNS,
                             "derived_before_scaling": c in ENGINEERED})
    write_csv(run / "feature_dictionary.csv", feature_dict)
    group_rows = []
    for col in ["_branch", "_surgeon"] + ([cfg.site_col] if cfg.site_col else []):
        if not data[col].notna().any():
            continue
        for group, subset in data.groupby(col, dropna=False):
            for label in cfg.outcomes:
                available = subset[label].dropna()
                group_rows.append({"group_type": col, "group": str(group), "outcome": label,
                                   "n_rows": len(subset), "outcome_observed_n": len(available),
                                   "outcome_missing_n": int(subset[label].isna().sum()),
                                   "positive_n": int(available.sum()),
                                   "prevalence": float(available.mean()) if len(available) else np.nan})
    write_csv(run / "group_counts.csv", group_rows)
    if cfg.branch_col and cfg.surgeon_col:
        cross = pd.crosstab(data["_surgeon"], data["_branch"], dropna=False).reset_index()
        write_csv(run / "surgeon_branch_distribution.csv", cross)

    correlations = []
    for m in features["Motion"]:
        for other in ["HIGHT", "BEFORE_SIZE", "BEFORE_WEIGHT", "fat_ext_volume", "operation_time"] + QUALITY_COLUMNS:
            if other not in data:
                continue
            pair = data[[m, other]].dropna()
            rho = np.nan
            if len(pair) >= 5 and pair[m].nunique() > 1 and pair[other].nunique() > 1:
                rho = float(spearmanr(pair[m], pair[other]).statistic)
            correlations.append({"motion_feature": m, "other_feature": other,
                                 "n_complete_pairs": len(pair), "spearman_rho": rho,
                                 "interpretation": "descriptive_association_not_causation"})
    write_csv(run / "motion_size_quality_associations.csv", correlations)
    if cfg.site_col:
        bysite = data.groupby(cfg.site_col, dropna=False)[features["Motion"]].agg(["count", "mean", "median", "std"])
        bysite.columns = [f"{a}__{b}" for a, b in bysite.columns]
        write_csv(run / "motion_by_site.csv", bysite.reset_index())

        site_test_rows = []

        major_sites = {"0", "1", "2"}
        site_text = data[cfg.site_col].map(category_text)

        for motion_feature in features["Motion"]:
            groups = []
            group_ns = {}

            for site in ["0", "1", "2"]:
                values = pd.to_numeric(
                    data.loc[site_text == site, motion_feature],
                    errors="coerce"
                ).dropna()

                groups.append(values.to_numpy(dtype=float))
                group_ns[site] = int(len(values))

            if all(len(g) > 0 for g in groups):
                h_stat, p_value = kruskal(*groups)

                n_total = sum(len(g) for g in groups)
                k = len(groups)

                epsilon_sq = (
                    (h_stat - k + 1) / (n_total - k)
                    if n_total > k else np.nan
                )
                epsilon_sq = max(0.0, float(epsilon_sq)) if np.isfinite(epsilon_sq) else np.nan

            else:
                h_stat = np.nan
                p_value = np.nan
                epsilon_sq = np.nan

            site_test_rows.append({
                "motion_feature": motion_feature,
                "sites_compared": "0;1;2",
                "n_site_0": group_ns.get("0", 0),
                "n_site_1": group_ns.get("1", 0),
                "n_site_2": group_ns.get("2", 0),
                "kruskal_H": h_stat,
                "p_value": p_value,
                "epsilon_squared": epsilon_sq,
                "note": "sites_3_4_5_excluded_from_inference_due_to_small_n",
            })

        site_tests = pd.DataFrame(site_test_rows)

        valid = site_tests["p_value"].notna()
        pvals = site_tests.loc[valid, "p_value"].to_numpy(dtype=float)

        adjusted = np.full(len(pvals), np.nan)

        if len(pvals):
            order = np.argsort(pvals)
            ranked = pvals[order]
            m = len(ranked)

            bh = ranked * m / np.arange(1, m + 1)
            bh = np.minimum.accumulate(bh[::-1])[::-1]
            bh = np.clip(bh, 0, 1)

            adjusted[order] = bh

        site_tests["p_fdr_bh"] = np.nan
        site_tests.loc[valid, "p_fdr_bh"] = adjusted
        site_tests["fdr_significant_0_05"] = (
            site_tests["p_fdr_bh"] < 0.05
        )

        write_csv(run / "motion_site_tests.csv", site_tests)
    if not cfg.confirm_reviewed_features:
        LOG.warning("Cohesion/Separation 계산, 단위, 결과 판정을 확인하지 않은 잠정 분석입니다.")
    write_json(run / "data_manifest.json", {
        "csv_path": str(csv_path), "csv_sha256": file_hash(csv_path), "input_n": len(data),
        "quality_csv_sha256": file_hash(Path(cfg.quality_csv)) if cfg.quality_csv else None,
        "assumption": "one_patient_one_row", "PSENTRY_provided": bool(cfg.id_col),
        "prior_exclusions_or_imputation": "unknown_from_this_csv",
        "features_reviewed_by_user_declaration": cfg.confirm_reviewed_features,
        "no_global_maximum_deletion": True,
        "ultrasound_timing": cfg.ultrasound_timing,
    })
    return data, features, categorical


def make_cohorts(data: pd.DataFrame, cfg: Config) -> dict[str, pd.DataFrame]:
    cohorts = {"full": data.copy()}
    criteria = [("detection_rate", cfg.min_detection_rate, "min"),
                ("interpolated_fraction", cfg.max_interpolated_fraction, "max"),
                ("rejected_fraction", cfg.max_rejected_fraction, "max")]
    active = [(c, t, side) for c, t, side in criteria if t is not None]
    if active:

        mask = np.ones(len(data), dtype=bool)
        for c, t, side in active:
            if c not in data:
                raise ValueError(f"품질 기준을 적용하려면 {c} 열이 필요합니다.")
            passing = data[c] >= t if side == "min" else data[c] <= t
            mask &= data[c].notna().to_numpy() & passing.to_numpy()
        cohorts["quality_filtered"] = data.loc[mask].copy()
    return cohorts


def outer_splits(data: pd.DataFrame, y: np.ndarray, scheme: str,
                 cfg: Config) -> list[tuple[int, int, np.ndarray, np.ndarray]]:

    counts = np.bincount(y, minlength=2)
    if counts.min() < 3:
        raise AnalysisUnavailable("해당 결과의 실제 한 클래스가 3명 미만입니다.")
    indices = np.arange(len(data))
    if scheme == "holdout":
        tr, te = train_test_split(indices, test_size=cfg.test_size,
                                 random_state=cfg.seed, stratify=y)
        return [(0, 0, tr, te)]
    if scheme == "cv":
        k = min(cfg.folds, int(counts.min()))
        results = []
        for repeat in range(cfg.repeats):
            splitter = StratifiedKFold(k, shuffle=True, random_state=cfg.seed + repeat)
            results.extend((repeat, f, tr, te)
                           for f, (tr, te) in enumerate(splitter.split(indices, y)))
        return results
    group_col = "_branch" if scheme == "branch" else "_surgeon"
    groups = data[group_col].astype(str).to_numpy()
    if len(np.unique(groups)) < 2:
        raise AnalysisUnavailable(f"{scheme} 그룹이 2개 미만입니다.")
    return [(0, f, tr, te) for f, (tr, te) in enumerate(
        LeaveOneGroupOut().split(indices, y, groups))]


def make_outer_manifest(data: pd.DataFrame, y: np.ndarray,
                        splits: list[tuple[int, int, np.ndarray, np.ndarray]],
                        group_col: Optional[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    assignment, counts = [], []
    for repeat, fold, tr, te in splits:
        assert not set(data.iloc[tr]["_case"]) & set(data.iloc[te]["_case"])
        if group_col:
            assert not set(data.iloc[tr][group_col]) & set(data.iloc[te][group_col])
        shared = {}
        for col in ["_branch", "_surgeon"]:
            shared[f"shared{col}_count"] = len(set(data.iloc[tr][col].dropna()) &
                                                  set(data.iloc[te][col].dropna()))
        counts.append({"repeat": repeat, "fold": fold, "train_n": len(tr),
                       "train_events": int(y[tr].sum()), "evaluation_n": len(te),
                       "evaluation_events": int(y[te].sum()), **shared})
        for role, ix in [("train", tr), ("evaluation", te)]:
            for j in ix:
                assignment.append({"repeat": repeat, "fold": fold, "role": role,
                                   "case": data.iloc[j]["_case"], "y_true": int(y[j]),
                                   "branch": data.iloc[j]["_branch"],
                                   "surgeon": data.iloc[j]["_surgeon"]})
    return pd.DataFrame(assignment), pd.DataFrame(counts)


def compute_shap(bundle: ModelBundle, X_test: pd.DataFrame,
                 repeat: int, fold: int, path: Path) -> Optional[pd.DataFrame]:

    try:
        import shap
    except ImportError:
        LOG.warning("shap가 없어 SHAP 생성을 건너뜁니다. 설치 후 해당 조건을 다시 실행하세요.")
        write_json(path / "shap_unavailable.json", {"reason": "shap_not_installed"})
        return None
    processor = bundle.base.processor
    matrix = processor.transform(X_test)
    explainer = shap.TreeExplainer(bundle.base.forest, feature_perturbation="tree_path_dependent")
    pieces = []
    shap_starts = list(range(0, len(X_test), 128))
    shap_progress = ProgressETA(
        f"SHAP repeat={repeat} fold={fold}", len(shap_starts), log_every=1
    )
    for start in shap_starts:
        val = explainer.shap_values(matrix[start:start+128], check_additivity=True)
        if isinstance(val, list):
            val = np.asarray(val[1])
        else:
            val = np.asarray(val)
            if val.ndim == 3 and val.shape[-1] == 2:
                val = val[:, :, 1]
        if val.ndim != 2 or val.shape[1] != matrix.shape[1]:
            raise ValueError(f"SHAP 반환 형식 확인 필요: {val.shape}")
        pieces.append(val)
        shap_progress.update(
            detail=f"cases {min(start+128, len(X_test))}/{len(X_test)}"
        )
    encoded_shap = np.vstack(pieces)
    expected = np.asarray(explainer.expected_value).reshape(-1)
    expected_positive = float(expected[1] if len(expected) == 2 else expected[0])
    predicted = bundle.base.predict_proba(X_test)
    if not np.allclose(encoded_shap.sum(axis=1) + expected_positive, predicted, atol=1e-5):
        raise ValueError("SHAP 합계와 보정 전 RF 확률이 맞지 않습니다.")
    aggregated = np.zeros((len(X_test), len(processor.features)))
    for i, parent in enumerate(processor.encoded_parents_):
        aggregated[:, processor.features.index(parent)] += encoded_shap[:, i]
    display = processor._frame(X_test)
    for c in processor.categorical:
        display[c] = np.nan  
    rows = []
    for j, feature in enumerate(processor.features):
        for i, case in enumerate(X_test["_case"]):
            rows.append({"case": case, "repeat": repeat, "fold": fold, "feature": feature,
                         "display_value": display.iloc[i][feature],
                         "display_scale": "log_raw" if processor.log_cohesion and feature in LOG_COLUMNS else "raw",
                         "shap_value": aggregated[i, j], "expected_value": expected_positive,
                         "raw_prediction": predicted[i], "explained_output": "uncalibrated_RF_probability"})
    result = pd.DataFrame(rows)
    path.mkdir(parents=True, exist_ok=True)
    result.to_csv(path / "shap_values.csv.gz", index=False, compression="gzip", encoding="utf-8-sig")
    return result


def plot_shap_summary(values: pd.DataFrame, path: Path, title: str) -> None:

    ranks = values.assign(abs_shap=values.shap_value.abs()).groupby("feature").abs_shap.mean().sort_values()
    write_csv(path / "mean_absolute_shap.csv", ranks.rename("mean_abs_shap").reset_index())
    fold_ranks = (values.assign(abs_shap=values.shap_value.abs())
                  .groupby(["fold", "feature"]).abs_shap.mean().reset_index())
    fold_ranks["rank"] = fold_ranks.groupby("fold").abs_shap.rank(ascending=False, method="average")
    write_csv(path / "fold_shap_ranks.csv", fold_ranks)
    plt.figure(figsize=(10, 8))
    top = ranks.tail(15)
    plt.barh(top.index, top.values)
    plt.xlabel("Mean absolute SHAP, raw RF probability")
    plt.title(title + "\nFirst repeat, all out-of-fold evaluation cases", fontsize=11)
    plt.tight_layout()
    plt.savefig(path / "shap_importance.png", dpi=200)
    plt.close()

    plt.figure(figsize=(11, 8))
    rng = np.random.default_rng(42)
    for i, feature in enumerate(top.index):
        a = values.loc[values.feature == feature, "shap_value"].to_numpy()
        plt.scatter(a, i + rng.uniform(-.22, .22, len(a)), s=9, alpha=.35)
    plt.yticks(np.arange(len(top)), top.index)
    plt.axvline(0, linestyle=":", linewidth=1)
    plt.xlabel("SHAP contribution to raw RF probability")
    plt.title(title + "\nAll evaluation cases in first repeat", fontsize=11)
    plt.tight_layout()
    plt.savefig(path / "shap_distribution.png", dpi=200)
    plt.close()
    tails = []
    tail_features = list(dict.fromkeys(ranks.tail(5).index.tolist() +
                                      [c for c in ["Cohesion_out", "cohesion_max", "totalStrokeCount", "Separation_out"]
                                       if c in ranks.index]))
    context_group_rows = []
    categorical_context = set(values.attrs.get("categorical_context", []))
    for feature in tail_features:
        subset = values.loc[values.feature == feature].copy()
        subset["display_value"] = pd.to_numeric(subset.display_value, errors="coerce")
        subset = subset.dropna(subset=["display_value"])
        if subset.empty or subset.display_value.nunique() < 3:
            continue
        lo, hi = subset.display_value.quantile([.1, .9])
        subset["region"] = np.where(subset.display_value <= lo, "low_tail",
                                    np.where(subset.display_value >= hi, "high_tail", "middle"))
        for region, part in subset.groupby("region"):
            tail_row = {"feature": feature, "region": region, "n": len(part),
                        "mean_shap": part.shap_value.mean(), "median_value": part.display_value.median(),
                        "p10_boundary": lo, "p90_boundary": hi}

            for context in [c for c in part if c.startswith("context_")]:
                numeric_context = pd.to_numeric(part[context], errors="coerce")
                if numeric_context.notna().any() and context not in categorical_context:
                    tail_row[context + "_median"] = numeric_context.median()
                    tail_row[context + "_observed_n"] = int(numeric_context.notna().sum())
                    if context == "context_outcome":
                        tail_row["observed_event_fraction_descriptive"] = numeric_context.mean()
                else:
                    for category, count in part[context].fillna("__MISSING__").value_counts().items():
                        context_group_rows.append({"feature": feature, "region": region,
                                                   "context_variable": context, "category": category,
                                                   "n": int(count), "fraction": float(count/len(part))})
            tails.append(tail_row)
        plt.figure(figsize=(9, 6))
        plt.scatter(subset.display_value, subset.shap_value, s=12, alpha=.45)
        plt.axhline(0, linestyle=":", linewidth=1)
        scale = subset.display_scale.iloc[0]
        plt.xlabel(f"{feature} ({scale})", fontsize=12)
        plt.ylabel("SHAP contribution to raw RF probability", fontsize=12)
        plt.title(title, fontsize=11)
        plt.tight_layout()
        plt.savefig(path / f"shap_dependence_{safe_name(feature)}.png", dpi=200)
        plt.close()
    write_csv(path / "shap_low_middle_high.csv", tails)
    write_csv(path / "shap_tail_context_groups.csv", context_group_rows)

DCA_THRESHOLDS = [.01, .02, .03, .05, .075, .10, .15, .20, .25, .30, .40]


def sample_indices(n: int, groups: Optional[np.ndarray], count: int,
                   seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    if groups is None:
        return [rng.integers(0, n, size=n, dtype=np.int32) for _ in range(count)]
    group_keys = np.unique(groups)
    members = [np.flatnonzero(groups == g).astype(np.int32) for g in group_keys]
    return [np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
            for _ in range(count)]


def interval(values: np.ndarray, confidence: float) -> tuple[float, float, int]:
    values = np.asarray(values, float)
    valid = values[np.isfinite(values)]

    if len(values) == 0 or len(valid) < max(20, math.ceil(.90 * len(values))):
        return np.nan, np.nan, len(valid)
    alpha = 1-confidence
    return float(np.quantile(valid, alpha/2)), float(np.quantile(valid, 1-alpha/2)), len(valid)


def all_repeat_mean(values: list[float]) -> float:
    a = np.asarray(values, float)
    return float(a.mean()) if np.isfinite(a).all() else np.nan


def dca_vector(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    t = np.array(DCA_THRESHOLDS)
    flagged = p[:, None] >= t[None, :]
    tp = ((y[:, None] == 1) & flagged).sum(axis=0)
    fp = ((y[:, None] == 0) & flagged).sum(axis=0)
    net = tp/len(y) - fp/len(y) * t/(1-t)
    treat_all = y.mean() - (1-y.mean()) * t/(1-t)
    return np.column_stack([net, treat_all, net-treat_all])


def choose_bootstrap_groups(reference: pd.DataFrame, scheme: str,
                            cfg: Config) -> tuple[Optional[np.ndarray], str, str]:
    unit = cfg.bootstrap_unit
    if unit == "auto":
        if scheme in {"branch", "surgeon"}:
            unit = scheme
        elif cfg.surgeon_col:
            unit = "surgeon"
        elif cfg.branch_col:
            unit = "branch"
        else:
            unit = "patient"
    if unit == "patient":
        return None, unit, "conditional_patient_bootstrap"
    if reference[unit].isna().any():
        return None, unit, "CI_unavailable_missing_group_ID"
    groups = reference[unit].astype(str).to_numpy()
    if len(np.unique(groups)) < cfg.min_bootstrap_groups:
        return groups, unit, "CI_suppressed_too_few_groups"
    return groups, unit, "conditional_cluster_bootstrap"


def summarize_predictions(predictions: pd.DataFrame, cfg: Config, run: Path) -> None:

    summary_rows, comparison_rows, dca_rows, repeat_rows, calibration_rows = [], [], [], [], []
    for key, group in predictions.groupby(["cohort", "scheme", "outcome"], sort=False):
        cohort, scheme, outcome = key
        meta = {"cohort": cohort, "scheme": scheme, "outcome": outcome}
        repeats = sorted(group["repeat"].unique())
        methods: dict[tuple[str, str, str], dict[int, pd.DataFrame]] = {}
        for (family, strategy), method in group.groupby(["family", "strategy"], sort=False):
            for channel in ["raw", "calibrated"]:
                indexed = {}
                for rep in repeats:
                    subset = method.loc[method["repeat"] == rep].sort_values("case").reset_index(drop=True)
                    if subset.empty or subset.case.duplicated().any():
                        raise ValueError("각 반복에서 각 환자의 외부 평가 예측은 정확히 한 개여야 합니다.")
                    indexed[int(rep)] = subset
                methods[(family, strategy, channel)] = indexed
        first = next(iter(methods.values()))
        ref = first[int(repeats[0])]
        case_list, y = ref.case.tolist(), ref.y_true.to_numpy(int)
        for byrepeat in methods.values():
            for table in byrepeat.values():
                if table.case.tolist() != case_list or not np.array_equal(table.y_true.to_numpy(int), y):
                    raise ValueError("paired 비교에 필요한 동일 환자/정답 정렬이 일치하지 않습니다.")
        cluster, unit, ci_kind = choose_bootstrap_groups(ref, scheme, cfg)
        allowed = ci_kind in {"conditional_patient_bootstrap", "conditional_cluster_bootstrap"}
        boot_indices = sample_indices(len(y), cluster, cfg.n_bootstrap,
                                      seed_for(cfg.seed, "bootstrap", *key)) if allowed else []
        bootstrap_total = len(methods) * len(boot_indices)
        bootstrap_progress = ProgressETA(
            f"bootstrap {cohort}/{scheme}/{outcome}",
            bootstrap_total,
            log_every=max(1, bootstrap_total // 20) if bootstrap_total else 1,
        )
        if not allowed:
            LOG.warning("%s: %s. 그룹별 성능은 유지하지만 CI는 NA입니다.", key, ci_kind)
        ci_meta = {"n_unique_patients": len(y), "n_events": int(y.sum()), "n_repeats": len(repeats),
                   "resampling_unit": unit, "ci_method": ci_kind,
                   "confidence_level": cfg.confidence,
                   "ci_scope": "conditional_on_saved_predictions_not_full_training_procedure",
                   "multiplicity": "unadjusted_exploratory"}
        boot_results = {}
        point_results = {}
        for method_key, byrepeat in methods.items():
            family, strategy, channel = method_key
            mmeta = {**meta, "family": family, "strategy": strategy, "channel": channel}
            if len(boot_indices):
                LOG.info(
                    "bootstrap 조건 시작: outcome=%s family=%s strategy=%s channel=%s, %s회",
                    outcome, family, strategy, channel, len(boot_indices),
                )
            probas = [table[f"p_{channel}"].to_numpy(float) for table in byrepeat.values()]
            decisions = [table[f"pred_{channel}"].to_numpy(int) for table in byrepeat.values()]
            points = [metrics(y, p, d) for p, d in zip(probas, decisions)]
            means = {m: all_repeat_mean([point[m] for point in points]) for m in METRICS_FOR_CI}
            point_results[method_key] = means
            for rep, table, point in zip(repeats, byrepeat.values(), points):
                p = table[f"p_{channel}"].to_numpy(float)
                diagnostics = calibration_diagnostics(y, p)
                base_brier = float(brier_score_loss(y, table.train_prevalence.to_numpy(float)))
                repeat_rows.append({**mmeta, "repeat": rep, **point, **diagnostics,
                                    "train_prevalence_baseline_brier": base_brier,
                                    "no_skill_ap_reference": float(y.mean()),
                                    "note": "actual_OOF_counts_not_averaged_confusion_cells"})
                bins = calibration_bins(y, p)
                for record in bins.to_dict("records"):
                    calibration_rows.append({**mmeta, "repeat": rep, **record})
            arrays = {m: np.full(len(boot_indices), np.nan) for m in METRICS_FOR_CI}
            dca_boot = np.full((len(boot_indices), len(DCA_THRESHOLDS), 3), np.nan)
            for b, ix in enumerate(boot_indices):

                per_rep = [metrics(y[ix], p[ix], d[ix]) for p, d in zip(probas, decisions)]
                for m in METRICS_FOR_CI:
                    arrays[m][b] = all_repeat_mean([a[m] for a in per_rep])
                if channel == "calibrated":
                    dca_boot[b] = np.mean([dca_vector(y[ix], p[ix]) for p in probas], axis=0)
                bootstrap_progress.update(
                    detail=f"{family}/{strategy}/{channel}, {b+1}/{len(boot_indices)}"
                )
            boot_results[method_key] = arrays
            for m in METRICS_FOR_CI:
                low, high, valid = interval(arrays[m], cfg.confidence)
                vals = np.asarray([a[m] for a in points], float)
                summary_rows.append({**mmeta, **ci_meta, "metric": m, "estimate": means[m],
                                     "ci_low": low, "ci_high": high,
                                     "bootstrap_requested": cfg.n_bootstrap,
                                     "bootstrap_valid": valid,
                                     "repeat_sd_descriptive": float(np.std(vals, ddof=1)) if len(vals)>1 else np.nan,
                                     "ci_available": bool(np.isfinite(low) and np.isfinite(high))})
            if channel == "calibrated":
                point_dca = np.mean([dca_vector(y, p) for p in probas], axis=0)
                for j, t in enumerate(DCA_THRESHOLDS):
                    row = {**mmeta, **ci_meta, "threshold": t, "treat_none_net_benefit": 0.}
                    for c, name in enumerate(["net_benefit", "treat_all_net_benefit", "delta_vs_treat_all"]):
                        lo, hi, valid = interval(dca_boot[:, j, c], cfg.confidence)
                        row.update({name: point_dca[j, c], name+"_ci_low": lo,
                                    name+"_ci_high": hi, name+"_bootstrap_valid": valid})
                    dca_rows.append(row)

        LOG.info("paired 비교 시작: %s/%s/%s", cohort, scheme, outcome)
        pairs = []
        for strategy in cfg.strategies:
            for channel in ["raw", "calibrated"]:
                for a, b in [("Motion", "Patient"), ("Combined", "Patient"), ("Augmented", "Combined")]:
                    pairs.append(((a, strategy, channel), (b, strategy, channel), "feature_family"))
        for family in cfg.families:
            for strategy in [s for s in cfg.strategies if s != "none"]:
                for channel in ["raw", "calibrated"]:
                    pairs.append(((family, strategy, channel), (family, "none", channel), "imbalance_ablation"))
        for left, right, kind in pairs:
            if left not in methods or right not in methods:
                continue
            for m in ["auroc", "auprc_ap", "brier"]:
                delta = boot_results[left][m] - boot_results[right][m]
                low, high, valid = interval(delta, cfg.confidence)
                comparison_rows.append({**meta, **ci_meta, "comparison_kind": kind,
                                        "left_model": "/".join(left), "right_model": "/".join(right),
                                        "metric": m, "difference_left_minus_right": point_results[left][m]-point_results[right][m],
                                        "ci_low": low, "ci_high": high, "bootstrap_valid": valid,
                                        "favorable_direction": "negative" if m == "brier" else "positive",
                                        "automatic_superiority_claim": False})
        LOG.info("paired 비교 완료: %s/%s/%s", cohort, scheme, outcome)

        write_csv(run / "performance_summary.csv", summary_rows)
        write_csv(run / "paired_comparisons.csv", comparison_rows)
        write_csv(run / "repeat_metrics.csv", repeat_rows)
        write_csv(run / "calibration" / "calibration_bins.csv", calibration_rows)
        write_csv(run / "clinical" / "decision_curves_summary.csv", dca_rows)


def make_plots(predictions: pd.DataFrame, run: Path) -> None:

    for key, group in predictions[predictions["repeat"] == 0].groupby(
            ["cohort", "scheme", "outcome", "strategy"], sort=False):
        cohort, scheme, outcome, strategy = key
        directory = run / "plots" / safe_name("__".join(key))
        directory.mkdir(parents=True, exist_ok=True)
        title = f"{LABEL_MAP[outcome]}, {scheme}, {strategy}\nFirst repeat, held-out predictions"
        for channel in ["raw", "calibrated"]:
            plt.figure(figsize=(8, 7))
            for family, part in group.groupby("family", sort=False):
                y, p = part.y_true.to_numpy(int), part[f"p_{channel}"].to_numpy(float)
                if len(np.unique(y)) == 2:
                    fpr, tpr, _ = roc_curve(y, p)
                    plt.plot(fpr, tpr, label=f"{family}: {roc_auc_score(y,p):.3f}")
            plt.plot([0, 1], [0, 1], linestyle="--", label="No discrimination")
            plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
            plt.title(title + f"\n{channel}", fontsize=11); plt.legend()
            plt.tight_layout(); plt.savefig(directory / f"ROC_{channel}.png", dpi=180); plt.close()
            plt.figure(figsize=(8, 7))
            prevalence = float(group.groupby("case").y_true.first().mean())
            for family, part in group.groupby("family", sort=False):
                y, p = part.y_true.to_numpy(int), part[f"p_{channel}"].to_numpy(float)
                if len(np.unique(y)) == 2:
                    precision, recall, _ = precision_recall_curve(y, p)
                    plt.step(recall, precision, where="post", label=f"{family}: AP {average_precision_score(y,p):.3f}")
            plt.axhline(prevalence, linestyle="--", label=f"Prevalence: {prevalence:.3f}")
            plt.xlabel("Recall"); plt.ylabel("Precision")
            plt.title(title + f"\n{channel}", fontsize=11); plt.legend()
            plt.tight_layout(); plt.savefig(directory / f"PR_{channel}.png", dpi=180); plt.close()
            plt.figure(figsize=(8, 7))
            for family, part in group.groupby("family", sort=False):
                bins = calibration_bins(part.y_true.to_numpy(int), part[f"p_{channel}"].to_numpy(float))
                plt.plot(bins.predicted_mean, bins.observed_fraction, marker="o", label=family)
            plt.plot([0, 1], [0, 1], linestyle="--", label="Ideal calibration")
            plt.xlim(0, 1); plt.ylim(0, 1)
            plt.xlabel("Mean predicted probability"); plt.ylabel("Observed fraction")
            plt.title(title + f"\n{channel}", fontsize=11); plt.legend()
            plt.tight_layout(); plt.savefig(directory / f"calibration_{channel}.png", dpi=180); plt.close()
        plt.figure(figsize=(8, 7))
        for family, part in group.groupby("family", sort=False):
            table = clinical_table(part.y_true.to_numpy(int), part.p_calibrated.to_numpy(float), DCA_THRESHOLDS)
            plt.plot(table.threshold, table.net_benefit, label=family)
        plt.plot(table.threshold, table.treat_all_net_benefit, linestyle="--", label="Treat all")
        plt.axhline(0, linestyle=":", label="Treat none")
        plt.xlabel("Decision threshold"); plt.ylabel("Net benefit")
        plt.title(title + "\nCalibrated scores; assumed decisions, not treatment effects", fontsize=10)
        plt.legend(); plt.tight_layout(); plt.savefig(directory / "decision_curve.png", dpi=180); plt.close()
        for family, part in group.groupby("family", sort=False):
            cm = confusion_matrix(part.y_true, part.pred_calibrated, labels=[0,1])
            plt.figure(figsize=(6, 6))
            plt.imshow(cm)
            for r in range(2):
                for c in range(2):
                    plt.text(c, r, str(int(cm[r,c])), ha="center", va="center", fontsize=18,
                             bbox={"facecolor": "white", "alpha": .8})
            plt.xticks([0,1],["Negative","Positive"]); plt.yticks([0,1],["Negative","Positive"])
            plt.xlabel("Predicted"); plt.ylabel("Actual")
            plt.title(f"{family}: {LABEL_MAP[outcome]}\nActual counts, training-selected thresholds", fontsize=11)
            plt.tight_layout(); plt.savefig(directory / f"confusion_{family}.png", dpi=180); plt.close()

REVIEWER_MAP = [
    ("E5, E6, AW8b, BS-m1, BS-m5, BS-m6", "audit_and_load / cohort_flow", "입력 CSV의 분모, 변수별 결측, 실제 사건 수, 나이 통계", "CSV 이전의 모집/제외 기록은 별도 필요"),
    ("E8", "FoldPreprocessor / missingness", "fold 내부 median 및 iterative 민감도 옵션", "MCAR 검정과 정식 MI pooling은 미구현, 원결측 자료 필요"),
    ("E9, G4, AW1, BS-M1", "summarize_predictions", "환자/그룹 paired bootstrap, 차이 CI, 반복 SD 분리", "조건부 구간, 전체 개발 절차의 CI 아님"),
    ("G1, AW2a, AW2b, AW2c, BS-M8", "numeric_audit / quality CSV", "수치 범위와 품질 비율 점검", "원좌표 계산, detector 지표, 3D 참조오차, 반복 측정은 별도 필요"),
    ("AW2d, AW6", "make_cohorts / associations / SHAP tails", "고정된 품질 기준 제외 민감도, 양쪽 꼬리 분포", "품질 기준의 사전 근거와 실제 기록 품질 자료 필요"),
    ("G2, G3, AW3, BS-M6, BS-M10", "outer_splits / inner_splits", "LOBO, LOSO, 내부 grouped CV, 그룹 중복 확인", "실제 지점/술자 열 필요, 독립 기관 검증 아님"),
    ("AW3, AW5d, AW9, BS-m3", "build_feature_sets", "sex, 확인된 site, 모형군 입력 합집합, 변수 사전", "술자 경력/수술 범위/관리 변수는 확인 후 추가"),
    ("AW4, BS-M7", "ultrasound_timing / feature_dictionary", "시점 미확인 변수 기본 제외, 정의/시점 확인표", "결과 판정, 눈가림, 추적 탈락 사유는 임상 기록 필요"),
    ("AW1, BS-M2, BS-m9", "outer_splits / fit_bundle", "실제 반복 CV와 validation 예측, 분할 manifest", "기존 test 사용 이력을 없애거나 새 외부자료로 만들지 않음"),
    ("AW5f, AW7, BS-M3", "fit_base / augment_training_only", "학습 내부 증강, 4조건 비교, 범주형 SMOTENC", "stack은 수정된 비교 조건, 과거 계산의 정확한 재현 아님"),
    ("AW7, AW9", "select_parameters", "학습 내부 depth/leaf 튜닝 (--tune)", "사건/그룹 부족 시 고정 설정 fallback을 기록"),
    ("AW4, BS-M4", "fit_bundle / calibration_diagnostics", "학습 OOF sigmoid, raw/calibrated Brier와 보정도", "희소사건의 slope는 불안정할 수 있음"),
    ("AW4, BS-M5", "choose_threshold", "학습 OOF로 임계값 선정, 외부 평가 전에 고정", "과거 a priori 근거를 증명하지 않음, 임상 기준 합의 필요"),
    ("G5, AW4, BS-M4", "clinical_table / decision curves", "표시 비율, PPV, 검출당 표시 수, treat-all/none 비교", "예방 처치 효과 및 실제 임상 유용성 실험 아님"),
    ("G5, AW6, BS-M9", "compute_shap / plot_shap_summary", "첫 반복 전체 OOF의 SHAP, fold 순위, 비선형 표시", "보정 전 RF 설명, 인과/임상 임계값 아님"),
    ("AW5b, AW5g, BS-m6, BS-m9", "metrics / manifests", "정수 CM와 정확한 분모, outcome별 층화 기록", "평균 CM 셀을 int로 절삭하지 않음"),
    ("AW9", "make_derived_frame", "raw-scale BMI, 비율 및 상호작용, 확정된 입력 목록", "물리 단위와 집계식 자체는 CSV로 검증 불가"),
    ("BS-m7", "metrics / PR plot", "Average Precision 명시, 발생률 기준선", "AP와 trapezoidal AUPRC를 혼용하지 않음"),
    ("E13, AW5c, BS-m8", "make_plots", "큰 글꼴, 개별 그림, 실제 평가 counts", "원고 그림 번호/문구는 별도 수정"),
    ("AW8c, BS-m10", "main / self_test", "source snapshot, 설정/환경/hash, 합성 점검", "환자 자료/모델의 공개 가능 여부는 별도 승인"),
    ("E1~E4, E7, E10~E12, E14~E15, AW8a/d/e/f, BS-m11/m12", "limitations.md", "코드 밖에서 답해야 할 항목을 기록", "제목/원고/IRB/동의/자금/AI/CRediT/체크리스트 수작업 필요"),
]


def write_notes(run: Path, cfg: Config) -> None:
    write_csv(run / "reviewer_mapping.csv", [{"reviewer_comments": a, "code_section": b,
                                            "implemented": c, "still_required": d}
                                           for a, b, c, d in REVIEWER_MAP])



def configure_logging(run: Path) -> None:
    LOG.setLevel(logging.INFO)
    for handler in list(LOG.handlers):
        handler.close()
        LOG.removeHandler(handler)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler(sys.stdout); stream.setFormatter(formatter)
    file = logging.FileHandler(run / "run.log", encoding="utf-8"); file.setFormatter(formatter)
    LOG.addHandler(stream); LOG.addHandler(file)


def run_analysis(cfg: Config, audit_only: bool = False) -> Path:

    if cfg.outdir:
        run = Path(cfg.outdir).expanduser().resolve()
        if run.exists() and any(run.iterdir()):
            raise FileExistsError(f"출력 폴더가 비어 있지 않습니다: {run}. 새 경로를 지정하세요.")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run = Path(cfg.csv).expanduser().resolve().parent / f"jmir_r1_{timestamp}"
    run.mkdir(parents=True, exist_ok=True)
    configure_logging(run)
    LOG.info("%s 시작. 데이터/학습/평가 provenance를 %s에 저장합니다.", VERSION, run)

    write_json(run / "run_config.json", asdict(cfg))
    source = Path(__file__).resolve()
    shutil.copy2(source, run / "source_snapshot.py")
    versions = {"python": sys.version, "platform": platform.platform(), "version": VERSION,
                "source_sha256": file_hash(source), "started_utc": datetime.now(timezone.utc).isoformat()}
    for name in ["numpy", "pandas", "scipy", "scikit-learn", "imbalanced-learn", "matplotlib", "joblib", "shap"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not_installed"
    write_json(run / "environment.json", versions)
    write_notes(run, cfg)
    data, features, categorical = audit_and_load(cfg, run)
    if audit_only:
        write_json(run / "completion.json", {"status": "audit_only_no_models_trained"})
        return run
    cohorts = make_cohorts(data, cfg)
    flow_rows, split_count_rows, unavailable, all_predictions, fold_metric_rows = [], [], [], [], []
    clinical_rows, legacy_rows, shap_failures = [], [], []
    job_count = 0
    for cohort, cohort_df in cohorts.items():
        for scheme in cfg.schemes:
            group_col = "_branch" if scheme == "branch" else ("_surgeon" if scheme == "surgeon" else None)
            if scheme == "branch" and not cfg.branch_col or scheme == "surgeon" and not cfg.surgeon_col:
                unavailable.append({"cohort": cohort, "scheme": scheme, "reason": "required_group_column_not_configured"})
                LOG.warning("%s 검증 생략: 실제 그룹 열이 지정되지 않았습니다.", scheme)
                continue
            for outcome in cfg.outcomes:
                observed = cohort_df[outcome].notna()
                group_known = cohort_df[group_col].notna() if group_col else pd.Series(True, index=cohort_df.index)
                eligible = cohort_df.loc[observed & group_known].copy().reset_index(drop=True)
                y = eligible[outcome].to_numpy(dtype=int)
                flow_rows.append({"cohort": cohort, "scheme": scheme, "outcome": outcome,
                                  "input_csv_n": len(data), "cohort_before_outcome_selection_n": len(cohort_df),
                                  "quality_excluded_n": len(data)-len(cohort_df),
                                  "missing_outcome_n": int((~observed).sum()),
                                  "missing_group_among_observed_n": int((observed & ~group_known).sum()),
                                  "eligible_n": len(eligible), "eligible_events": int(y.sum())})
                write_csv(run / "cohort_flow.csv", flow_rows)
                meta = {"cohort": cohort, "scheme": scheme, "outcome": outcome}
                stem = safe_name(f"{cohort}__{scheme}__{outcome}")
                try:
                    splits = outer_splits(eligible, y, scheme, cfg)

                    for repeat, f, tr, te in splits:
                        g = eligible.iloc[tr][group_col].astype(str).to_numpy() if group_col else None
                        inner_splits(y[tr], g, cfg.inner_folds, seed_for(cfg.seed, "check", repeat, f))
                except AnalysisUnavailable as exc:
                    unavailable.append({**meta, "reason": str(exc), "action": "whole_scheme_outcome_unavailable_no_folds_dropped"})
                    LOG.warning("%s 계산 불가: %s", stem, exc)
                    continue
                assignments, split_counts = make_outer_manifest(eligible, y, splits, group_col)
                write_csv(run / "splits" / f"{stem}_assignments.csv", assignments)
                for record in split_counts.to_dict("records"):
                    split_count_rows.append({**meta, **record})
                write_csv(run / "outer_split_counts.csv", split_count_rows)
                LOG.info("%s: n=%s, events=%s, outer fits per condition=%s", stem, len(y), int(y.sum()), len(splits))
                condition_fit_total = len(splits) * len(cfg.families) * len(cfg.strategies)
                fit_progress = ProgressETA(
                    f"모델 학습 {stem}", condition_fit_total, log_every=1
                )
                for family in cfg.families:
                    for strategy in cfg.strategies:
                        method_meta = {**meta, "family": family, "strategy": strategy}
                        job_stem = f"{stem}__{family}__{strategy}"
                        parts, shap_parts = [], []
                        for repeat, f, tr, te in splits:
                            model_seed = seed_for(cfg.seed, scheme, outcome, cohort, repeat, f)
                            X_train, X_test = eligible.iloc[tr], eligible.iloc[te]
                            groups = X_train[group_col].astype(str).to_numpy() if group_col else None
                            audit_dir = run / "splits" / job_stem / f"repeat_{repeat}_fold_{f}"
                            LOG.info(
                                "fit 시작 [%s/%s]: %s repeat=%s fold=%s",
                                fit_progress.done + 1, fit_progress.total, job_stem, repeat, f,
                            )
                            bundle = fit_bundle(X_train, y[tr], groups, features[family], categorical,
                                                strategy, cfg, model_seed, audit_dir)
                            p_raw, p_cal = bundle.predict(X_test)
                            assert np.isfinite(p_raw).all() and np.isfinite(p_cal).all()
                            frame = pd.DataFrame({
                                **{k: [v]*len(te) for k,v in method_meta.items()},
                                "case": X_test["_case"].to_numpy(), "repeat": repeat, "fold": f,
                                "branch": X_test["_branch"].to_numpy(), "surgeon": X_test["_surgeon"].to_numpy(),
                                "y_true": y[te], "p_raw": p_raw, "p_calibrated": p_cal,
                                "threshold_raw": bundle.raw_threshold,
                                "threshold_calibrated": bundle.calibrated_threshold,
                                "pred_raw": (p_raw >= bundle.raw_threshold).astype(int),
                                "pred_calibrated": (p_cal >= bundle.calibrated_threshold).astype(int),
                                "train_prevalence": float(y[tr].mean()),
                            })
                            parts.append(frame)
                            for channel in ["raw", "calibrated"]:
                                m = metrics(frame.y_true.to_numpy(int), frame[f"p_{channel}"].to_numpy(float),
                                            frame[f"pred_{channel}"].to_numpy(int))
                                fold_metric_rows.append({**method_meta, "repeat": repeat, "fold": f,
                                                         "channel": channel, **m,
                                                         "threshold": bundle.raw_threshold if channel == "raw" else bundle.calibrated_threshold})
                            if cfg.save_models:
                                model_path = run / "models" / job_stem / f"repeat_{repeat}_fold_{f}.joblib"
                                model_path.parent.mkdir(parents=True, exist_ok=True)
                                joblib.dump({"version": VERSION, "bundle": bundle, "model_context": method_meta,
                                             "training_cases": X_train["_case"].tolist(),
                                             "evaluation_cases": X_test["_case"].tolist(),
                                             "not_a_full_data_deployment_model": True}, model_path, compress=3)
                            if cfg.shap and repeat == 0 and family == cfg.shap_family and strategy == cfg.shap_strategy:
                                spath = run / "shap" / job_stem / f"fold_{f}"
                                try:
                                    values = compute_shap(bundle, X_test, repeat, f, spath)
                                    if values is not None:
                                        shap_parts.append(values)
                                except Exception as exc:
                                    shap_failures.append({**method_meta, "fold": f, "error": repr(exc)})
                                    LOG.warning("SHAP 계산 실패, 실패를 기록하고 평가 결과는 보관: %s", exc)
                            job_count += 1
                            fit_progress.update(
                                detail=f"{job_stem} repeat={repeat} fold={f}; 전체 누적 fit={job_count}"
                            )
                        result = pd.concat(parts, ignore_index=True)
                        all_predictions.append(result)
                        write_csv(run / "predictions" / f"{job_stem}.csv", result)
                        write_csv(run / "fold_metrics.csv", fold_metric_rows)
                        for repeat, perrepeat in result.groupby("repeat", sort=False):
                            ct = clinical_table(perrepeat.y_true.to_numpy(int), perrepeat.p_calibrated.to_numpy(float),
                                                DCA_THRESHOLDS)
                            clinical_rows.extend({**method_meta, "repeat": repeat, **row}
                                                 for row in ct.to_dict("records"))

                            legacy_t = LEGACY_THRESHOLDS[outcome]
                            lm = metrics(perrepeat.y_true.to_numpy(int), perrepeat.p_raw.to_numpy(float),
                                         (perrepeat.p_raw.to_numpy(float) >= legacy_t).astype(int))
                            legacy_rows.append({**method_meta, "repeat": repeat, "legacy_score_threshold": legacy_t,
                                                **lm, "note": "not_claimed_a_priori_or_calibrated_risk_threshold"})
                        if shap_parts:
                            full_shap = pd.concat(shap_parts, ignore_index=True)
                            n_expected = result.loc[result["repeat"] == 0, "case"].nunique()
                            n_shap = full_shap.case.nunique()
                            spath = run / "shap" / job_stem
                            write_json(spath / "coverage.json", {"expected_cases": n_expected, "shap_cases": n_shap,
                                                                  "all_cases_covered": n_shap == n_expected})
                            context_cols = [c for c in ["operation_time", "totalStrokeCount", "BEFORE_SIZE",
                                                          "BEFORE_WEIGHT", "HIGHT", "_branch", "_surgeon"] +
                                            QUALITY_COLUMNS + ([cfg.site_col] if cfg.site_col else [])
                                            if c in eligible]
                            context = eligible[["_case", outcome] + context_cols].rename(
                                columns={"_case": "case", outcome: "context_outcome",
                                         **{c: "context_" + c for c in context_cols}})
                            full_shap = full_shap.merge(context, on="case", how="left", validate="many_to_one")
                            full_shap.attrs["categorical_context"] = ["context_" + c for c in
                                                                      ["_branch", "_surgeon"] +
                                                                      ([cfg.site_col] if cfg.site_col else [])]
                            full_shap.to_csv(spath / "all_oof_shap_with_context.csv.gz", index=False,
                                             encoding="utf-8-sig", compression="gzip")
                            if n_shap == n_expected and cfg.plots:
                                plot_shap_summary(full_shap, spath, LABEL_MAP[outcome] + f", {scheme}, {strategy}")

    write_csv(run / "clinical" / "threshold_operating_points.csv", clinical_rows)
    write_csv(run / "clinical" / "legacy_score_thresholds_descriptive.csv", legacy_rows)
    write_json(run / "unavailable_analyses.json", unavailable)
    write_json(run / "shap_failures.json", shap_failures)
    if not all_predictions:
        write_json(run / "completion.json", {"status": "no_evaluable_analysis", "reasons": unavailable})
        raise AnalysisUnavailable("평가 가능한 분석이 없습니다. unavailable_analyses.json을 확인하세요.")
    predictions = pd.concat(all_predictions, ignore_index=True)
    LOG.info("환자별 예측 저장 완료. 조건부 bootstrap 및 paired 비교를 계산합니다.")
    summarize_predictions(predictions, cfg, run)
    if cfg.plots:
        LOG.info("그림 생성 시작")
        plot_started = time.perf_counter()
        make_plots(predictions, run)
        LOG.info("그림 생성 완료 | 소요 %s", format_duration(time.perf_counter() - plot_started))
    status = "completed_with_unavailable_items" if unavailable or shap_failures else "completed"
    write_json(run / "completion.json", {"status": status, "outer_fits_completed": job_count,
                                         "prediction_rows": len(predictions), "pilot": cfg.pilot,
                                         "finished_utc": datetime.now(timezone.utc).isoformat(),
                                         "not_equivalent_to_all_reviewer_items_resolved": True})
    LOG.info("분석 종료: %s", run)
    return run


def synthetic_data(n: int = 180, seed: int = 417) -> pd.DataFrame:

    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "PSENTRY": [f"synthetic_{i:05d}" for i in range(n)],
        "branch_id": [f"B{i % 6}" for i in range(n)],
        "surgeon_id": [f"S{i % 12}" for i in range(n)],
        "surgical_site": rng.choice(["arm", "abdomen", "hip"], n),
        "age": rng.uniform(22, 65, n), "sex": rng.choice(["F", "M"], n),
        "fat_ext_volume": rng.uniform(500, 3500, n), "BEFORE_SIZE": rng.uniform(40, 100, n),
        "BEFORE_WEIGHT": rng.uniform(45, 100, n), "HIGHT": rng.uniform(150, 185, n),
        "Supersonic_Waves_Size": rng.uniform(1, 10, n), "operation_time": rng.uniform(35, 150, n),
        "totalStrokeCount": rng.integers(1000, 12000, n), "totalStrokeVelocity": rng.uniform(1, 10, n),
        "cohesion_average": rng.lognormal(1, .7, n), "cohesion_max": rng.lognormal(2.5, .5, n),
        "Cohesion_out": rng.integers(0, 100, n), "Separation_average": rng.uniform(50, 300, n),
        "Separation_max": rng.uniform(350, 1500, n), "Separation_min": rng.uniform(1, 40, n),
        "Separation_out": rng.integers(0, 8, n), "Cohesion_min": rng.uniform(.1, 1, n),
        "cohesion_count": rng.integers(100, 1000, n), "Separation_count": rng.integers(2, 15, n),
        "detection_rate": rng.uniform(.7, 1, n), "fallback_rate": rng.uniform(0, .3, n),
        "interpolated_fraction": rng.uniform(0, .2, n), "rejected_fraction": rng.uniform(0, .1, n),
    })
    for i, label in enumerate(LABEL_MAP):

        target = ((np.arange(n)//6 + i) % 5 == 0).astype(int)
        frame[label] = target
    frame.loc[[3, 11, 17], "BEFORE_WEIGHT"] = np.nan
    frame.loc[[7, 21], "sex"] = np.nan
    return frame


def self_test() -> None:

    with tempfile.TemporaryDirectory(prefix="jmir_r1_selftest_") as temp:
        root = Path(temp)
        csv_path = root / "synthetic_ONLY.csv"
        data = synthetic_data()
        data.to_csv(csv_path, index=False)
        cfg = Config(csv=str(csv_path), id_col="PSENTRY", site_col="surgical_site",
                     branch_col="branch_id", surgeon_col="surgeon_id", n_estimators=8,
                     inner_folds=2, folds=3, repeats=2, n_jobs=1, n_bootstrap=25,
                     outcomes=["Rad_4week"], strategies=["none", "weight"],
                     families=list(FAMILIES), tune=True, shap=False, plots=False,
                     pilot=True, bootstrap_unit="patient")
        test_audit = root / "audit"; test_audit.mkdir()
        loaded, sets, cats = audit_and_load(cfg, test_audit)

        raw = make_derived_frame(loaded, sets["Augmented"], True)
        idx = loaded.BEFORE_WEIGHT.first_valid_index()
        assert np.isclose(raw.loc[idx, "BMI"], loaded.loc[idx,"BEFORE_WEIGHT"] / (loaded.loc[idx,"HIGHT"]/100)**2)
        assert set(sets["Combined"]) == set(sets["Patient"]) | set(sets["Motion"])
        assert "Supersonic_Waves_Size" not in sets["Patient"]
        assert category_text(1) == category_text(1.0)
        assert category_text(np.nan) == "__MISSING__"
        y = loaded.Rad_4week.to_numpy(int)

        before = loaded.copy(deep=True)
        augmented, ay = augment_training_only(loaded, y, 42)
        pd.testing.assert_frame_equal(loaded, before)
        assert len(augmented) == len(y) + y.sum()
        assert len(ay) == len(augmented)
        assert set(augmented.sex.dropna()).issubset(set(loaded.sex.dropna()))

        tr, te = train_test_split(np.arange(len(y)), test_size=.3, stratify=y, random_state=42)
        params = {"max_depth": 5, "min_samples_leaf": 2}
        for family in FAMILIES:
            for strategy in STRATEGIES:
                model = fit_base(loaded.iloc[tr], y[tr], sets[family], cats, strategy, params, cfg, 42)
                center = model.processor.scaler_.mean_.copy()
                p = model.predict_proba(loaded.iloc[te])
                assert len(p) == len(te) and ((p >= 0) & (p <= 1)).all()
                changed = loaded.iloc[te].copy(); changed["BEFORE_WEIGHT"] = 10000.
                model.predict_proba(changed)
                assert np.array_equal(center, model.processor.scaler_.mean_)

        groups = loaded._branch.to_numpy()
        for tr0, te0 in inner_splits(y, groups, 3, 42):
            assert not set(groups[tr0]) & set(groups[te0])

        m = metrics(np.zeros(10, int), np.full(10, .1), np.zeros(10, int))
        assert np.isnan(m["auroc"]) and np.isnan(m["auprc_ap"]) and m["tn"] == 10
        assert np.isclose(m["brier"], .01)

        cfg.outdir = str(root / "holdout")
        run = run_analysis(cfg)
        saved = pd.read_csv(run / "paired_comparisons.csv")
        assert len(saved) > 0 and saved.ci_low.notna().any()
        assert (pd.read_csv(run / "outer_split_counts.csv").evaluation_n == len(te)).all()

        cfg.outdir = str(root / "cv"); cfg.schemes = ["cv"]
        cfg.families = ["Patient", "Combined"]; cfg.strategies = ["none"]
        cfg.tune = False
        run = run_analysis(cfg)
        files = list((run / "predictions").glob("*Combined*none.csv"))
        saved = pd.read_csv(files[0])
        assert len(saved) == len(data) * cfg.repeats
        assert saved.groupby(["repeat", "case"]).size().eq(1).all()
        assert saved.groupby("repeat").case.nunique().eq(len(data)).all()

        cfg.outdir = str(root / "branch"); cfg.schemes = ["branch"]
        cfg.families = ["Combined"]; cfg.bootstrap_unit = "auto"
        run = run_analysis(cfg)
        score = pd.read_csv(run / "performance_summary.csv")
        assert score.ci_low.isna().all() and score.ci_method.eq("CI_suppressed_too_few_groups").all()

        bundle = fit_bundle(loaded.iloc[tr], y[tr], None, sets["Combined"], cats, "none", cfg, 52)
        target = root / "roundtrip.joblib"
        joblib.dump(bundle, target)
        recovered = joblib.load(target)
        assert np.allclose(bundle.predict(loaded.iloc[te])[1], recovered.predict(loaded.iloc[te])[1])

        for handler in list(LOG.handlers):
            handler.close()
            LOG.removeHandler(handler)
    print("SELF-TEST PASS: raw features, no fit on test, all model/ablation paths, nested tuning,")
    print("calibration/thresholds, repeated OOF coverage, grouped splits, conditional CI, model round trip.")



def predict_saved_model(model_path: str, csv: str, outdir: Optional[str]) -> Path:

    payload = joblib.load(model_path)  
   
    data, _ = normalize_columns(pd.read_csv(csv))
    bundle = payload["bundle"]
    raw, cal = bundle.predict(data)
    out = Path(outdir) if outdir else Path(csv).resolve().parent / f"jmir_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("예측 출력 폴더가 이미 존재하고 비어 있지 않습니다.")
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "predictions.csv", pd.DataFrame({
        "input_row": np.arange(len(data)), "p_raw": raw, "p_calibrated": cal,
        "threshold_raw": bundle.raw_threshold, "threshold_calibrated": bundle.calibrated_threshold,
        "pred_raw": (raw >= bundle.raw_threshold).astype(int),
        "pred_calibrated": (cal >= bundle.calibrated_threshold).astype(int),
    }))
    write_json(out / "prediction_manifest.json", {"model_sha256": file_hash(Path(model_path)),
                                                 "csv_sha256": file_hash(Path(csv)),
                                                 "clinical_validation_status": "not_established_by_this_script"})
    return out


def main() -> int:
    args = parse_args()
    if args.self_test:
        with threadpool_limits(limits=1):
            self_test()
        return 0
    if args.predict_model:
        out = predict_saved_model(args.predict_model, args.csv, args.outdir)
        print(f"예측 저장: {out}")
        return 0
    options = vars(args).copy()
    audit_only = options.pop("audit_only")
    options.pop("self_test"); options.pop("predict_model")
    cfg = Config(**options)
    for name in ["schemes", "families", "strategies", "outcomes", "extra_patient_cols", "extra_categorical_cols"]:
        setattr(cfg, name, list(dict.fromkeys(getattr(cfg, name))))
    if min(cfg.folds, cfg.inner_folds) < 2 or cfg.repeats < 1:
        raise ValueError("folds/inner-folds는 2 이상, repeats는 1 이상이어야 합니다.")
    if not (0 < cfg.test_size < 1 and 0 < cfg.confidence < 1 and 0 < cfg.target_recall <= 1):
        raise ValueError("test-size, confidence, target-recall 범위를 확인하세요.")
    if not 0 <= cfg.fixed_threshold <= 1 or cfg.n_bootstrap < 0:
        raise ValueError("threshold 또는 n-bootstrap 범위를 확인하세요.")
    if min(cfg.n_estimators, cfg.max_depth, cfg.min_samples_leaf) < 1 or cfg.n_jobs == 0:
        raise ValueError("RF 설정 값을 확인하세요.")
    if cfg.min_bootstrap_groups < 2:
        raise ValueError("min-bootstrap-groups는 2 이상이어야 합니다.")
    for threshold in [cfg.min_detection_rate, cfg.max_interpolated_fraction, cfg.max_rejected_fraction]:
        if threshold is not None and not 0 <= threshold <= 1:
            raise ValueError("기록 품질 기준은 0~1 비율이어야 합니다.")
    if cfg.pilot:
        cfg.n_estimators = 25
        cfg.families = ["Patient", "Combined"]
        cfg.strategies = ["none", "weight"]
        cfg.outcomes = ["Rad_4week"]
        cfg.repeats = 2
        cfg.folds = 3
        cfg.inner_folds = 2
        cfg.tune = False
        cfg.n_bootstrap = 30
        cfg.shap = False
        cfg.plots = False
    out = run_analysis(cfg, audit_only=audit_only)
    print(f"완료. 결과 폴더: {out}")
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("사용자가 실행을 중단했습니다. 저장된 완료 조건과 로그를 확인하세요.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        LOG.exception("실행 실패: %s", exc)
        raise SystemExit(1)

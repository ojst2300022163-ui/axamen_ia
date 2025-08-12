import os
import json
import pickle
from typing import Dict, Tuple, List

import numpy as np
import matplotlib

# Non-interactive backend for server environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import db
from .models import PatientCase

# Paths
BASE_DIR = "/workspace"
DATA_PATH = os.path.join(BASE_DIR, "heart.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
STATIC_DIR = os.path.join(BASE_DIR, "static")
METRICS_JSON = os.path.join(MODELS_DIR, "metrics.json")

FEATURE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]
TARGET_COLUMN = "target"


# -----------------------
# Data loading utilities
# -----------------------

def load_base_dataset() -> Dict[str, np.ndarray]:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        rows: List[List[float]] = []
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != len(header):
                continue
            rows.append([float(x) for x in parts])
    data = np.array(rows, dtype=float)
    col_to_idx = {name: i for i, name in enumerate(header)}
    X = data[:, [col_to_idx[c] for c in FEATURE_COLUMNS]]
    y = data[:, col_to_idx[TARGET_COLUMN]].astype(int)
    return {"X": X, "y": y}


def get_labeled_cases_from_db() -> Dict[str, np.ndarray]:
    rows = (
        db.session.query(PatientCase)
        .filter(PatientCase.target.isnot(None))
        .all()
    )
    if not rows:
        return {"X": np.zeros((0, len(FEATURE_COLUMNS))), "y": np.zeros((0,), dtype=int)}

    X_list: List[List[float]] = []
    y_list: List[int] = []
    for r in rows:
        X_list.append([
            r.age, r.sex, r.cp, r.trestbps, r.chol, r.fbs, r.restecg,
            r.thalach, r.exang, r.oldpeak, r.slope, r.ca, r.thal
        ])
        y_list.append(int(r.target))
    X = np.array(X_list, dtype=float)
    y = np.array(y_list, dtype=int)
    return {"X": X, "y": y}


def get_training_dataset() -> Dict[str, np.ndarray]:
    base = load_base_dataset()
    extra = get_labeled_cases_from_db()
    if extra["X"].shape[0] > 0:
        X = np.vstack([base["X"], extra["X"]])
        y = np.concatenate([base["y"], extra["y"]])
        return {"X": X, "y": y}
    return base


# -----------------------
# Simple ML models (NumPy)
# -----------------------

class LogisticRegressionGD:
    def __init__(self, learning_rate: float = 0.01, num_iter: int = 4000, l2: float = 0.0):
        self.learning_rate = learning_rate
        self.num_iter = num_iter
        self.l2 = l2
        self.weights: np.ndarray | None = None

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n_samples, n_features = X.shape
        Xb = np.hstack([np.ones((n_samples, 1)), X])
        self.weights = np.zeros(n_features + 1)
        for _ in range(self.num_iter):
            logits = Xb @ self.weights
            preds = self._sigmoid(logits)
            error = preds - y
            grad = (Xb.T @ error) / n_samples + self.l2 * np.r_[0.0, self.weights[1:]]
            self.weights -= self.learning_rate * grad

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.weights is not None
        n_samples = X.shape[0]
        Xb = np.hstack([np.ones((n_samples, 1)), X])
        logits = Xb @ self.weights
        return self._sigmoid(logits)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)


class KNNClassifier:
    def __init__(self, k: int = 5):
        self.k = k
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # Standardize for distance fairness
        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0) + 1e-8
        self.X_train = (X - self._mean) / self._std
        self.y_train = y.astype(int)

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        return (X - self._mean) / self._std

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = self._standardize(X)
        # Compute pairwise distances to training set
        # Efficient enough for small datasets
        dists = np.sqrt(((Xs[:, None, :] - self.X_train[None, :, :]) ** 2).sum(axis=2))
        knn_idx = np.argpartition(dists, self.k, axis=1)[:, : self.k]
        knn_labels = self.y_train[knn_idx]
        # Majority vote
        votes = knn_labels.sum(axis=1)
        preds = (votes >= (self.k / 2.0)).astype(int)
        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xs = self._standardize(X)
        dists = np.sqrt(((Xs[:, None, :] - self.X_train[None, :, :]) ** 2).sum(axis=2))
        knn_idx = np.argpartition(dists, self.k, axis=1)[:, : self.k]
        knn_labels = self.y_train[knn_idx]
        proba = knn_labels.mean(axis=1)
        return proba


# -----------------------
# Training and evaluation
# -----------------------

def _train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, seed: int = 42):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int(n * test_size)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    acc = float((y_true == y_pred).mean())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    cm = np.zeros((2, 2), dtype=int)
    for a, b in zip(y_true, y_pred):
        cm[a, b] += 1
    return {"accuracy": acc, "precision": precision, "recall": recall, "cm": cm}


def _save_confusion_matrix(cm: np.ndarray, title: str, filename: str) -> str:
    os.makedirs(os.path.join(STATIC_DIR, "metrics"), exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title(title)
    fig.tight_layout()
    out_path = os.path.join(STATIC_DIR, "metrics", filename)
    fig.savefig(out_path)
    plt.close(fig)
    return f"/static/metrics/{filename}"


def build_pipelines():
    # Return model instances
    return {
        "logreg": LogisticRegressionGD(learning_rate=0.05, num_iter=5000, l2=0.0),
        "knn": KNNClassifier(k=7),
    }


def train_and_persist_models(_unused=None) -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    data = get_training_dataset()
    X, y = data["X"], data["y"]
    X_train, X_test, y_train, y_test = _train_test_split(X, y, test_size=0.2, seed=42)

    models = build_pipelines()
    metrics: Dict[str, Dict] = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        m = _metrics(y_test, y_pred)

        # Save model
        model_path = os.path.join(MODELS_DIR, f"{name}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # Save confusion matrix figure
        cm_url = _save_confusion_matrix(m["cm"], f"Matriz de confusión - {name}", f"cm_{name}.png")

        metrics[name] = {
            "accuracy": float(m["accuracy"]),
            "precision": float(m["precision"]),
            "recall": float(m["recall"]),
            "confusion_matrix_image": cm_url,
        }

    # Save metrics json
    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def ensure_models_trained():
    required = [os.path.join(MODELS_DIR, f"{name}.pkl") for name in ["logreg", "knn"]]
    if not all(os.path.exists(p) for p in required):
        train_and_persist_models()


def load_model(name: str):
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    if not os.path.exists(path):
        ensure_models_trained()
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_with_model(model_name: str, features_dict: Dict[str, float]) -> Tuple[int, float]:
    model_map = {
        "logreg": "logreg",
        "knn": "knn",
        "lr": "logreg",
    }
    key = model_map.get(model_name, "logreg")
    model = load_model(key)
    X = build_patient_array_from_dict(features_dict)
    proba = None
    if hasattr(model, "predict_proba"):
        proba_arr = model.predict_proba(X)
        proba = float(proba_arr[0]) if np.ndim(proba_arr) == 1 else float(proba_arr[:, 0][0])
    else:
        # Fallback: try predict to get label; set 0.0/1.0 as pseudo probability
        label = int(model.predict(X)[0])
        return label, float(label)
    label = int(model.predict(X)[0])
    return label, float(proba)


def build_patient_array_from_dict(features_dict: Dict[str, float]) -> np.ndarray:
    row = [features_dict[col] for col in FEATURE_COLUMNS]
    return np.array([row], dtype=float)


def load_cached_metrics() -> Dict:
    if os.path.exists(METRICS_JSON):
        with open(METRICS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}
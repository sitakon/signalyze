"""
ML model training and evaluation on signals, PSD curves, or extracted features.
"""

import copy
from typing import Dict, List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from .features import extract_all_features, features_to_matrix
from .spectral import compute_welch

FeatureType = Literal["time", "spectral", "both", "psd", "raw"]
ModelType = Literal["rf", "svm", "lr", "gb", "knn"]

_MODEL_TEMPLATES: Dict[str, object] = {
    "rf":  RandomForestClassifier(n_estimators=100, random_state=42),
    "svm": SVC(kernel="rbf", probability=True, random_state=42),
    "lr":  LogisticRegression(max_iter=1000, random_state=42),
    "gb":  GradientBoostingClassifier(n_estimators=100, random_state=42),
    "knn": KNeighborsClassifier(n_neighbors=5),
}


# ---------------------------------------------------------------------------
# Feature matrix construction
# ---------------------------------------------------------------------------


def build_feature_matrix(
    signals: List[np.ndarray],
    fs: float,
    feature_type: FeatureType = "both",
    nperseg: Optional[int] = None,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build a 2-D feature matrix from a collection of signals.

    Parameters
    ----------
    signals : list of 1-D arrays
    fs : float
        Sampling frequency in Hz.
    feature_type : {'time', 'spectral', 'both', 'psd', 'raw'}
        Representation for each signal:

        - ``'time'``     — time-domain statistical features
        - ``'spectral'`` — spectral features derived from the PSD
        - ``'both'``     — time + spectral (default)
        - ``'psd'``      — raw Welch PSD curve, flattened per signal
        - ``'raw'``      — raw signal samples, flattened (all must be equal length)

    nperseg : int, optional
        Welch segment length. Used for ``'spectral'``, ``'both'``, ``'psd'``.
    bands : dict, optional
        Custom ``{name: (f_low, f_high)}`` bands for spectral features.

    Returns
    -------
    X : ndarray, shape (n_signals, n_features)
    feature_names : list of str
    """
    if feature_type == "raw":
        lengths = {len(s) for s in signals}
        if len(lengths) > 1:
            raise ValueError(
                "All signals must have the same length for feature_type='raw'"
            )
        X = np.stack([np.asarray(s, dtype=np.float64).ravel() for s in signals])
        return X, [f"sample_{i}" for i in range(X.shape[1])]

    if feature_type == "psd":
        psds = [compute_welch(s, fs, nperseg=nperseg)[1] for s in signals]
        min_len = min(len(p) for p in psds)
        X = np.stack([p[:min_len] for p in psds])
        ref_freqs = compute_welch(signals[0], fs, nperseg=nperseg)[0]
        return X, [f"psd_{f:.2f}Hz" for f in ref_freqs[:min_len]]

    include_time = feature_type in ("time", "both")
    include_spectral = feature_type in ("spectral", "both")
    feat_dicts = [
        extract_all_features(
            s, fs,
            nperseg=nperseg,
            bands=bands,
            include_time=include_time,
            include_spectral=include_spectral,
        )
        for s in signals
    ]
    return features_to_matrix(feat_dicts)


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


def build_model(model_type: ModelType = "rf") -> Pipeline:
    """
    Return a sklearn Pipeline: ``StandardScaler → classifier``.

    Parameters
    ----------
    model_type : {'rf', 'svm', 'lr', 'gb', 'knn'}
        - ``'rf'``  — Random Forest (100 trees)
        - ``'svm'`` — SVM with RBF kernel
        - ``'lr'``  — Logistic Regression
        - ``'gb'``  — Gradient Boosting (100 trees)
        - ``'knn'`` — k-Nearest Neighbours (k=5)
    """
    if model_type not in _MODEL_TEMPLATES:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Choose from {list(_MODEL_TEMPLATES)}."
        )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    copy.deepcopy(_MODEL_TEMPLATES[model_type])),
    ])


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    model_type: ModelType = "rf",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Pipeline, LabelEncoder, Dict]:
    """
    Train a model on (X, y) with a stratified hold-out split.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
    y : array-like, shape (n_samples,)
        Class labels (int or str).
    model_type : str, default 'rf'
    test_size : float, default 0.2
    random_state : int, default 42

    Returns
    -------
    model : fitted Pipeline
    label_encoder : fitted LabelEncoder
    results : dict
        Keys: ``accuracy``, ``balanced_accuracy``, ``classification_report``,
        ``X_test``, ``y_test`` (encoded), ``classes``.
    """
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=test_size, stratify=y_enc, random_state=random_state
    )

    model = build_model(model_type)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    results = {
        "accuracy":              float(np.mean(y_pred == y_test)),
        "balanced_accuracy":     float(balanced_accuracy_score(y_test, y_pred)),
        "classification_report": classification_report(
            y_test, y_pred, target_names=le.classes_.astype(str)
        ),
        "classes":  le.classes_,
        "X_test":   X_test,
        "y_test":   y_test,
    }
    return model, le, results


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


def cross_validate_model(
    X: np.ndarray,
    y: np.ndarray,
    model_type: ModelType = "rf",
    cv: int = 5,
    random_state: int = 42,
) -> Dict:
    """
    Evaluate a model with stratified k-fold cross-validation.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
    y : array-like, shape (n_samples,)
    model_type : str, default 'rf'
    cv : int, default 5
    random_state : int, default 42

    Returns
    -------
    results : dict
        Keys: ``accuracy_per_fold``, ``balanced_accuracy_per_fold``,
        ``mean_accuracy``, ``std_accuracy``,
        ``mean_balanced_accuracy``, ``std_balanced_accuracy``.
    """
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    model = build_model(model_type)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    cv_out = cross_validate(
        model, X, y_enc,
        cv=skf,
        scoring=["accuracy", "balanced_accuracy"],
        return_train_score=False,
    )

    acc = cv_out["test_accuracy"]
    bacc = cv_out["test_balanced_accuracy"]
    return {
        "accuracy_per_fold":           acc,
        "balanced_accuracy_per_fold":  bacc,
        "mean_accuracy":               float(acc.mean()),
        "std_accuracy":                float(acc.std()),
        "mean_balanced_accuracy":      float(bacc.mean()),
        "std_balanced_accuracy":       float(bacc.std()),
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def plot_confusion_matrix(
    model: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    class_names: Optional[List[str]] = None,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    normalize: Optional[Literal["true", "pred", "all"]] = None,
) -> plt.Axes:
    """
    Plot the confusion matrix for a fitted model on (X, y).

    Parameters
    ----------
    model : fitted Pipeline
    X : ndarray
    y : array-like
        Encoded labels (as returned in ``results['y_test']``).
    class_names : list of str, optional
    ax : matplotlib Axes, optional
    title : str, optional
    normalize : {'true', 'pred', 'all'}, optional
        Normalise the confusion matrix. None (default) shows raw counts.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    cm = confusion_matrix(y, model.predict(X), normalize=normalize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(title or "Confusion Matrix")
    plt.tight_layout()
    return ax


def plot_feature_importance(
    model: Pipeline,
    feature_names: List[str],
    top_n: int = 20,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    """
    Horizontal bar chart of feature importances for tree-based models (rf, gb).

    Raises ValueError for models without ``feature_importances_``.

    Parameters
    ----------
    model : fitted Pipeline
    feature_names : list of str
    top_n : int, default 20
    ax : matplotlib Axes, optional
    title : str, optional
    """
    clf = model.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        raise ValueError(
            f"{type(clf).__name__} does not expose feature_importances_. "
            "Use model_type='rf' or 'gb'."
        )

    importances = clf.feature_importances_
    idx = np.argsort(importances)[::-1][:top_n]
    names = [feature_names[i] for i in idx]
    values = importances[idx]

    if ax is None:
        _, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))

    ax.barh(names[::-1], values[::-1])
    ax.set_xlabel("Importance")
    ax.set_title(title or f"Top-{top_n} Feature Importances")
    plt.tight_layout()
    return ax


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    signals: List[np.ndarray],
    labels: List,
    fs: float,
    feature_type: FeatureType = "both",
    model_type: ModelType = "rf",
    nperseg: Optional[int] = None,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
    cv: int = 5,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Pipeline, LabelEncoder, np.ndarray, List[str], Dict, Dict]:
    """
    End-to-end pipeline: feature extraction → train/test split → cross-validation.

    Parameters
    ----------
    signals : list of 1-D arrays
    labels : list
        Class label per signal.
    fs : float
        Sampling frequency in Hz.
    feature_type : {'time', 'spectral', 'both', 'psd', 'raw'}, default 'both'
    model_type : {'rf', 'svm', 'lr', 'gb', 'knn'}, default 'rf'
    nperseg : int, optional
    bands : dict, optional
    cv : int, default 5
    test_size : float, default 0.2
    random_state : int, default 42

    Returns
    -------
    model : fitted Pipeline
    label_encoder : fitted LabelEncoder
    X : ndarray  —  full feature matrix (n_signals, n_features)
    feature_names : list of str
    train_results : dict  (from :func:`train_model`)
    cv_results : dict    (from :func:`cross_validate_model`)
    """
    y = np.asarray(labels)
    X, feature_names = build_feature_matrix(
        signals, fs, feature_type=feature_type, nperseg=nperseg, bands=bands
    )
    model, le, train_results = train_model(
        X, y, model_type=model_type, test_size=test_size, random_state=random_state
    )
    cv_results = cross_validate_model(
        X, y, model_type=model_type, cv=cv, random_state=random_state
    )
    return model, le, X, feature_names, train_results, cv_results

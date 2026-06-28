"""
Explainability analysis for trained ML models.

Supported methods:
  - SHAP (TreeExplainer for rf/gb, KernelExplainer for everything else)
  - LIME (tabular, single-instance)
  - Permutation feature importance (sklearn, model-agnostic)
"""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _transform(model: Pipeline, X: np.ndarray) -> np.ndarray:
    """Return X after applying all pipeline steps except the final classifier."""
    return model[:-1].transform(X)


def _shap_values_for_display(
    shap_values, class_idx: int
) -> np.ndarray:
    """
    Normalise the varied shapes SHAP returns into a 2-D array
    (n_samples, n_features) for a single class.
    """
    if isinstance(shap_values, list):
        # Older SHAP: list of (n_samples, n_features) arrays, one per class
        idx = min(class_idx, len(shap_values) - 1)
        return shap_values[idx]
    if shap_values.ndim == 3:
        # Newer SHAP multi-class: (n_samples, n_features, n_classes)
        return shap_values[:, :, class_idx]
    # Binary or regression: (n_samples, n_features)
    return shap_values


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------


def explain_shap(
    model: Pipeline,
    X: np.ndarray,
    feature_names: List[str],
    X_background: Optional[np.ndarray] = None,
    n_background: int = 100,
    class_idx: int = 1,
    max_display: int = 20,
) -> Dict:
    """
    Compute SHAP values and produce summary plots.

    Uses ``TreeExplainer`` for tree-based classifiers (rf, gb) and
    ``KernelExplainer`` for all others (svm, lr, knn).

    Parameters
    ----------
    model : fitted Pipeline (StandardScaler → classifier)
    X : ndarray, shape (n_samples, n_features)
        Samples to explain (unscaled; the scaler inside the Pipeline is applied
        internally).
    feature_names : list of str
    X_background : ndarray, optional
        Background dataset for KernelExplainer. Defaults to ``X``.
    n_background : int, default 100
        Number of background samples drawn from ``X_background`` for
        KernelExplainer (ignored for TreeExplainer).
    class_idx : int, default 1
        Class whose SHAP values are displayed (for multi-class models).
    max_display : int, default 20
        Maximum number of features shown in the plots.

    Returns
    -------
    result : dict
        ``shap_values``   — raw SHAP output (list or ndarray, unprocessed)
        ``shap_display``  — 2-D array (n_samples, n_features) for ``class_idx``
        ``expected_value``— model baseline (scalar or list)
        ``X_transformed`` — scaled X passed to the explainer
        ``fig_beeswarm``  — matplotlib Figure (beeswarm summary plot)
        ``fig_bar``       — matplotlib Figure (mean |SHAP| bar chart)
    """
    import shap

    clf = model.named_steps["clf"]
    X_tr = _transform(model, X)

    if hasattr(clf, "feature_importances_"):
        explainer = shap.TreeExplainer(clf, X_tr)
        shap_values = explainer.shap_values(X_tr)
    else:
        bg = X_background if X_background is not None else X
        bg_tr = _transform(model, bg)
        background = shap.sample(bg_tr, min(n_background, len(bg_tr)))
        explainer = shap.KernelExplainer(clf.predict_proba, background)
        shap_values = explainer.shap_values(X_tr)

    shap_display = _shap_values_for_display(shap_values, class_idx)
    expected_value = explainer.expected_value

    # Beeswarm summary plot
    fig_beeswarm = plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_display, X_tr,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    plt.title(f"SHAP Summary (class {class_idx})", pad=12)
    plt.tight_layout()

    # Bar chart of mean |SHAP|
    fig_bar = plt.figure(figsize=(8, max(4, min(max_display, len(feature_names)) * 0.4)))
    shap.summary_plot(
        shap_display, X_tr,
        feature_names=feature_names,
        plot_type="bar",
        max_display=max_display,
        show=False,
    )
    plt.title(f"Mean |SHAP| (class {class_idx})", pad=12)
    plt.tight_layout()

    return {
        "shap_values":    shap_values,
        "shap_display":   shap_display,
        "expected_value": expected_value,
        "X_transformed":  X_tr,
        "fig_beeswarm":   fig_beeswarm,
        "fig_bar":        fig_bar,
    }


def explain_shap_instance(
    model: Pipeline,
    instance: np.ndarray,
    shap_result: Dict,
    feature_names: List[str],
    instance_idx: int = 0,
    class_idx: int = 1,
) -> plt.Figure:
    """
    Waterfall plot for a single instance using a pre-computed SHAP result.

    Parameters
    ----------
    model : fitted Pipeline
    instance : ndarray, shape (1, n_features) or (n_features,)
        The sample to explain (unscaled).
    shap_result : dict
        Output of :func:`explain_shap`.
    feature_names : list of str
    instance_idx : int, default 0
        Row index within ``shap_result['shap_display']`` to visualise.
    class_idx : int, default 1

    Returns
    -------
    fig : matplotlib Figure
    """
    import shap

    sv = shap_result["shap_display"][instance_idx]
    ev = shap_result["expected_value"]
    base = ev[class_idx] if isinstance(ev, (list, np.ndarray)) else ev
    X_tr = shap_result["X_transformed"]

    expl = shap.Explanation(
        values=sv,
        base_values=base,
        data=X_tr[instance_idx],
        feature_names=feature_names,
    )

    fig, ax = plt.subplots(figsize=(10, max(4, len(feature_names) * 0.3)))
    shap.waterfall_plot(expl, max_display=20, show=False)
    plt.title(f"SHAP Waterfall — instance {instance_idx} (class {class_idx})", pad=12)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# LIME
# ---------------------------------------------------------------------------


def explain_lime(
    model: Pipeline,
    X_background: np.ndarray,
    instance: np.ndarray,
    feature_names: List[str],
    class_names: List[str],
    num_features: int = 10,
    class_idx: int = 1,
    num_samples: int = 5000,
    random_state: int = 42,
) -> Dict:
    """
    Explain a single prediction with LIME.

    LIME perturbs the original (unscaled) feature space and calls the full
    Pipeline's ``predict_proba``, so the internal scaler is applied
    transparently on every perturbed sample.

    Parameters
    ----------
    model : fitted Pipeline
    X_background : ndarray, shape (n_samples, n_features)
        Training data used to learn the feature distribution (unscaled).
    instance : 1-D array, shape (n_features,)
        The single sample to explain (unscaled).
    feature_names : list of str
    class_names : list of str
    num_features : int, default 10
        Maximum number of features shown in the explanation.
    class_idx : int, default 1
        Class whose local explanation is displayed.
    num_samples : int, default 5000
        Number of perturbed samples generated by LIME.
    random_state : int, default 42

    Returns
    -------
    result : dict
        ``explanation`` — lime Explanation object
        ``fig``         — matplotlib Figure
    """
    from lime.lime_tabular import LimeTabularExplainer

    explainer = LimeTabularExplainer(
        training_data=X_background,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification",
        random_state=random_state,
    )

    instance = np.asarray(instance, dtype=np.float64).ravel()
    explanation = explainer.explain_instance(
        data_row=instance,
        predict_fn=model.predict_proba,
        num_features=num_features,
        labels=(class_idx,),
        num_samples=num_samples,
    )

    fig = explanation.as_pyplot_figure(label=class_idx)
    fig.suptitle(f"LIME Explanation (class '{class_names[class_idx]}')", y=1.02)
    plt.tight_layout()

    return {"explanation": explanation, "fig": fig}


# ---------------------------------------------------------------------------
# Permutation feature importance
# ---------------------------------------------------------------------------


def explain_permutation(
    model: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_repeats: int = 30,
    top_n: int = 20,
    scoring: str = "balanced_accuracy",
    random_state: int = 42,
) -> Dict:
    """
    Compute and plot permutation feature importance.

    Works with any fitted estimator — the full Pipeline is used so the scaler
    is part of each evaluation.

    Parameters
    ----------
    model : fitted Pipeline
    X : ndarray, shape (n_samples, n_features)
        Evaluation data (unscaled).
    y : array-like, shape (n_samples,)
        Encoded labels (integer).
    feature_names : list of str
    n_repeats : int, default 30
        Number of times each feature is permuted.
    top_n : int, default 20
        Number of top features to display.
    scoring : str, default 'balanced_accuracy'
        Sklearn scoring string used for importance measurement.
    random_state : int, default 42

    Returns
    -------
    result : dict
        ``importances_mean`` — ndarray (n_features,), mean drop in score
        ``importances_std``  — ndarray (n_features,), std across repeats
        ``sorted_indices``   — indices sorted by descending importance
        ``fig``              — matplotlib Figure
    """
    perm = permutation_importance(
        model, X, y,
        n_repeats=n_repeats,
        scoring=scoring,
        random_state=random_state,
        n_jobs=-1,
    )

    means = perm.importances_mean
    stds = perm.importances_std
    sorted_idx = np.argsort(means)[::-1]
    display_idx = sorted_idx[:top_n]

    names = [feature_names[i] for i in display_idx]
    imp_mean = means[display_idx]
    imp_std = stds[display_idx]

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
    ax.barh(
        names[::-1], imp_mean[::-1],
        xerr=imp_std[::-1],
        align="center",
        color="steelblue",
        ecolor="black",
        capsize=3,
    )
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel(f"Mean decrease in {scoring}")
    ax.set_title(f"Permutation Feature Importance (top {top_n})")
    plt.tight_layout()

    return {
        "importances_mean": means,
        "importances_std":  stds,
        "sorted_indices":   sorted_idx,
        "fig":              fig,
    }


# ---------------------------------------------------------------------------
# Combined
# ---------------------------------------------------------------------------


def explain_all(
    model: Pipeline,
    X_train: np.ndarray,
    X_explain: np.ndarray,
    y_explain: np.ndarray,
    feature_names: List[str],
    class_names: List[str],
    lime_instance_idx: int = 0,
    shap_class_idx: int = 1,
    lime_class_idx: int = 1,
    shap_max_display: int = 20,
    perm_top_n: int = 20,
    perm_n_repeats: int = 30,
    perm_scoring: str = "balanced_accuracy",
) -> Dict:
    """
    Run SHAP, LIME, and permutation importance in one call.

    Parameters
    ----------
    model : fitted Pipeline
    X_train : ndarray
        Training data used as background for SHAP KernelExplainer and LIME
        (unscaled).
    X_explain : ndarray
        Samples to explain (unscaled). Used for SHAP and permutation.
    y_explain : array-like
        Encoded labels corresponding to ``X_explain``.
    feature_names : list of str
    class_names : list of str
    lime_instance_idx : int, default 0
        Index into ``X_explain`` for the LIME single-instance explanation.
    shap_class_idx : int, default 1
    lime_class_idx : int, default 1
    shap_max_display : int, default 20
    perm_top_n : int, default 20
    perm_n_repeats : int, default 30
    perm_scoring : str, default 'balanced_accuracy'

    Returns
    -------
    results : dict with keys ``'shap'``, ``'lime'``, ``'permutation'``
        Each value is the dict returned by the corresponding ``explain_*``
        function.
    """
    shap_result = explain_shap(
        model, X_explain, feature_names,
        X_background=X_train,
        class_idx=shap_class_idx,
        max_display=shap_max_display,
    )

    lime_result = explain_lime(
        model, X_train,
        instance=X_explain[lime_instance_idx],
        feature_names=feature_names,
        class_names=class_names,
        class_idx=lime_class_idx,
    )

    perm_result = explain_permutation(
        model, X_explain, y_explain, feature_names,
        n_repeats=perm_n_repeats,
        top_n=perm_top_n,
        scoring=perm_scoring,
    )

    return {"shap": shap_result, "lime": lime_result, "permutation": perm_result}

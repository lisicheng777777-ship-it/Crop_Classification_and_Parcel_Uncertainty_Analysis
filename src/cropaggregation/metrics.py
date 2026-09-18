import numpy as np
import warnings
from sklearn import metrics as skmetrics
from sklearn.metrics import log_loss
from sklearn.exceptions import UndefinedMetricWarning


def sanitize_probabilities(y_prob, n_classes=None, eps=1e-8):
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if y_prob.ndim != 2:
        raise ValueError(f"Expected a 2D probability array, got shape {y_prob.shape}")

    if n_classes is not None and y_prob.shape[1] != int(n_classes):
        raise ValueError(f"Expected {n_classes} probability columns, got {y_prob.shape[1]}")

    invalid_rows = ~np.isfinite(y_prob).all(axis=1)
    y_prob = np.nan_to_num(y_prob, nan=0.0, posinf=0.0, neginf=0.0)
    y_prob = np.clip(y_prob, 0.0, 1.0)

    row_sum = y_prob.sum(axis=1, keepdims=True)
    zero_rows = row_sum[:, 0] <= eps
    if np.any(zero_rows):
        fill_classes = int(n_classes or y_prob.shape[1])
        y_prob[zero_rows, :] = 1.0 / max(fill_classes, 1)
        row_sum = y_prob.sum(axis=1, keepdims=True)

    y_prob = y_prob / np.maximum(row_sum, eps)
    y_prob = np.clip(y_prob, eps, 1.0)
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
    if np.any(invalid_rows):
        print(f"Warning: sanitized {int(invalid_rows.sum())} rows with invalid predicted probabilities.")
    return y_prob


def _classification_report(y_true, y_pred, Classes):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        try:
            return skmetrics.classification_report(
                y_true, y_pred, labels=np.arange(len(Classes)),
                target_names=Classes, digits=6, zero_division=1
            )
        except TypeError:
            return skmetrics.classification_report(
                y_true, y_pred, labels=np.arange(len(Classes)),
                target_names=Classes, digits=6
            )


def _precision_score(y_true, y_pred, average, labels=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        try:
            return skmetrics.precision_score(
                y_true, y_pred, average=average, labels=labels, zero_division=1
            )
        except TypeError:
            return skmetrics.precision_score(y_true, y_pred, average=average, labels=labels)


def compute_ece(y_true, y_prob, n_bins=15):

    y_true = np.asarray(y_true)

    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)

    ece = 0.0

    for i in range(n_bins):

        mask = (
            (confidences > bin_boundaries[i]) &
            (confidences <= bin_boundaries[i + 1])
        )

        if np.sum(mask) == 0:
            continue

        accuracy = np.mean(
            predictions[mask] == y_true[mask]
        )

        confidence = np.mean(
            confidences[mask]
        )

        ece += (
            np.sum(mask) / len(y_true)
        ) * abs(accuracy - confidence)

    return float(ece)


def compute_brier(y_true, y_prob, n_classes):

    y_true_onehot = np.eye(n_classes)[y_true]

    brier = np.mean(
        np.sum(
            (y_prob - y_true_onehot) ** 2,
            axis=1
        )
    )

    return float(brier)


def predictive_entropy(y_prob):

    y_prob = np.clip(
        y_prob,
        1e-8,
        1.0
    )

    entropy = -np.sum(
        y_prob * np.log(y_prob),
        axis=1
    )

    return entropy


def metrics(y_true, y_pred, y_prob, Classes, n_classes=None):
    if n_classes is None:
        n_classes = len(Classes)
    labels = np.arange(int(n_classes))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        warnings.simplefilter("ignore", DeprecationWarning)
        report = _classification_report(y_true, y_pred, Classes)
        confusion_matrix = skmetrics.confusion_matrix(y_true, y_pred, labels=labels)
        accuracy = skmetrics.accuracy_score(y_true, y_pred)
        if hasattr(skmetrics, "balanced_accuracy_score"):
            balanced_accuracy = skmetrics.balanced_accuracy_score(y_true, y_pred)
        else:
            balanced_accuracy = skmetrics.recall_score(y_true, y_pred, average="macro")
        kappa = skmetrics.cohen_kappa_score(y_true, y_pred)
        f1_micro = skmetrics.f1_score(y_true, y_pred, average="micro")
        f1_macro = skmetrics.f1_score(y_true, y_pred, average="macro", labels=labels)
        f1_weighted = skmetrics.f1_score(y_true, y_pred, average="weighted")
        recall_micro = skmetrics.recall_score(y_true, y_pred, average="micro")
        recall_macro = skmetrics.recall_score(y_true, y_pred, average="macro", labels=labels)
        recall_weighted = skmetrics.recall_score(y_true, y_pred, average="weighted")
        precision_micro = _precision_score(y_true, y_pred, average="micro")
        precision_macro = _precision_score(y_true, y_pred, average="macro", labels=labels)
        precision_weighted = _precision_score(y_true, y_pred, average="weighted")
    # Use the supplied n_classes, or infer it when omitted.
    y_prob = sanitize_probabilities(y_prob, n_classes=n_classes)
    ece = compute_ece(y_true, y_prob)
    brier = compute_brier(y_true, y_prob, n_classes)
    nll = log_loss(y_true, y_prob, labels=np.arange(n_classes))
    entropy = predictive_entropy(y_prob)
    mean_entropy = np.mean(entropy)
    return dict(
        report=report,
        confusion_matrix=confusion_matrix,
        accuracy=accuracy,
        balanced_accuracy=balanced_accuracy,
        kappa=kappa,
        f1_micro=f1_micro,
        f1_macro=f1_macro,
        f1_weighted=f1_weighted,
        recall_micro=recall_micro,
        recall_macro=recall_macro,
        recall_weighted=recall_weighted,
        precision_micro=precision_micro,
        precision_macro=precision_macro,
        precision_weighted=precision_weighted,
        ECE=ece,
        Brier=brier,
        NLL=nll,
        Entropy=mean_entropy
    )


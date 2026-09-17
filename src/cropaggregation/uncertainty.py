"""Risk-coverage calculations selected from the experiment implementation."""
import numpy as np
from scipy.integrate import trapezoid
from sklearn.metrics import average_precision_score, roc_auc_score

def _curve(errors, uncertainty):
    order = np.argsort(np.asarray(uncertainty, dtype=float), kind="mergesort")
    accepted_errors = np.asarray(errors, dtype=float)[order]
    n = len(order)
    coverage = np.arange(1, n + 1, dtype=float) / n
    risk = np.cumsum(accepted_errors) / np.arange(1, n + 1, dtype=float)
    generalized_risk = coverage * risk
    return coverage, risk, generalized_risk


def _summary(errors, uncertainty):
    errors = np.asarray(errors, dtype=int)
    uncertainty = np.asarray(uncertainty, dtype=float)
    coverage, risk, generalized = _curve(errors, uncertainty)
    if len(errors) < 2 or np.unique(errors).size < 2:
        auroc = auprc = np.nan
    else:
        auroc = roc_auc_score(errors, uncertainty)
        auprc = average_precision_score(errors, uncertainty)
    # Normalize trapezoidal integration to the observed [1/N, 1] support.
    denominator = max(1.0 - 1.0 / len(errors), np.finfo(float).eps)
    aurc = trapezoid(risk, coverage) / denominator if len(errors) > 1 else np.nan
    augrc = trapezoid(generalized, coverage) / denominator if len(errors) > 1 else np.nan
    ideal_uncertainty = errors.astype(float)
    _, ideal_risk, _ = _curve(errors, ideal_uncertainty)
    ideal_aurc = trapezoid(ideal_risk, coverage) / denominator if len(errors) > 1 else np.nan
    return {
        "n": len(errors), "error_n": int(errors.sum()), "error_rate": errors.mean(),
        "auroc": auroc, "auprc": auprc, "aurc": aurc,
        "excess_aurc": aurc - ideal_aurc, "augrc": augrc,
    }


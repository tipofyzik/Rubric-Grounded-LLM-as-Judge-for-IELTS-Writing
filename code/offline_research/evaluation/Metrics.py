"""Integer-band metrics shared by AES and judge reports."""
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score

METRICS = {
    "mae": "MAE", "mse": "MSE", "rmse": "RMSE", "qwk": "QWK",
    "exact_accuracy": "Exact accuracy", "within_one_accuracy": "Within-1 accuracy",
    "pearson": "Pearson r", "spearman": "Spearman rho", "bias": "Mean bias",
}
LOWER_IS_BETTER = {"mae", "mse", "rmse"}


def integer_bands(values: Any) -> Any:
    """Integer bands.

    Args:
        values (Any): Caller-supplied settings or values used by this operation.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    if any(isinstance(current_value, (bool, np.bool_)) for current_value in np.asarray(values, dtype=object).flat):
        raise ValueError("Boolean scores are not integer bands")
    raw = np.asarray(values)
    if raw.dtype.kind not in "iuf" or raw.dtype.kind == "b":
        raise ValueError("Scores must be numeric integer bands in 0-9")
    result = raw.astype(float)
    if not (np.isfinite(result) & (result >= 0) & (result <= 9) & (result % 1 == 0)).all():
        raise ValueError("Scores must be finite integer bands in 0-9; no rounding")
    return result.astype(int)


def metric_values(truth: Any, predictions: Any) -> Any:
    """Metric values.

    Args:
        truth (Any): Truth used by this operation.
        predictions (Any): Predictions used by this operation.

    Returns:
        Any: The computed result for the supplied inputs.
    """
    actual, predicted = integer_bands(truth), integer_bands(predictions)
    if actual.ndim != 1 or actual.shape != predicted.shape:
        raise ValueError("Metrics need equally shaped one-dimensional scores")
    if not len(actual):
        return {name: None for name in METRICS}, {name: "no valid observations" for name in METRICS}
    error = predicted - actual
    values = {
        "mae": float(np.abs(error).mean()), "mse": float(np.square(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "exact_accuracy": float(np.mean(error == 0)),
        "within_one_accuracy": float(np.mean(np.abs(error) <= 1)), "bias": float(error.mean()),
    }
    reasons = {}
    if len(np.unique(np.concatenate((actual, predicted)))) < 2:
        values["qwk"], reasons["qwk"] = None, "single identical category"
    else:
        qwk = float(cohen_kappa_score(actual, predicted, labels=list(range(10)), weights="quadratic"))
        values["qwk"] = qwk if np.isfinite(qwk) else None
        if values["qwk"] is None:
            reasons["qwk"] = "undefined expected disagreement"
    for name, function in (("pearson", pearsonr), ("spearman", spearmanr)):
        if len(actual) < 2 or np.std(actual) == 0 or np.std(predicted) == 0:
            values[name], reasons[name] = None, "fewer than two observations or constant scores"
        else:
            value = float(function(actual, predicted)[0])
            values[name] = value if np.isfinite(value) else None
            if values[name] is None:
                reasons[name] = "undefined correlation"
    return values, reasons

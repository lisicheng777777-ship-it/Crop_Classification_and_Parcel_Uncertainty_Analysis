import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from .data import readSITSData
from .evaluation import evaluate,tune_temperature
from .modeling import get_model,load_model_checkpoint
from .training import train_epoch
CLASS_VALUES=np.asarray([1,2,3],dtype=np.int64)
def _mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return os.path.normpath(path)


def _csv_files(dataset_dir):
    train = sorted(Path(dataset_dir).glob("*_train_datasat.csv"))
    test = sorted(Path(dataset_dir).glob("*_test_datasat.csv"))
    if len(train) != 1 or len(test) != 1:
        raise RuntimeError(
            f"{dataset_dir} must contain exactly one train and one test datasat CSV"
        )
    return str(train[0]), str(test[0])


def _load_csv(path, n_channels=11):
    X, parcel_ids, y_original = readSITSData(path)
    if X.shape[1] != 77:
        raise ValueError(f"Expected 77 features, got {X.shape[1]} in {path}")
    if X.shape[1] % n_channels:
        raise ValueError(f"Features cannot be reshaped by {n_channels} channels: {path}")
    unknown = sorted(set(np.unique(y_original).tolist()) - set(CLASS_VALUES.tolist()))
    if unknown:
        raise ValueError(f"Unexpected classes in no-other dataset {path}: {unknown}")
    lookup = {int(value): idx for idx, value in enumerate(CLASS_VALUES)}
    y = np.asarray([lookup[int(value)] for value in y_original], dtype=np.int64)
    X = X.reshape(len(X), X.shape[1] // n_channels, n_channels).astype(np.float32)
    return X, np.asarray(parcel_ids, dtype=np.int64), y


def _load_dataset(dataset_dir, n_channels=11):
    train_file, test_file = _csv_files(dataset_dir)
    train = _load_csv(train_file, n_channels)
    test = _load_csv(test_file, n_channels)
    return {
        "dataset_dir": os.path.normpath(dataset_dir),
        "train_file": train_file,
        "test_file": test_file,
        "train": train,
        "test": test,
        "all": tuple(np.concatenate([train[i], test[i]], axis=0) for i in range(3)),
    }


def _fit_normalizer(X):
    # Preserve only the final feature/channel axis.  This supports both
    # [N,T,C] time series and [N,T,H,W,C] spatial patches.
    reduction_axes = tuple(range(X.ndim - 1))
    low = np.percentile(X, 2, axis=reduction_axes).astype(np.float32)
    high = np.percentile(X, 98, axis=reduction_axes).astype(np.float32)
    return low, high


def _normalize(X, low, high):
    shape = (1,) * (X.ndim - 1) + (len(low),)
    return ((X - low.reshape(shape)) / (high.reshape(shape) - low.reshape(shape) + 1e-8)).astype(np.float32)


def _parcel_table(parcel_ids, y):
    frame = pd.DataFrame({"parcel_id": parcel_ids, "truth": y})
    counts = frame.groupby("parcel_id")["truth"].nunique()
    bad = counts[counts > 1]
    if len(bad):
        raise ValueError(f"Parcels with inconsistent labels: {bad.index[:10].tolist()}")
    return frame.groupby("parcel_id", as_index=False)["truth"].first()


def _stratified_parcel_folds(parcel_ids, y, n_splits, seed):
    parcel = _parcel_table(parcel_ids, y)
    minimum = int(parcel["truth"].value_counts().min())
    folds = min(int(n_splits), minimum)
    if folds < 2:
        raise ValueError("At least two parcels per class are required for CV")
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    result = []
    for fold, (_, test_idx) in enumerate(splitter.split(parcel["parcel_id"], parcel["truth"])):
        ids = set(parcel.iloc[test_idx]["parcel_id"].astype(int).tolist())
        result.append(np.asarray([int(pid) in ids for pid in parcel_ids], dtype=bool))
    return result


def _validation_mask(parcel_ids, y, seed, n_splits=5):
    return _stratified_parcel_folds(parcel_ids, y, n_splits, seed)[0]


def _device(name):
    if str(name).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def _train_predict(
    X_train,
    parcel_train,
    y_train,
    X_test,
    parcel_test,
    y_test,
    output_dir,
    hyperparameters,
    device,
    epochs,
    batch_sizes,
    monitor,
    learning_rate,
    weight_decay,
    class_weight,
    seed,
    model_name="XGBoost",
    split_seed=None,
):
    output_dir = _mkdir(output_dir)
    pixel_path = os.path.join(output_dir, "pixel_predictions.csv")
    if os.path.exists(pixel_path):
        return pd.read_csv(pixel_path)

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    val_mask = _validation_mask(
        parcel_train, y_train, seed if split_seed is None else int(split_seed)
    )
    fit_mask = ~val_mask
    low, high = _fit_normalizer(X_train[fit_mask])
    X_fit = _normalize(X_train[fit_mask], low, high)
    X_val = _normalize(X_train[val_mask], low, high)
    X_eval = _normalize(X_test, low, high)
    np.savetxt(os.path.join(output_dir, "normalization_min.csv"), low[None, :], delimiter=",")
    np.savetxt(os.path.join(output_dir, "normalization_max.csv"), high[None, :], delimiter=",")

    dev = _device(device)
    model = get_model(
        model_name,
        # Time-series tensors are [N,T,C], while STSMamba patches are
        # [N,T,H,W,C].  The spectral/channel dimension is always the last one.
        X_fit.shape[-1],
        len(CLASS_VALUES),
        X_fit.shape[1],
        dev,
        **hyperparameters,
    )
    checkpoint = os.path.join(output_dir, "Best_model")
    train_epoch(
        model,
        X_fit,
        y_train[fit_mask],
        X_val,
        y_train[val_mask],
        checkpoint,
        dev,
        batch_sizes,
        epochs,
        monitor,
        learning_rate,
        weight_decay,
        polygon_ids_train=parcel_train[fit_mask],
        class_weight=class_weight,
        eval_mc_samples=1,
    )
    load_model_checkpoint(model, checkpoint, dev)
    temperature = tune_temperature(model, X_val, y_train[val_mask], dev, batch_sizes)
    with open(os.path.join(output_dir, "calibration_temperature.txt"), "w", encoding="utf-8") as stream:
        stream.write(f"{temperature:.8f}\n")
    _, y_true, y_pred, y_prob, epistemic = evaluate(
        model, X_eval, y_test, dev, batch_sizes, temperature=temperature, mc_samples=30
    )
    prob = y_prob.cpu().numpy()
    frame = pd.DataFrame({
        "parcel_id": parcel_test.astype(int),
        "truth": y_true.cpu().numpy().astype(int),
        "pred": y_pred.cpu().numpy().astype(int),
        "epistemic": epistemic.cpu().numpy().astype(float),
    })
    for idx in range(prob.shape[1]):
        frame[f"prob_{idx}"] = prob[:, idx]
    frame.to_csv(pixel_path, index=False, encoding="utf-8-sig")
    return frame


def _parcel_predictions(pixel):
    prob_cols = sorted([col for col in pixel if col.startswith("prob_")])
    rows = []
    max_entropy = np.log(max(len(prob_cols), 2))
    for parcel_id, part in pixel.groupby("parcel_id", sort=True):
        truth_values = part["truth"].unique()
        if len(truth_values) != 1:
            raise ValueError(f"Inconsistent truth for parcel {parcel_id}")
        probabilities = part[prob_cols].to_numpy(dtype=float)
        mean_prob = probabilities.mean(axis=0)
        pred = int(np.argmax(mean_prob))
        parcel_entropy = float(predictive_entropy(mean_prob[None, :])[0])
        pixel_entropy = float(predictive_entropy(probabilities).mean())
        sorted_prob = np.sort(mean_prob)
        rows.append({
            "parcel_id": int(parcel_id),
            "truth": int(truth_values[0]),
            "pred": pred,
            "correct": int(pred == int(truth_values[0])),
            "pixel_n": len(part),
            "parcel_confidence": float(mean_prob.max()),
            "confidence_uncertainty": float(1 - mean_prob.max()),
            "margin_risk": float(1 - (sorted_prob[-1] - sorted_prob[-2])),
            "parcel_entropy": parcel_entropy,
            "parcel_entropy_norm": parcel_entropy / max_entropy,
            "pixel_entropy_mean": pixel_entropy,
            "pixel_entropy_norm": pixel_entropy / max_entropy,
            "scale_gap": parcel_entropy - pixel_entropy,
            "scale_gap_norm": max(0.0, parcel_entropy - pixel_entropy) / max_entropy,
            "epistemic_mean": float(part["epistemic"].mean()),
        })
        for idx, value in enumerate(mean_prob):
            rows[-1][f"prob_{idx}"] = float(value)
    result = pd.DataFrame(rows)
    epi_max = result["epistemic_mean"].max()
    result["epistemic_norm"] = (
        result["epistemic_mean"] / epi_max if epi_max and np.isfinite(epi_max) else 0.0
    )
    return result

import gc, json, os, random
import numpy as np
import pandas as pd
import torch
from .data import readSITSData, read_minMaxVal, save_minMaxVal, extractValSet
from .modeling import get_model, load_model_checkpoint
from .training import train_epoch
from .evaluation import evaluate, load_temperature, tune_temperature, save_temperature
COMMON_CLASSES=(1,2,3)
N_CHANNELS=11
BATCH_SIZES=[128,128,8292]
EPOCHS=50
OUTPUT_ROOT='outputs/fixed'
def _json(path, value):
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(value, file_obj, ensure_ascii=False, indent=2)


def _load_common_split(dataset_dir, split):
    csv_path = os.path.join(dataset_dir, f"samples_{split}_datasat.csv")
    metadata_path = os.path.join(
        dataset_dir, f"samples_{split}_datasat_pixel_metadata.csv"
    )
    X, parcels, labels = readSITSData(csv_path)
    labels = labels.astype(int)
    metadata = pd.read_csv(metadata_path)
    if not (len(X) == len(parcels) == len(labels) == len(metadata)):
        raise ValueError(f"{dataset_dir}/{split} rows are not aligned")
    if not np.array_equal(parcels, metadata["polygon_id"].to_numpy(dtype=np.uint64)):
        raise ValueError(f"{dataset_dir}/{split} polygon IDs are not aligned")
    common = np.isin(labels, COMMON_CLASSES)
    X, parcels, labels = X[common], parcels[common], labels[common]
    metadata = metadata.loc[common].reset_index(drop=True)
    y = np.asarray([COMMON_CLASSES.index(int(value)) for value in labels], dtype=np.int64)
    return X, parcels, y, metadata


def _set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _train_2021_checkpoint(model_name, training_scheme, ring_weights,
                           train_data, test_data, patch_cache, device,
                           output_root=None, seed=42, source_label="2021_2022"):
    if output_root is None:
        output_dir = os.path.join(
            OUTPUT_ROOT, "reverse_2021_source_selection", training_scheme, model_name
        )
    else:
        output_dir = os.path.join(output_root, training_scheme, model_name)
    os.makedirs(output_dir, exist_ok=True)
    aggregation_path = os.path.join(output_dir, "source_validation_pixel_probabilities.csv")
    if os.path.isfile(aggregation_path):
        return pd.read_csv(aggregation_path)

    _set_seed(seed)
    X_train_center, parcels_train, y_train, metadata_train = train_data
    X_test_center, parcels_test, y_test, metadata_test = test_data
    X_train_center = X_train_center.copy()
    parcels_train = parcels_train.copy()
    y_train = y_train.copy()
    metadata_train = metadata_train.copy()
    ring_weights = np.asarray(ring_weights, dtype=np.float32)
    sample_weights = ring_weights[metadata_train["ring_id"].to_numpy(dtype=np.int64)]
    retained = sample_weights > 0
    X_train_center = X_train_center[retained]
    parcels_train = parcels_train[retained]
    y_train = y_train[retained]
    metadata_train = metadata_train.loc[retained].reset_index(drop=True)
    sample_weights = sample_weights[retained]

    time_steps = X_train_center.shape[1] // N_CHANNELS
    center_train = X_train_center.reshape(-1, time_steps, N_CHANNELS)
    center_test = X_test_center.reshape(-1, time_steps, N_CHANNELS)
    minimum = np.percentile(center_train, 2, axis=(0, 1))
    maximum = np.percentile(center_train, 98, axis=(0, 1))
    save_minMaxVal(os.path.join(output_dir, "min_Max.txt"), minimum, maximum)
    np.savez(os.path.join(output_dir, "normalization.npz"), minimum=minimum, maximum=maximum)
    hyperparameter = {"input_dims": (9, 2), "phenology_indices": (4, 5, 6, 7, 8)}
    if model_name == "STSMamba":
        X_train = patch_cache["train"][retained]
        X_test = patch_cache["test"]
        shape = (1, 1, 1, 1, N_CHANNELS)
        X_train = (X_train - minimum.reshape(shape)) / (
            maximum.reshape(shape) - minimum.reshape(shape) + 1e-8
        )
        X_test = (X_test - minimum.reshape(shape)) / (
            maximum.reshape(shape) - minimum.reshape(shape) + 1e-8
        )
    else:
        X_train = (center_train - minimum) / (maximum - minimum + 1e-8)
        X_test = (center_test - minimum) / (maximum - minimum + 1e-8)
    X_train = np.clip(X_train, 0, 1).astype(np.float32)
    X_test = np.clip(X_test, 0, 1).astype(np.float32)

    (X_fit, parcels_fit, y_fit, weights_fit,
     X_val, parcels_val, y_val, _) = extractValSet(
        X_train, parcels_train, y_train, val_rate=0.1,
        return_polygon_ids=True, sample_weights=sample_weights,
    )
    model = get_model(
        model_name, N_CHANNELS, len(COMMON_CLASSES), time_steps, device,
        **hyperparameter,
    )
    checkpoint = os.path.join(output_dir, "Best_model")
    if not os.path.isfile(checkpoint):
        train_epoch(
            model, X_fit, y_fit, X_val, y_val, checkpoint,
            device, BATCH_SIZES, EPOCHS, "kappa", 3e-4, 1e-3,
            polygon_ids_train=parcels_fit,
            parcel_consistency_weight=0.0,
            class_weight="balanced", label_smoothing=0.03,
            focal_gamma=0.0, sample_weights_train=weights_fit,
            eval_mc_samples=1,
        )
    load_model_checkpoint(model, checkpoint, device)
    temperature_path = os.path.join(output_dir, "calibration_temperature.txt")
    if os.path.isfile(temperature_path):
        temperature = load_temperature(temperature_path)
    else:
        temperature = tune_temperature(model, X_val, y_val, device, BATCH_SIZES)
        save_temperature(temperature_path, temperature)
    temperature = load_temperature(temperature_path)
    mc_samples = 1 if model_name == "XGBoost" else 30
    _, y_true, _, y_score, epistemic = evaluate(
        model, X_test, y_test, device, BATCH_SIZES,
        temperature=temperature, mc_samples=mc_samples,
    )
    probabilities = y_score.cpu().numpy()
    pixel_frame = metadata_test[["polygon_id", "ring_id", "distance_px"]].copy()
    pixel_frame["y_true"] = y_true.cpu().numpy()
    pixel_frame["epistemic"] = epistemic.cpu().numpy()
    for class_id in range(len(COMMON_CLASSES)):
        pixel_frame[f"prob_class_{class_id}"] = probabilities[:, class_id]
    pixel_frame.to_csv(os.path.join(output_dir, "source_test_pixel_probabilities.csv"), index=False)
    _, val_truth, _, val_prob, _ = evaluate(model, X_val, y_val, device, BATCH_SIZES, temperature=temperature, mc_samples=mc_samples)
    val_meta = pd.concat([metadata_train.loc[metadata_train.polygon_id.eq(pid), ['polygon_id','ring_id','distance_px']] for pid in pd.unique(parcels_val)], ignore_index=True)
    assert np.array_equal(val_meta.polygon_id.to_numpy(), parcels_val)
    val_meta['y_true']=val_truth.cpu().numpy()
    for c in range(3):val_meta[f'prob_class_{c}']=val_prob.cpu().numpy()[:,c]
    val_meta.to_csv(aggregation_path,index=False)
    _json(os.path.join(output_dir, "run_config.json"), {
        "direction": f"{source_label}_multiseed_internal",
        "model": model_name,
        "training_scheme": training_scheme,
        "training_ring_weights": ring_weights.tolist(),
        "common_classes": list(COMMON_CLASSES),
        "epochs": EPOCHS,
        "seed": int(seed),
    })
    del model, X_train, X_test
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pixel_frame

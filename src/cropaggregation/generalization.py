import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
D0_GRID=(.5,.75,1,1.5,2,3,4,5,6,8,10,12)
RASTERS={}
def _csv(value, cast=str):
    return [cast(x.strip()) for x in str(value).split(",") if x.strip()]


def _hyperparameters(args):
    return {
        "input_dims": tuple(_csv(args.input_dims, int)),
        "phenology_indices": tuple(_csv(args.phenology_indices, int)),
    }


def _metadata(root, split):
    path = Path(root) / f"samples_{split}_datasat_pixel_metadata.csv"
    d = pd.read_csv(path)
    required = {"polygon_id", "distance_px"}
    if not required.issubset(d):
        raise ValueError(f"Missing {sorted(required-set(d.columns))} in {path}")
    return d


def _attach_metadata(pixel, meta):
    if len(pixel) != len(meta):
        raise ValueError(f"Prediction/metadata rows differ: {len(pixel)} vs {len(meta)}")
    if not np.array_equal(pixel.parcel_id.to_numpy(int), meta.polygon_id.to_numpy(int)):
        raise ValueError("Prediction and distance metadata parcel IDs are not row-aligned")
    result = pixel.copy()
    result["distance_px"] = meta.distance_px.to_numpy(float)
    return result


def _prob_cols(d):
    return sorted([c for c in d if c.startswith("prob_")], key=lambda x: int(x.split("_")[-1]))


def _aggregate(pixel, d0):
    pcols = _prob_cols(pixel)
    d = pixel.distance_px.to_numpy(float)
    weights = np.ones(len(pixel)) if d0 == 0 else np.minimum(1.0, np.maximum(d, 0) / float(d0))
    # The same weighted mean, computed in arrays instead of repeated DataFrame indexing.
    parcels, first, inverse = np.unique(pixel.parcel_id.to_numpy(int), return_index=True, return_inverse=True)
    total = np.bincount(inverse, weights=weights)
    if np.any(total <= 0):
        raise ValueError('A parcel has no positive aggregation weights')
    probability = pixel[pcols].to_numpy(float)
    probability = np.column_stack([np.bincount(inverse, weights=weights*probability[:, c]) / total for c in range(len(pcols))])
    probability /= probability.sum(axis=1, keepdims=True)
    truth = pixel.truth.to_numpy(int)[first]
    pred = probability.argmax(axis=1)
    result = pd.DataFrame({'parcel_id':parcels, 'truth':truth, 'pred':pred,
                           'correct':(pred==truth).astype(int),
                           'effective_pixels':total**2/np.bincount(inverse, weights=weights**2)})
    for c in range(len(pcols)):
        result[f'prob_{c}'] = probability[:, c]
    return result


def _parcel_losses(parcel):
    p = parcel[_prob_cols(parcel)].to_numpy(float); y = parcel.truth.to_numpy(int)
    return np.sum((p - np.eye(p.shape[1])[y])**2, axis=1)


def _select_one_se(calibration):
    """Smallest d0 within one SE of the minimum mean parcel Brier."""
    rows = []
    for d0 in D0_GRID:
        parcel = _aggregate(calibration, d0)
        losses = _parcel_losses(parcel)
        rows.append({"d0": d0, "mean_brier": losses.mean(), "se_brier": losses.std(ddof=1)/np.sqrt(len(losses)), "n": len(losses)})
    curve = pd.DataFrame(rows)
    best = curve.loc[curve.mean_brier.idxmin()]
    eligible = curve[curve.mean_brier <= best.mean_brier + best.se_brier]
    selected = float(eligible.d0.min())
    return selected, curve, float(best.d0)


def _metrics(parcel, method, selected_d0, support):
    p = parcel[_prob_cols(parcel)].to_numpy(float); y = parcel.truth.to_numpy(int)
    return {"method": method, "selected_d0": selected_d0, "parcels": len(parcel), "support": support,
            "macro_f1": f1_score(y, parcel.pred, average="macro", zero_division=0),
            "brier": _parcel_losses(parcel).mean(),
            "nll": -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean(),
            "accuracy": parcel.correct.mean(), "mean_effective_pixels": parcel.effective_pixels.mean() if "effective_pixels" in parcel else np.nan}


def _spatial_predictions(data, meta, assignment, year, model, seed, args, output):
    from .research import _mkdir, _train_predict
    X, parcels, y = data["all"]
    if len(meta) != len(X) or not np.array_equal(meta.polygon_id.to_numpy(int), parcels):
        raise ValueError(f"All-pixel metadata mismatch for {year}")
    fold_map = assignment.set_index("parcel_id").spatial_cv_fold.to_dict()
    pixel_folds = np.array([fold_map[int(p)] for p in parcels], int)
    parts = []
    for fold in sorted(np.unique(pixel_folds)):
        destination = _mkdir(os.path.join(output, "spatial_cv", year, model, f"seed_{seed}", f"fold_{fold}"))
        eval_mask = pixel_folds == fold
        pixel = _train_predict(X[~eval_mask], parcels[~eval_mask], y[~eval_mask], X[eval_mask], parcels[eval_mask], y[eval_mask],
            destination, _hyperparameters(args), args.device, args.epoch, args.batch_size_list, args.monitor,
            args.learning_rate, args.weight_decay, args.class_weight, seed + fold, model_name=model,
            split_seed=args.supplement_split_seed)
        pixel = _attach_metadata(pixel, meta.loc[eval_mask].reset_index(drop=True)); pixel["fold"] = fold
        parts.append(pixel)
    combined = pd.concat(parts, ignore_index=True)
    path = Path(output) / "spatial_cv" / year / model / f"seed_{seed}" / "spatial_oof_pixel_predictions_with_distance.csv"
    combined.to_csv(path, index=False, encoding="utf-8-sig")
    return combined


def _data_for_model(data, meta, year, model, output, n_channels):
    from .spatial_patches import extract_spatial_patches
    if model != "STSMamba":
        return data
    cache = Path(output) / "00_patch_cache" / f"{year}_STSMamba_patch3.npy"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        patches = np.load(cache, mmap_mode="r")
    else:
        patches = extract_spatial_patches(RASTERS[year], meta, n_channels, patch_size=3)
        np.save(cache, patches)
    n_train = len(data["train"][0])
    if len(patches) != len(data["all"][0]):
        raise ValueError(f"STSMamba patch cache length mismatch for {year}")
    result = dict(data)
    result["train"] = (patches[:n_train], data["train"][1], data["train"][2])
    result["test"] = (patches[n_train:], data["test"][1], data["test"][2])
    result["all"] = (patches, data["all"][1], data["all"][2])
    return result


def _crossfit_distance(oof, destination):
    predictions, curves, choices = [], [], []
    for fold in sorted(oof.fold.unique()):
        calibration = oof[oof.fold != fold].reset_index(drop=True)
        target = oof[oof.fold == fold].reset_index(drop=True)
        selected, curve, raw_best = _select_one_se(calibration)
        curve["target_fold"] = fold; curves.append(curve)
        choices.append({"target_fold": fold, "selected_d0_one_se": selected, "raw_minimum_d0": raw_best,
                        "raw_minimum_at_upper_bound": raw_best == max(D0_GRID)})
        p = _aggregate(target, selected); p["fold"] = fold; p["selected_d0"] = selected; predictions.append(p)
    parcel = pd.concat(predictions, ignore_index=True)
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    parcel.to_csv(destination / "M2_spatial_crossfit_parcel_predictions.csv", index=False)
    pd.concat(curves).to_csv(destination / "d0_calibration_curves.csv", index=False)
    pd.DataFrame(choices).to_csv(destination / "d0_crossfit_choices.csv", index=False)
    m0 = _aggregate(oof, 0)
    metrics = pd.DataFrame([_metrics(m0, "M0_equal", 0, "all spatial OOF parcels"),
                            _metrics(parcel, "M2_linear_one_se_crossfit", np.nan, "all spatial OOF parcels")])
    metrics.to_csv(destination / "spatial_M0_M2_metrics.csv", index=False)
    return metrics


def _source_d0(source_oof, destination):
    selected, curve, raw_best = _select_one_se(source_oof)
    Path(destination).mkdir(parents=True, exist_ok=True)
    curve.to_csv(Path(destination) / "source_oof_d0_curve.csv", index=False)
    pd.DataFrame([{"selected_d0_one_se": selected, "raw_minimum_d0": raw_best,
                   "raw_minimum_at_upper_bound": raw_best == max(D0_GRID)}]).to_csv(Path(destination) / "frozen_source_d0.csv", index=False)
    return selected


def _temporal(source, target, target_meta, source_oof, source_year, target_year, model, seed, args, output):
    from .research import _mkdir, _train_predict
    destination = _mkdir(os.path.join(output, "temporal", f"{source_year}_to_{target_year}", model, f"seed_{seed}"))
    Xs, ps, ys = source["all"]; Xt, pt, yt = target["all"]
    pixel = _train_predict(Xs, ps, ys, Xt, pt, yt, destination, _hyperparameters(args), args.device, args.epoch,
        args.batch_size_list, args.monitor, args.learning_rate, args.weight_decay, args.class_weight, seed,
        model_name=model, split_seed=args.supplement_split_seed)
    pixel = _attach_metadata(pixel, target_meta)
    pixel.to_csv(Path(destination) / "target_pixel_predictions_with_distance.csv", index=False)
    d0 = _source_d0(source_oof, destination)
    m0, m2 = _aggregate(pixel, 0), _aggregate(pixel, d0)
    m0.to_csv(Path(destination) / "M0_target_parcel_predictions.csv", index=False)
    m2.to_csv(Path(destination) / "M2_target_parcel_predictions.csv", index=False)
    metrics = pd.DataFrame([_metrics(m0, "M0_equal", 0, "all target parcels"),
                            _metrics(m2, "M2_source_frozen_one_se", d0, "all target parcels")])
    metrics.insert(0, "direction", f"{source_year}_to_{target_year}"); metrics.insert(1, "model", model); metrics.insert(2, "seed", seed)
    metrics.to_csv(Path(destination) / "temporal_M0_M2_metrics.csv", index=False)
    return metrics

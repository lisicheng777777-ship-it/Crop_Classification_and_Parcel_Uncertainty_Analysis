# Data dictionary and experiment matrix

## Processed experimental data

| Season | Training pixels | Training parcels | Test pixels | Test parcels |
|---|---:|---:|---:|---:|
| 2025–2026 | 9,035 | 317 | 4,860 | 138 |
| 2021–2022 | 10,751 | 560 | 4,638 | 231 |

Each `data/processed/<season>/` directory contains:

- `samples_train_datasat.csv` and `samples_test_datasat.csv`: headerless tables; first column is class (1=wheat, 2=rapeseed, 3=vegetables), second is parcel ID, followed by 77 time-series features (7 periods × 11 channels).
- Corresponding `*_pixel_metadata.csv`: aligned row-for-row, including `polygon_id`, `ring_id`, `distance_px`, and spatial coordinates `x`, `y` used for STSMamba patch extraction.
- `parcels.gpkg`: actual experimental parcel geometry, joined by explicit parcel ID, not row position.
- `spatial_folds.csv`: parcel IDs, 30 m spatial groups and fold numbers 0–4. This records the existing experiment; a spatial-group name does not imply a strict exclusion buffer or nested validation.
- Historical-season `parcel_id_mapping.csv`: mapping used when consolidating the earlier data preparation versions.

Model probabilities use zero-based class indices 0, 1, 2 in the same wheat/rapeseed/vegetables order. Fixed-test probabilities use `polygon_id`, `y_true`, `prob_class_0` through `prob_class_2`. Generalization tables use `parcel_id`, `truth`, `prob_0` through `prob_2`; `canonical()` explicitly translates the schema.

## Experiments and checkpoint coverage

Five models × five seeds × two seasons gives 50 fixed-test configurations. Each spatial configuration has five held-out folds, yielding 250 spatial checkpoints. Two transfer directions × five models × five seeds yields another 50 checkpoints. Total: **350 checkpoints**. Spatial results for both seasons are required because each temporal source domain supplies its own OOF calibration probabilities.

`results/fixed/runs/<season>/seed_<seed>/T0_standard/<model>/` stores validation/test probabilities, normalization, temperature and historical run records. The corresponding weights are under `checkpoints/fixed/runs/`.

`results/generalization/spatial_cv/<season>/<model>/seed_<seed>/` stores fold predictions, the combined OOF table and `distance_selection/` results. Weights are under the matching `checkpoints/spatial_cv/` path.

`results/generalization/temporal/<source>_to_<target>/<model>/seed_<seed>/` stores target predictions, source-frozen d0, normalization and temperature. Weights are under the matching `checkpoints/temporal/` path.

## Full-resolution data attachment

`data/raw/image/2021-2022/S1S2_Fusion_2021_2022.tif` is the earlier-season 77-band image. The primary season uses `Fusion_202511_202605.tif` for fixed experiments and `Fusion_202511_202605_parcel_mask.tif` for generalization. The supplied original vector folders and required Shapefile sidecars are preserved in `data/raw/label/`.

The attachment manifest records the exact original bytes, including ancillary files. Transient lock files are excluded. These are the archived experiment inputs; they are not all individual pre-compositing Sentinel acquisition scenes. The table-to-raster check samples pixel centers and does not establish the physical interpretation of every band.

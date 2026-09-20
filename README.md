# Crop Classification and Parcel Uncertainty Analysis

Reproducible training, inference and numerical analysis for boundary-reliability-aware parcel crop probability aggregation. The primary season is **2025–2026**; **2021–2022** supplies the second domain for bidirectional temporal transfer.

The repository contains one selected implementation of the five experimental models (XGBoost, LSTM, TempCNN, VanillaTransformer and STSMamba), five seeds (`42, 3407, 2025, 2026, 2027`), the experimental feature tables and parcel splits, and archived probabilities and reference metrics. It does not bundle historical source trees, unused model variants or paper-layout scripts.

This repository is organized around the five-model experiments reported in the manuscript. Start with the reproduction commands below. See the [Publication guide](docs/PUBLICATION.md) for data attachments and GitHub publication steps, and [Validation](docs/VALIDATION.md) for verification results. The [public code repository](https://github.com/lisicheng777777-ship-it/Crop_Classification_and_Parcel_Uncertainty_Analysis) and [data/model release v1.0.0](https://github.com/lisicheng777777-ship-it/Crop_Classification_and_Parcel_Uncertainty_Analysis/releases/tag/v1.0.0) are available online.

## Recompute the archived results

Use Python 3.10 for the tested setup. Package metadata permits Python 3.10–3.12; the latter two were not tested here. The tested environment is recorded in [environment_training.json](docs/environment_training.json).

```bash
python -m pip install -r requirements-analysis.txt
python scripts/verify_repository.py
python scripts/validate.py
python scripts/reproduce.py
python analyze_uncertainty.py
```

These commands use distributed probabilities and do not require PyTorch, a GPU or the large raster attachment. Reproduction compares 150 fixed-test metric rows, 100 spatial metric rows and 100 temporal metric rows with archived references, using an absolute tolerance of `1e-8`. Outputs go to `outputs/`, which is ignored by Git. No figure/Word/PPT generation is required.

## Train and predict

Install the full dependencies and restore the release attachment for STSMamba and archived-checkpoint inference:

```bash
python -m pip install -r requirements.txt
python scripts/data_archive.py restore --directory release_assets
python train.py --experiment fixed --years 2021_2022 2025_2026 --dry-run
```

Raw images and checkpoints may already be present in a local complete copy. A Git clone alone contains the processed tables and probabilities; download **all** `crop-data.zip.NNN` files from the project's actual release into `release_assets/` before restoring. Download links are configured in `manifests/data_release.json`. To download and restore all nine parts automatically, run `python scripts/data_archive.py restore --directory release_assets --download`. Data and model attachments are hosted in [release v1.0.0](https://github.com/lisicheng777777-ship-it/Crop_Classification_and_Parcel_Uncertainty_Analysis/releases/tag/v1.0.0).

```bash
# Small executable example: one model and one seed on the primary season.
python train.py --experiment fixed --models XGBoost --seeds 42 --device cpu --output outputs/example_fixed
python predict.py --year 2025_2026 --model XGBoost --seed 42 --run-dir outputs/example_fixed/runs/2025_2026/seed_42/T0_standard/XGBoost --output outputs/example_prediction.csv

# Complete experimental configurations. Change cuda to cpu when needed.
python train.py --experiment fixed --years 2021_2022 2025_2026 --device cuda
python train.py --experiment spatial --years 2021_2022 2025_2026 --device cuda
python train.py --experiment temporal --device cuda

# Analyze newly trained results.
python analyze.py --experiment fixed --input outputs/fixed --output outputs/analysis_fixed
python analyze.py --experiment spatial --input outputs/generalization --output outputs/analysis_spatial
python analyze.py --experiment temporal --input outputs/generalization --output outputs/analysis_temporal
```

Configuration paths are resolved relative to the repository, not the author's drive letters. Fixed and generalization experiments use different primary-season rasters; do not interchange them. The default configuration uses CPU. `--epochs` changes neural-network training epochs; it does not change XGBoost's tree count. Use separate output directories when changing experimental settings.

`predict.py` is the fixed-split inference entry point. It supports both newly produced `normalization.npz` and the historical fixed-checkpoint decimal normalization protocol. Spatial and temporal inference occur in their respective training pipelines; their checkpoints are not interchangeable with fixed-split checkpoints.

## Contents

| Location | Purpose |
|---|---|
| `train.py`, `predict.py`, `analyze.py` | Selected training, fixed-split inference and numerical analysis entry points |
| `analyze_uncertainty.py` | Parcel entropy, confidence risk, archived pixel epistemic scores and risk–coverage analysis |
| `src/cropaggregation/` | Shared model, training, evaluation, aggregation and generalization implementation |
| `configs/` | Paper scope, model/seed settings and portable data paths |
| `data/processed/` | Two seasons of aligned features, labels, metadata, parcel geometry and spatial folds |
| `data/split_audit/` | Historical split and spatial overlap audit tables |
| `results/fixed/` | 50 fixed-test configurations, including training-side validation probabilities |
| `results/generalization/` | 50 spatial configurations (250 trained folds) and 50 directional transfer configurations |
| `results/reference/` | Archived fixed-test metrics and frozen-parameter analysis references |
| `data/raw/`, `checkpoints/` | Full original rasters/vectors and 350 trained checkpoints; supplied through release assets, ignored by Git |
| `manifests/` | File checksums and large-data release metadata |
| `scripts/` | Reproduction, validation, feature preparation, archive and release utilities |
| `docs/` | Data dictionary, validation evidence, limitations, provenance and publication instructions |

The feature extraction utility is `scripts/prepare_features.py`; use `--help` and supply the input raster, labeled vector and output path explicitly. Exact reported experiments use the distributed frozen tables and splits. The retained GEE script documents a historical export recipe; its band semantics have not been independently established for every archived feature column.

## Scope and interpretation

M0 is equal-weight parcel aggregation, M1 removes pixels less than one pixel from the boundary, and M2 applies a linearly saturating distance weight with a training-side or source-side selected recovery distance. [DATA.md](docs/DATA.md) records class encodings, sample counts and file schemas; [METHODS_AND_LIMITATIONS.md](docs/METHODS_AND_LIMITATIONS.md) explains calibration, uncertainty and reproducibility limits.

The full historical model collection has been load-tested, but all 350 models have **not** been retrained during repository preparation. Exact archived-probability reproduction is distinct from stochastic retraining. In particular, the original spatial parameter calibration is not strictly nested spatial validation.

## Availability and attribution

Publication status and an English manuscript statement are in [DATA_AND_CODE_AVAILABILITY.md](docs/DATA_AND_CODE_AVAILABILITY.md). Use the [code repository](https://github.com/lisicheng777777-ship-it/Crop_Classification_and_Parcel_Uncertainty_Analysis) and [data/model release v1.0.0](https://github.com/lisicheng777777-ship-it/Crop_Classification_and_Parcel_Uncertainty_Analysis/releases/tag/v1.0.0) when citing availability. The research software is distributed under [GNU GPL version 3](LICENSE). Verified TempCNN provenance and modification notices are recorded in [THIRD_PARTY_NOTICES.md](docs/THIRD_PARTY_NOTICES.md). The code license does not license datasets, trained weights or result tables; their reuse terms remain to be specified separately.

# Validation performed during repository preparation

Validation uses Python 3.10.18 and the package versions in `environment_training.json`. Machine-readable evidence is in `docs/validation/`.

| Check | Result |
|---|---|
| Fixed-test reproduction | 50 configurations × 3 methods = 150 metric rows; maximum difference 0 against the distributed archived metric CSV |
| Spatial reproduction | Both seasons, 50 configurations × 2 methods = 100 metric rows; maximum absolute difference below 3 × 10⁻¹⁰ |
| Temporal reproduction | Two directions, 50 configurations × 2 methods = 100 metric rows; maximum absolute difference below 6 × 10⁻¹⁰ |
| Checkpoint compatibility | All 350 checkpoints loaded; all produced finite three-class outputs in a forward-pass check |
| Dataset consistency | Two seasons checked for finite 79-column feature rows, metadata/parcel alignment, disjoint train/test parcel IDs, five-fold coverage and spatial group integrity |
| Raster/table relationship | 96 sampled pixel centers across season/split/raster-role combinations agree with the distributed feature rows within the stated raster tolerance |
| Real training smoke run | Primary-season XGBoost, seed 42, full configured tree count; completed training, calibration, validation and test prediction |
| New-checkpoint independent prediction | 4,860 primary-season pixels; maximum probability difference 0 versus the training output |
| Historical-checkpoint independent prediction | 4,860 primary-season XGBoost pixels; maximum probability difference 2 × 10⁻⁷ after restoring float32 normalization |
| Configuration validation | Fixed, spatial and temporal entry points pass dry runs using portable paths and all five models |
| Uncertainty entry point | Runs over all 25 primary-season fixed-test configurations |
| Source duplication | Nonempty byte-identical Python files are rejected by the repository validator |
| Unit checks | Aggregation/risk properties, multipart ZIP restoration and refusal to overwrite differing files are tested |

The shared spatial aggregation was checked against the earlier per-parcel implementation (maximum difference about 3.5 × 10⁻¹³ including effective-pixel statistics), then checked against all archived spatial metrics. The default reproduction tolerance is `1e-8` for numerical metrics. Archived checkpoint probabilities use a separate `1e-6` comparison tolerance due to saved decimal precision and evaluation arithmetic.

The data attachment is checked against its part-level and file-level SHA256 manifests, independently of the repository-file checksums. Final checks and attachment status are recorded in `validation/final_checks.json`.

These checks do not claim a fresh training of every historical model, complete semantic verification of all raster bands, or proof of strict nested validation. See `METHODS_AND_LIMITATIONS.md` for the boundaries of those claims.

# Selected implementation map

| Experiment/function | Distributed implementation | Experiment implementation selected during preparation |
|---|---|---|
| Fixed-split model fitting and probability generation | `train.py`, `src/cropaggregation/fixed.py` | T0 internal-split training in `temporal_best_transfer.py` |
| Spatial grouped folds and bidirectional temporal transfer | `train.py`, `generalization.py`, `research.py` | `reviewer_training_suite_0817.py` and shared `research_experiments.py` training helpers |
| Five models | `src/cropaggregation/models/`, `modeling.py` | Corresponding experimental LSTM, TempCNN, VanillaTransformer, XGBoost and STSMamba implementations |
| Training loss, calibration and MC inference | `training.py`, `evaluation.py`, `metrics.py` | Shared experimental training/evaluation modules, with unused model branches removed |
| M0/M1/M2 and fixed d0 selection | `aggregation.py`, `analyze.py` | Training-side frozen aggregation and paper numerical protocol |
| Spatial/temporal d0 selection | `generalization.py` | Historical OOF cross-calibration and source-frozen selection |
| Boundary, small-parcel, bootstrap and paired statistics | `analyze.py` | Numerical analysis selected from the paper experiment outputs/protocol |
| Risk coverage and error detection | `uncertainty.py`, `analyze_uncertainty.py` | `_curve` and `_summary` from the experiment's `selective_metrics.py` |
| Feature preparation | `scripts/prepare_features.py` | Distance-aware master feature extraction from `tool/makedataset.py`, with local defaults removed |
| Export recipe | `scripts/gee_export.js` | Retained historical Sentinel fusion export script, with its provenance limitation made explicit |

Model definitions and training algorithms are shared across entry points rather than copied into separate training/inference folders. `scripts/reproduce.py` calls the same numerical analysis code used for new outputs. The numerical tests cover aggregation support, label consistency, parameter ties, risk direction and attachment restoration. Only used code is shipped; earlier full source snapshots remain outside the publication repository.

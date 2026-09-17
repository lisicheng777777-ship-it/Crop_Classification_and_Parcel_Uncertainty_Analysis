"""Reference baselines for multitemporal crop classification.

All neural baselines accept ``[batch, time, channels]`` float tensors and
return unnormalised class logits.  Keeping this contract identical to
GBCA-STransformer5 makes training, calibration, MC-dropout inference and
parcel aggregation directly comparable.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .LSTM import LSTM as _LegacyLSTM
from .TempCNN import TempCNN as _LegacyTempCNN


class LSTMBaseline(_LegacyLSTM):
    """Bidirectional LSTM baseline with the repository's established defaults."""

    def forward(self, x):
        return self.logits(x)


class TempCNNBaseline(_LegacyTempCNN):
    """TempCNN baseline returning logits instead of log-softmax values."""

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv_bn_relu1(x)
        x = self.conv_bn_relu2(x)
        x = self.conv_bn_relu3(x)
        x = self.flatten(x)
        x = self.dense(x)
        return self.logsoftmax[0](x)


class VanillaTransformer(nn.Module):
    """Plain temporal Transformer without GBCA-specific components."""

    def __init__(self, input_dim, num_classes, sequencelength=7, d_model=64,
                 n_head=4, n_layers=2, d_inner=128, dropout=0.3,
                 pooling="mean"):
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError("d_model must be divisible by n_head")
        self.input_norm = nn.LayerNorm(input_dim)
        self.embedding = nn.Linear(input_dim, d_model)
        self.position = nn.Parameter(torch.zeros(1, sequencelength, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_inner,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.output_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)
        self.pooling = pooling
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, x):
        h = self.embedding(self.input_norm(x))
        h = h + self.position[:, :h.size(1)]
        h = self.encoder(h)
        if self.pooling == "last":
            h = h[:, -1]
        elif self.pooling == "max":
            h = h.max(dim=1).values
        else:
            h = h.mean(dim=1)
        return self.classifier(self.dropout(self.output_norm(h)))




class XGBoostBaseline(nn.Module):
    """XGBoost adapter exposing the same forward interface as torch models.

    The temporal tensor is flattened to ``time * channels`` features.  XGBoost
    remains a genuine tree model; ``training.train_epoch`` detects this class
    and calls :meth:`fit` rather than performing gradient descent.
    """

    is_tree_model = True

    def __init__(self, input_dim, num_classes, sequencelength=7,
                 n_estimators=500, max_depth=6, learning_rate=0.05,
                 subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                 random_state=42, n_jobs=-1, **kwargs):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.sequencelength = int(sequencelength)
        self.params = dict(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            learning_rate=float(learning_rate),
            subsample=float(subsample),
            colsample_bytree=float(colsample_bytree),
            reg_lambda=float(reg_lambda),
            random_state=int(random_state),
            n_jobs=int(n_jobs),
            objective="multi:softprob",
            num_class=self.num_classes,
            eval_metric="mlogloss",
        )
        self.estimator = None
        # A non-trainable anchor lets the generic code discover the device.
        self.register_buffer("_device_anchor", torch.empty(0), persistent=False)

    @staticmethod
    def _require_xgboost():
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "MODEL_NAME='XGBoost' requires the optional dependency. "
                "Install it with: python -m pip install xgboost"
            ) from exc
        return XGBClassifier

    @staticmethod
    def _flatten(x):
        return np.asarray(x, dtype=np.float32).reshape(len(x), -1)

    def fit(self, X, y, X_val=None, y_val=None, sample_weight=None):
        classifier = self._require_xgboost()
        self.estimator = classifier(**self.params)
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        if X_val is not None and y_val is not None and len(y_val):
            fit_kwargs["eval_set"] = [(self._flatten(X_val), np.asarray(y_val))]
            fit_kwargs["verbose"] = False
        self.estimator.fit(self._flatten(X), np.asarray(y), **fit_kwargs)
        return self

    def forward(self, x):
        if self.estimator is None:
            raise RuntimeError("XGBoost model has not been fitted or loaded")
        probabilities = self.estimator.predict_proba(
            self._flatten(x.detach().cpu().numpy())
        )
        logits = np.log(np.clip(probabilities, 1e-8, 1.0))
        return torch.as_tensor(logits, dtype=x.dtype, device=x.device)

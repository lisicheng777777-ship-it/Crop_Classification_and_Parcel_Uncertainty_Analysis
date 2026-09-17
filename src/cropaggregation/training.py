import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from .evaluation import evaluate
from .metrics import metrics
from .modeling import EarlyStopping, save_model_checkpoint


def _class_weights(y_train, num_classes, mode):
    if mode in (None, "none", "None", False):
        return None
    counts = np.bincount(y_train.astype(np.int64), minlength=num_classes).astype(np.float64)
    if mode == "balanced":
        weights = counts.sum() / (num_classes * np.maximum(counts, 1.0))
    elif mode in ("sqrt_balanced", "sqrt-balanced"):
        # A gentler alternative to inverse-frequency weighting.  It retains
        # minority-class support without pushing the majority class too
        # aggressively toward minority predictions.
        balanced = counts.sum() / (num_classes * np.maximum(counts, 1.0))
        weights = np.sqrt(balanced)
    else:
        weights = np.asarray([float(v) for v in str(mode).split(",")], dtype=np.float64)
        if len(weights) != num_classes:
            raise ValueError(f"class_weight expects {num_classes} values, got {len(weights)}")
    weights = weights / np.mean(weights)
    return torch.from_numpy(weights.astype(np.float32))


def _classification_loss(logits, target, class_weight=None, label_smoothing=0.0,
                         focal_gamma=0.0, sample_weight=None):
    if focal_gamma and focal_gamma > 0:
        log_prob = F.log_softmax(logits, dim=1)
        prob = log_prob.exp()
        target_prob = prob.gather(1, target.unsqueeze(1)).squeeze(1)
        ce = F.nll_loss(log_prob, target, weight=class_weight, reduction="none")
        loss = ((1.0 - target_prob).clamp(0, 1) ** float(focal_gamma)) * ce
    else:
        try:
            loss = F.cross_entropy(
                logits, target, weight=class_weight, reduction="none",
                label_smoothing=float(label_smoothing),
            )
        except TypeError:
            loss = F.cross_entropy(logits, target, weight=class_weight, reduction="none")
    if sample_weight is not None:
        sample_weight = sample_weight.to(loss.device, dtype=loss.dtype).reshape(-1)
        return (loss * sample_weight).sum() / sample_weight.sum().clamp_min(1e-12)
    return loss.mean()


def _parcel_consistency_loss(logits, parcel_ids, min_size=2):
    if parcel_ids is None:
        return logits.new_tensor(0.0)
    probs = F.softmax(logits, dim=1)
    unique_ids = torch.unique(parcel_ids)
    losses = []
    for parcel_id in unique_ids:
        mask = parcel_ids == parcel_id
        if int(mask.sum().item()) < int(min_size):
            continue
        parcel_prob = probs[mask]
        mean_prob = parcel_prob.mean(dim=0, keepdim=True)
        losses.append(F.mse_loss(parcel_prob, mean_prob.expand_as(parcel_prob), reduction="mean"))
    if not losses:
        return logits.new_tensor(0.0)
    return torch.stack(losses).mean()


def train_epoch(model, X_train, y_train, X_test, y_test,model_save_path,
                device, batch_size_list=None,epoch=20, monitor="test_loss",
                learning_rate=0.001, weight_decay=0, eval_temperature=1.0,
                polygon_ids_train=None, parcel_consistency_weight=0.0,
                parcel_consistency_min_size=2, class_weight="none",
                label_smoothing=0.0, focal_gamma=0.0,
                eval_mc_samples=1, sample_weights_train=None):

    if batch_size_list is None:
        batch_size_list = [4096, 8192, 8192]
    if getattr(model, "is_tree_model", False):
        counts = np.bincount(y_train.astype(np.int64), minlength=int(np.max(y_train)) + 1)
        sample_weight = None
        if class_weight in ("balanced", "sqrt_balanced", "sqrt-balanced"):
            per_class = len(y_train) / (len(counts) * np.maximum(counts, 1))
            if class_weight in ("sqrt_balanced", "sqrt-balanced"):
                per_class = np.sqrt(per_class)
            sample_weight = per_class[y_train.astype(np.int64)]
        elif class_weight not in (None, "none", "None", False):
            per_class = np.asarray([float(v) for v in str(class_weight).split(",")])
            sample_weight = per_class[y_train.astype(np.int64)]
        if sample_weights_train is not None:
            boundary_weight = np.asarray(sample_weights_train, dtype=np.float64)
            sample_weight = boundary_weight if sample_weight is None else sample_weight * boundary_weight
        print("Training XGBoost on flattened temporal features...")
        model.fit(X_train, y_train, sample_weight=sample_weight)
        save_model_checkpoint(model, model_save_path)
        test_loss, y_true, y_pred, y_score, epistemic = evaluate(
            model, X_test, y_test, device, batch_size_list, temperature=1.0,
            mc_samples=eval_mc_samples
        )
        classes = [f"class {i}" for i in np.unique(y_true.cpu())]
        scores = metrics(
            y_true.detach().cpu().numpy(),
            y_pred.detach().cpu().numpy(),
            y_score.detach().cpu().numpy(),
            classes,
        )
        scores["Epistemic"] = float(epistemic.mean())
        scores["Aleatoric"] = max(0.0, scores["Entropy"] - scores["Epistemic"])
        scores.update(epoch=0, train_loss=np.nan, test_loss=float(test_loss[0]))
        pd.DataFrame([scores]).set_index("epoch").to_csv(
            os.path.join(os.path.dirname(model_save_path), "trainlog.csv")
        )
        return
    num_classes = int(np.max(y_train)) + 1
    class_weight_tensor = _class_weights(y_train, num_classes, class_weight)
    if class_weight_tensor is not None:
        class_weight_tensor = class_weight_tensor.to(device)
    # covert numpy to pytorch tensor and put into gpu
    X_train = torch.from_numpy(X_train.astype(np.float32))
    if sys.platform == "win32":
        y_train = torch.from_numpy(y_train.astype(np.int64))
    elif "linux" in sys.platform:
        y_train = torch.from_numpy(y_train.astype(np.int_))

    # add channel dimension to time series data
    if len(X_train.shape) == 2:
        X_train = X_train.unsqueeze_(1)

    # ---- optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
    )
    warmup_epochs = min(5, max(1, int(epoch) // 10))

    def lr_multiplier(epoch_index):
        """Linear warmup followed by cosine decay."""
        if epoch_index < warmup_epochs:
            return float(epoch_index + 1) / float(warmup_epochs)
        decay_epochs = max(int(epoch) - warmup_epochs, 1)
        progress = min(
            float(epoch_index - warmup_epochs) / float(decay_epochs), 1.0
        )
        return 0.05 + 0.95 * 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lr_multiplier
    )

    # build dataloader
    use_parcel_consistency = parcel_consistency_weight is not None and float(parcel_consistency_weight) > 0 and polygon_ids_train is not None
    use_sample_weights = sample_weights_train is not None
    boundary_weights_tensor = None
    if use_sample_weights:
        boundary_weights_tensor = torch.from_numpy(
            np.asarray(sample_weights_train, dtype=np.float32)
        )
    if use_parcel_consistency:
        _, polygon_codes = np.unique(np.asarray(polygon_ids_train), return_inverse=True)
        polygon_ids_train = torch.from_numpy(polygon_codes.astype(np.int64))
        if use_sample_weights:
            train_dataset = TensorDataset(X_train, y_train, boundary_weights_tensor, polygon_ids_train)
        else:
            train_dataset = TensorDataset(X_train, y_train, polygon_ids_train)
    else:
        if use_sample_weights:
            train_dataset = TensorDataset(X_train, y_train, boundary_weights_tensor)
        else:
            train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size_list[0], shuffle=True)

    # A patience equal to the full epoch budget disables early stopping in
    # practice and made the 300-fold spatial package needlessly expensive.
    # Seven validation checks is long enough to ignore short metric plateaus
    # while still restoring the best checkpoint below for evaluation.
    early_stopping = EarlyStopping(patience=7, path=model_save_path)

    log = list()

    start = time.time()
    epoch_loss = 0
    epoch_bar = tqdm(range(epoch), desc="Training epochs", unit="epoch")
    for epoch_idx in epoch_bar:
        model.train()
        epoch_loss = 0
        epoch_ce_loss = 0
        epoch_parcel_loss = 0
        batch_bar = tqdm(train_loader, desc=f"Train batches {epoch_idx + 1}/{epoch}", unit="batch", leave=False)
        for sample in batch_bar:
            optimizer.zero_grad()
            x_batch = sample[0].to(device)
            y_batch = sample[1].to(device)
            weight_batch = sample[2].to(device) if use_sample_weights else None
            parcel_index = 3 if use_sample_weights else 2
            parcel_batch = sample[parcel_index].to(device) if use_parcel_consistency else None

            log_proba = model(
                x_batch
            )

            ce_loss = _classification_loss(
                log_proba,
                y_batch,
                class_weight=class_weight_tensor,
                label_smoothing=label_smoothing,
                focal_gamma=focal_gamma,
                sample_weight=weight_batch,
            )
            parcel_loss = _parcel_consistency_loss(
                log_proba,
                parcel_batch,
                min_size=parcel_consistency_min_size,
            ) if use_parcel_consistency else log_proba.new_tensor(0.0)
            output = ce_loss + float(parcel_consistency_weight) * parcel_loss

            output.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += output.item()
            epoch_ce_loss += ce_loss.item()
            epoch_parcel_loss += parcel_loss.item()
            batch_bar.set_postfix(
                loss=f"{output.item():.6f}",
                ce=f"{ce_loss.item():.6f}",
                parcel=f"{parcel_loss.item():.6f}",
            )


        train_loss = epoch_loss / max(len(train_loader), 1)
        scheduler.step()

        # get test loss
        model.eval()
        test_loss, y_true, y_pred, y_score, epistemic = evaluate(
            model, X_test, y_test, device, batch_size_list, temperature=eval_temperature,
            mc_samples=eval_mc_samples
        )

        Classes = [f'class {i}' for i in np.unique(y_true.cpu())]
        scores = metrics(
            y_true.cpu().numpy(),
            y_pred.cpu().numpy(),
            y_score.cpu().numpy(),
            Classes
        )
        # 在 train_epoch 中，scores = metrics(...) 之后，添加：
        scores["Epistemic"] = epistemic.mean().item()  # 已有，但确保存在
        # 计算 Aleatoric
        if "Entropy" in scores and "Epistemic" in scores:
            aleatoric = max(0.0, scores["Entropy"] - scores["Epistemic"])
            scores["Aleatoric"] = aleatoric
        else:
            scores["Aleatoric"] = float('nan')

        test_loss = test_loss.cpu().detach().numpy()[0]

        scores["epoch"] = epoch_idx
        scores["train_loss"] = train_loss
        scores["train_ce_loss"] = epoch_ce_loss / max(len(train_loader), 1)
        scores["train_parcel_loss"] = epoch_parcel_loss / max(len(train_loader), 1)
        scores["test_loss"] = test_loss
        scores["learning_rate"] = optimizer.param_groups[0]["lr"]
        scores["time"] = (time.time() - start) / 60
        log.append(scores)
        log_df = pd.DataFrame(log).set_index("epoch")
        log_df.to_csv(os.path.join(os.path.dirname(model_save_path), "trainlog.csv"))

        print(
            f'train_loss={train_loss:.6f}, '
            f'ce={scores["train_ce_loss"]:.6f}, '
            f'parcel={scores["train_parcel_loss"]:.6f}, '
            f'test_loss={test_loss:.6f}, '
            f'lr={scores["learning_rate"]:.2e}, '
            f'OA={scores["accuracy"]:.4f}, '
            f'F1={scores["f1_macro"]:.4f}, '
            f'Kappa={scores["kappa"]:.4f}, '
            f'ECE={scores["ECE"]:.4f}, '
            f'Brier={scores["Brier"]:.4f}, '
            f'NLL={scores["NLL"]:.4f}, '
            f'Entropy={scores["Entropy"]:.4f}, '
            f'Aleatoric={scores["Aleatoric"]:.4f}, '
            f'Epistemic={scores["Epistemic"]:.4f}'
        )
        epoch_bar.set_postfix(
            train_loss=f"{train_loss:.4f}",
            test_loss=f"{test_loss:.4f}",
            OA=f"{scores['accuracy']:.4f}",
            F1=f"{scores['f1_macro']:.4f}",
        )
        # # if kapp < 0.01, there is no need to train any more
        # if scores["kappa"] < 0.01 and epoch >= 1:
        #     print("training terminated for no accuray")
        #     break

        # early_stopping needs the monitor to check if it has improved,
        # and if it has, it will make a checkpoint of the current model
        score = scores[monitor]
        if "loss" in monitor:
            score = -score

        early_stopping(score, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # Uncomment to save model of last epoch, comment to save mode before early stopping
    # torch.save(model.state_dict(), model_save_path)




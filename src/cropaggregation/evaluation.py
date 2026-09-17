import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from .metrics import metrics

def _to_tensor_labels(y):
    if sys.platform == "win32":
        return torch.from_numpy(y.astype(np.int64))
    if "linux" in sys.platform:
        return torch.from_numpy(y.astype(np.int_))
    return torch.from_numpy(y.astype(np.int64))


def save_temperature(path, temperature):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{float(temperature):.8f}\n")


def load_temperature(path, default=1.0):
    try:
        with open(path, "r", encoding="utf-8") as f:
            value = float(f.read().strip())
        if np.isfinite(value) and value > 0:
            return value
    except Exception:
        pass
    return default


def tune_temperature(model, X, y, device, batch_size_list=None, max_iter=50):
    """
    Learn a scalar temperature on a held-out set for probability calibration.
    The model weights are frozen; only the positive temperature is optimized.
    """
    if batch_size_list is None:
        batch_size_list = [4096, 8192, 8192]
    if X is None or y is None or len(y) == 0:
        return 1.0

    X = torch.from_numpy(X.astype(np.float32))
    y = _to_tensor_labels(y)
    dataloader = DataLoader(TensorDataset(X, y), batch_size_list[1], shuffle=False)

    logits_list = []
    labels_list = []
    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in tqdm(dataloader, desc="Calibration batches", unit="batch", leave=False):
            logits_list.append(model(x_batch.to(device)).detach().cpu())
            labels_list.append(y_batch.detach().cpu())

    logits = torch.cat(logits_list).to(device)
    labels = torch.cat(labels_list).to(device)
    log_temperature = torch.zeros(1, device=device, requires_grad=True)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.05, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature).clamp(0.05, 10.0)
        loss = criterion(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = torch.exp(log_temperature).detach().clamp(0.05, 10.0).item()
    return float(temperature)


def evaluate(model, X, y, device, batch_size_list=None, temperature=1.0, mc_samples=30):

    if batch_size_list is None:
        batch_size_list = [4096, 8192, 8192]
    # covert numpy to pytorch tensor and put into gpu
    X = torch.from_numpy(X.astype(np.float32))
    y = _to_tensor_labels(y)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size_list[1], shuffle=False)

    model.eval()
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    with torch.no_grad():
        losses = list()
        y_true_list = list()
        y_pred_list = list()
        y_score_list = list()
        epistemic_list = []
        for x, y_true in tqdm(dataloader, desc="Evaluation batches", unit="batch", leave=False):
            mean_prob, epistemic = mc_dropout_predict(
                model,
                x.to(device),
                T=max(1, int(mc_samples)),
                temperature=temperature
            )

            log_proba = torch.log(
                mean_prob + 1e-8
            )

            epistemic_list.append(
                epistemic.cpu()
            )
            loss = criterion(log_proba, y_true.to(device))
            losses.append(loss)
            y_true_list.append(y_true)
            y_pred_list.append(log_proba.argmax(-1))
            y_score_list.append(log_proba.exp())

    return (
        torch.stack(losses),
        torch.cat(y_true_list),
        torch.cat(y_pred_list),
        torch.cat(y_score_list),
        torch.cat(epistemic_list)
    )


def enable_dropout(model):
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()


def mc_dropout_predict(model, x, T=30, temperature=1.0):
    original_mode = model.training
    enable_dropout(model)

    preds = []
    with torch.no_grad():
        for _ in range(T):
            logits = model(x)
            logits = logits / max(float(temperature), 1e-6)
            probs = F.softmax(logits, dim=1)
            preds.append(probs)

    preds = torch.stack(preds)
    mean_prob = preds.mean(0)
    eps = 1e-8
    total_entropy = -torch.sum(mean_prob * torch.log(mean_prob + eps), dim=1)
    expected_entropy = -torch.sum(preds * torch.log(preds + eps), dim=2).mean(0)
    epistemic = torch.clamp(total_entropy - expected_entropy, min=0.0)

    if original_mode:
        model.train()
    else:
        model.eval()

    return mean_prob, epistemic


def mc_dropout_decompose(model, x, T=30, temperature=1.0):
    """Return mean probabilities plus total, aleatoric and epistemic uncertainty."""
    original_mode = model.training
    enable_dropout(model)

    preds = []
    with torch.no_grad():
        for _ in range(T):
            logits = model(x)
            logits = logits / max(float(temperature), 1e-6)
            preds.append(F.softmax(logits, dim=1))

    preds = torch.stack(preds)
    mean_prob = preds.mean(0)
    eps = 1e-8
    total_entropy = -torch.sum(mean_prob * torch.log(mean_prob + eps), dim=1)
    aleatoric = -torch.sum(preds * torch.log(preds + eps), dim=2).mean(0)
    epistemic = torch.clamp(total_entropy - aleatoric, min=0.0)

    if original_mode:
        model.train()
    else:
        model.eval()

    return mean_prob, total_entropy, aleatoric, epistemic


def inference(model, X, n_classes, device, batch_size_list=None, return_epistemic=False, temperature=1.0):
    if batch_size_list is None:
        batch_size_list = [4096, 8192, 8192]
    X = torch.from_numpy(X.astype(np.float32))
    dataloader = DataLoader(X, batch_size_list[2], shuffle=False)

    original_mode = model.training
    model.eval()  # 默认关闭 Dropout

    y_predict_array = np.zeros((X.shape[0], n_classes), dtype=np.float32)
    epistemic_array = np.zeros((X.shape[0],), dtype=np.float32) if return_epistemic else None

    start = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference batches", unit="batch", leave=False):
            batch = batch.to(device)
            batch_size = batch.shape[0]

            if return_epistemic:
                mean_prob, epistemic = mc_dropout_predict(model, batch, T=30, temperature=temperature)
                epistemic_array[start:start + batch_size] = epistemic.cpu().numpy()
            else:
                logits = model(batch)
                logits = logits / max(float(temperature), 1e-6)
                mean_prob = F.softmax(logits, dim=1)

            # 直接保存Softmax概率
            y_predict_array[start:start + batch_size, :] = mean_prob.cpu().numpy()

            start += batch_size

    # 恢复原始模式
    if original_mode:
        model.train()
    else:
        model.eval()

    if return_epistemic:
        return y_predict_array, epistemic_array
    else:
        return y_predict_array


def transfer_evaluate(model, X, y, log_path, device, batch_size_list=None):
    if batch_size_list is None:
        batch_size_list = [4096, 8192, 8192]
    model.eval()
    test_loss, y_true, y_pred, y_score, epistemic = evaluate(model, X, y, device, batch_size_list)
    Classes = [f'class {i}' for i in np.unique(y_true.cpu())]
    scores = metrics(
        y_true.cpu().numpy(),
        y_pred.cpu().numpy(),
        y_score.cpu().numpy(),
        Classes
    )
    scores["Epistemic"] = epistemic.mean().item()
    if "Entropy" in scores:
        scores["Aleatoric"] = max(0.0, scores["Entropy"] - scores["Epistemic"])
    else:
        scores["Aleatoric"] = float('nan')
    print(f"OA={scores['accuracy']:.4f}, F1={scores['f1_macro']:.4f}, "
          f"Entropy={scores['Entropy']:.4f}, Aleatoric={scores['Aleatoric']:.4f}, Epistemic={scores['Epistemic']:.4f}")
    test_loss = test_loss.cpu().detach().numpy()[0]

    log = []
    scores["test_loss"] = test_loss
    log.append(scores)
    log_df = pd.DataFrame(log)
    log_df.to_csv(os.path.join(log_path, "transfer_log.csv"))



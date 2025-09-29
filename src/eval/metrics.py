import os, csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay
)

@torch.no_grad()
def predict_on_loader(net, loader: DataLoader, device):
    net.eval()
    crit = nn.CrossEntropyLoss(reduction="sum")
    total_loss, total_examples = 0.0, 0
    logits_all, y_all = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = net(X)
        loss = crit(logits, y)
        total_loss += loss.item()
        total_examples += y.size(0)
        logits_all.append(logits.detach().cpu())
        y_all.append(y.detach().cpu())
    logits = torch.cat(logits_all, dim=0)
    y_true = torch.cat(y_all, dim=0).numpy()
    y_proba = torch.softmax(logits, dim=1).numpy()
    y_pred = logits.argmax(dim=1).numpy()
    avg_loss = total_loss / max(1, total_examples)
    return avg_loss, y_true, y_pred, y_proba

def compute_global_metrics(y_true, y_pred, y_proba, num_classes: int):
    acc = float(accuracy_score(y_true, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    try:
        auroc = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
    except Exception:
        auroc = float("nan")
    return {
        "accuracy": acc,
        "macro_precision": float(prec),
        "macro_recall": float(rec),
        "macro_f1": float(f1),
        "macro_auroc": auroc,
    }

def save_round_metrics(csv_path: str, round_id: int, loss: float, m: dict):
    header = ["round", "loss", "accuracy", "macro_precision", "macro_recall", "macro_f1", "macro_auroc"]
    row = [round_id, loss, m["accuracy"], m["macro_precision"], m["macro_recall"], m["macro_f1"], m["macro_auroc"]]
    exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(header)
        w.writerow(row)

def save_confusion_matrix_png(y_true, y_pred, class_names, out_path: str, title: str = ""):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d', colorbar=False)
    if title:
        ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

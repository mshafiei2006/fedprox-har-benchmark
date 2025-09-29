import os, time, argparse
from collections import OrderedDict, defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import flwr as fl
from flwr.common import Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.simulation import run_simulation
from flwr.client import NumPyClient, ClientApp
from flwr.server.strategy import FedProx


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Project utils
from src.data.dataset import MotionSenseDataset, create_sequences, FEATURE_COLS, TARGET_MAP, IDX_TO_LABEL
from src.eval.metrics import (
    predict_on_loader, compute_global_metrics, save_round_metrics, save_confusion_matrix_png
)

# ===== Config =====
torch.manual_seed(42)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "datasets/MotionSense Dataset/Cleaned data"
SEQ_LEN = 64
STEP = 32
BATCH_SIZE = 32
NUM_EPOCHS = 5
NUM_ROUNDS = 20
FRACTION_FIT = 1.0
MIN_FIT_CLIENTS = 12
LEARNING_RATE = 1e-3
PROXIMAL_MU = 0.1
GRAD_CLIP = 1.0

OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

# ===== Models (baseline only here; plug your others if needed) =====
class MotionSenseLSTM(nn.Module):
    def __init__(self, input_dim=len(FEATURE_COLS), hidden_dim=64, num_classes=len(TARGET_MAP)):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim, num_layers=1,
            batch_first=True, bidirectional=False, dropout=0.0
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

def set_parameters(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

def train_fedprox(net, trainloader, mu: float, lr: float, local_epochs: int):
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    global_params = OrderedDict((k, v.detach().clone().to(DEVICE)) for k, v in net.state_dict().items())
    for _ in range(local_epochs):
        for X, y in trainloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            logits = net(X)
            base_loss = crit(logits, y)
            prox = 0.0
            for (name, w), w_g in zip(net.state_dict().items(), global_params.values()):
                if "num_batches_tracked" in name:
                    continue
                prox += torch.sum((w - w_g) ** 2)
            loss = base_loss + 0.5 * mu * prox
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
            opt.step()

# ===== Load per-client CSVs and create windows =====
def load_all_clients() -> Dict[str, pd.DataFrame]:
    data = {}
    for cid in range(1, 25):
        p = os.path.join(DATA_DIR, f"Client_{cid}.csv")
        if os.path.isfile(p):
            data[str(cid)] = pd.read_csv(p)
    return data

def build_sequences_by_client(dfs: Dict[str, pd.DataFrame]) -> Dict[str, List[Tuple[np.ndarray, str]]]:
    out = {}
    for cid, df in dfs.items():
        out[cid] = create_sequences(df, seq_len=SEQ_LEN, step=STEP)
    return out

def fit_global_scaler(train_sequences: Dict[str, List[Tuple[np.ndarray, str]]]) -> StandardScaler:
    windows = [x for pairs in train_sequences.values() for (x, _) in pairs]
    X = np.stack(windows, axis=0)                 # (N, T, F)
    X2d = X.reshape(-1, X.shape[-1])              # (N*T, F)
    scaler = StandardScaler().fit(X2d)
    return scaler

def apply_scaler(seq_dict: Dict[str, List[Tuple[np.ndarray, str]]], scaler: StandardScaler):
    out = {}
    for cid, pairs in seq_dict.items():
        norm_pairs = []
        for x, y in pairs:
            X2d = x.reshape(-1, x.shape[-1])
            x_norm = scaler.transform(X2d).reshape(x.shape).astype(np.float32)
            norm_pairs.append((x_norm, y))
        out[cid] = norm_pairs
    return out

# ===== Flower client =====
class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, cid):
        self.net = net
        self.trainloader = trainloader
        self.cid = cid

    def get_parameters(self, config=None):
        return [v.cpu().numpy() for v in self.net.state_dict().values()]

    def set_parameters(self, parameters):
        set_parameters(self.net, parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        mu = float(config.get("proximal_mu", PROXIMAL_MU))
        lr = float(config.get("lr", LEARNING_RATE))
        local_epochs = int(config.get("local_epochs", NUM_EPOCHS))
        start = time.perf_counter()
        train_fedprox(self.net, self.trainloader, mu=mu, lr=lr, local_epochs=local_epochs)
        elapsed = time.perf_counter() - start
        return self.get_parameters(), len(self.trainloader.dataset), {"client_train_time": float(elapsed)}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        return 0.0, 0, {"client_val_accuracy": 0.0}

def client_fn_factory(train_ids, train_sequences):
    def client_fn(context: Context):
        pid = int(context.node_config["partition-id"])
        cid = train_ids[pid]
        ds = MotionSenseDataset(train_sequences[cid])
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
        net = MotionSenseLSTM().to(DEVICE)  # swap if you add other models
        return FlowerClient(net, loader, cid).to_client()
    return client_fn

# ===== Server eval & metrics logging =====
def get_evaluate_fn(test_loader: DataLoader, total_rounds: int, run_tag: str):
    class_names = [IDX_TO_LABEL[i] for i in sorted(IDX_TO_LABEL)]
    def evaluate(server_round, parameters, config):
        net = MotionSenseLSTM().to(DEVICE)
        set_parameters(net, parameters)
        loss, y_true, y_pred, y_proba = predict_on_loader(net, test_loader, DEVICE)
        m = compute_global_metrics(y_true, y_pred, y_proba, num_classes=len(TARGET_MAP))
        # Save per-round metrics
        save_round_metrics(
            os.path.join(OUT_DIR, f"history_{run_tag}.csv"),
            server_round, loss, m
        )
        # Save confusion matrix ONLY for final round for the paper
        if server_round == total_rounds:
            save_confusion_matrix_png(
                y_true, y_pred, class_names,
                os.path.join(OUT_DIR, f"cm_round{server_round}_{run_tag}.png"),
                title=f"Confusion Matrix (Round {server_round})"
            )
        print(f"[Round {server_round}] Acc {m['accuracy']*100:.2f}% | F1 {m['macro_f1']:.4f} | "
              f"Prec {m['macro_precision']:.4f} | Rec {m['macro_recall']:.4f} | AUROC {m['macro_auroc']:.4f}")
        return loss, {
            "test_accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "macro_precision": m["macro_precision"],
            "macro_recall": m["macro_recall"],
            "macro_auroc": m["macro_auroc"],
        }
    return evaluate

def fit_metrics_aggregation_fn(results):
    total_examples = sum(n for n, _ in results) or 1
    avg_time = sum(n * d.get("client_train_time", 0.0) for n, d in results) / total_examples
    return {"avg_client_train_time": float(avg_time)}

# ===== Main =====
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default="lstm_baseline")
    args = parser.parse_args()

    # Load and split clients deterministically (same as your script)
    dfs = load_all_clients()
    all_ids = sorted(dfs.keys())
    rng = np.random.default_rng(42)
    rng.shuffle(all_ids)
    split = int(0.8 * len(all_ids))
    train_ids, test_ids = all_ids[:split], all_ids[split:]
    assert not set(train_ids) & set(test_ids)

    train_seqs = build_sequences_by_client({cid: dfs[cid] for cid in train_ids})
    test_seqs  = build_sequences_by_client({cid: dfs[cid] for cid in test_ids})

    # Fit scaler on TRAIN only, apply to both
    scaler = fit_global_scaler(train_seqs)
    train_seqs = apply_scaler(train_seqs, scaler)
    test_seqs  = apply_scaler(test_seqs, scaler)

    # Central test loader
    test_all = [pair for pairs in test_seqs.values() for pair in pairs]
    test_loader = DataLoader(MotionSenseDataset(test_all), batch_size=BATCH_SIZE, shuffle=False)

    # Strategy
    strategy = FedProx(
        fraction_fit=FRACTION_FIT,
        fraction_evaluate=0.0,
        min_fit_clients=MIN_FIT_CLIENTS,
        min_evaluate_clients=0,
        min_available_clients=len(train_ids),
        evaluate_fn=get_evaluate_fn(test_loader, NUM_ROUNDS, run_tag=args.tag),
        proximal_mu=PROXIMAL_MU,
        on_fit_config_fn=lambda rnd: {"proximal_mu": PROXIMAL_MU, "lr": LEARNING_RATE, "local_epochs": NUM_EPOCHS},
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
    )

    print(f"Starting FL simulation with {len(train_ids)} clients")
    run_simulation(
        server_app=ServerApp(server_fn=lambda ctx: ServerAppComponents(
            strategy=strategy, config=ServerConfig(num_rounds=NUM_ROUNDS)
        )),
        client_app=ClientApp(client_fn=client_fn_factory(train_ids, train_seqs)),
        num_supernodes=len(train_ids),
        backend_config={"client_resources": {"num_cpus": 1, "num_gpus": 0.0}},
    )

if __name__ == "__main__":
    main()

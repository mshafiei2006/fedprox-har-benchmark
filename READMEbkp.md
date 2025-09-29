# Federated HAR with FedProx: Benchmarking LSTM, PatchTST, and MLP on MotionSense

Privacy-preserving human activity recognition (HAR) using smartphones is challenging because real users generate **non-IID** data. This repo implements a **horizontal federated learning** setup with **FedProx** (μ=0.1) and benchmarks three neural families — **LSTM**, **PatchTST (Transformer)**, and **MLP** ; on the **MotionSense** dataset (24 users, 6 activities, 50 Hz).

> **Key finding:** Enhanced variants (BiLSTM-Proj, PatchTST-CLS, MLPMixer) increase training time without consistent accuracy gains. Baselines offer the best **efficiency/accuracy trade-off**:
>
> - **LSTM**: best accuracy (≈91.5%)  
> - **PatchTST**: balanced performance, top AUROC (≈98.4%)  
> - **MLP**: fastest (<1s/client/round) with competitive accuracy (≈89.7%)

---

## 1) Highlights

- **Federated algorithm:** FedProx (μ=0.1) to stabilize training under non-IID user distributions.  
- **Protocol:** 20 rounds, 12 train clients per round, 5 local epochs, batch size 32, Adam (LR=1e-3), grad clip=1.0.  
- **Preprocessing:** sliding windows (length=64, stride=32), **z-score normalisation fit on train users only**, applied to train & test.  
- **Evaluation:** centralized test over held-out users; Accuracy, Macro-F1, Macro-Precision/Recall, Macro-AUROC; Round-20 confusion matrix; average client train time.

---

## 2) Repository structure

```
fedprox-har-benchmark/
├─ datasets/
│  ├─ MotionSense Dataset/
│  │  ├─ dws_2/ sub_*.csv
│  │  ├─ jog_9/ sub_*.csv
│  │  ├─ sit_5/ sub_*.csv
│  │  ├─ std_6/ sub_*.csv
│  │  ├─ ups_4/ sub_*.csv
│  │  └─ wlk_8/ sub_*.csv
│  └─ Cleaned data/Client_*.csv
│
├─ src/
│  ├─ data/
│  │  ├─ prepare_motionsense.py      # cleaning script
│  │  └─ dataset.py                  # windowing + Dataset
│  ├─ models/
│  │  ├─ lstm.py                     # baseline + enhanced
│  │  ├─ patchtst.py                 # baseline + enhanced
│  │  └─ mlp.py                      # baseline + enhanced
│  ├─ fl/
│  │  └─ simulate_fedprox.py         # FL runner (FedProx)
│  └─ eval/
│     ├─ metrics.py                  
│     ├─ plots.py                    
│     └─ aggregate_history.py        
│
├─ results/ 
│  ├─ confusion_matrices/ 
│  │  ├─ LSTM - Baseline.png 
│  │  ├─ PatchTST - Baseline.png 
│  │  └─ MLP - Baseline.png 
│  ├─ logs/
│  │  ├─ lstm_fedprox.log 
│  │  ├─ patchtst_fedprox.log 
│  │  └─ mlp_fedprox.log 
│  ├─ metrics/
│  │  ├─ lstm_history.csv 
│  │  ├─ patchtst_history.csv 
│  │  └─ mlp_history.csv 
│  └─ plots/
│     ├─ bar.png 
│     ├─ curves.png 
│     ├─ radar.png 
│     └─ time.png
│
├─ README.md
├─ LICENSE
├─ requirements.txt
└─ .gitignore
```

---

## 3) Setup

### Option A — conda (recommended)
```bash
conda create -n fedprox-har python=3.12 -y
conda activate fedprox-har
pip install torch --index-url https://download.pytorch.org/whl/cpu  # or CUDA if available
pip install flwr==1.9.* pandas numpy scikit-learn matplotlib
```

### Option B — pip
```bash
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

## 4) Data preparation



## 5) Data preparation

1. Place the raw MotionSense folders under datasets/MotionSense Dataset/ as shown above.
2. Run the cleaner (writes one CSV per client under Cleaned data/):
```bash
python src/data/prepare_motionsense.py
```

---

## 6) Run a federated experiment (LSTM baseline)

```bash
python -m src.fl.simulate_fedprox --tag lstm_baseline
```

This will:
- Load 24 per-client CSVs.
- Split **users** 80/20 (train/test), disjoint.
- Create windows (64, step 32).
- **Fit global scaler on train** users’ windows; apply to train & test.
- Run FedProx for 20 rounds (24 clients/round).
- Log per-round metrics to `results/metrics/*_history.csv`.
- Save the Round-20 confusion matrix to `results/onfusion_matrices/* - Baseline.png`.
- Save client training logs to: `results/logs/*_fedprox.log`.

> To swap model families later, either (a) change the model class in `simulate_fedprox.py` or (b) add a `--model` arg and branch on it.

---

## 7) Plotting & aggregation

- **Curves (Accuracy/F1/AUROC)** across rounds:
```python
from src.eval.plots import plot_curves
paths = [
  "results/metrics/history_lstm_baseline.csv",
  "results/metrics/history_patchtst_baseline.csv",
  "results/metrics/history_mlp_baseline.csv",
]
labels = ["LSTM", "PatchTST", "MLP"]
plot_curves(paths, labels, metric="accuracy", out_path="results/plots/curves_acc.png", title="Accuracy vs Rounds")
```

- **Bar at Round-20** (Acc/F1/AUROC):
```python
from src.eval.plots import bar_at_round20
bar_at_round20(paths, labels, out_path="results/plots/bar.png", title="Round-20 Summary")
```

- **Radar chart** (add normalized efficiency if you want):
```python
from src.eval.plots import radar_summary
eff = {"LSTM": 0.3, "PatchTST": 0.55, "MLP": 1.0}  # example scaling
radar_summary(paths, labels, efficiency=eff, out_path="results/plots/radar.png", title="Trade-offs")
```

- **Combine histories** & print Round-20 table:
```bash
python -m src.eval.aggregate_history
```

---

## 8) Results (example numbers from the dissertation)

**Round 20 (baselines):**
- **LSTM** — Acc **0.915**, Macro-F1 **0.861**, AUROC **0.983**, time ≈ **4.8s/client/round**
- **PatchTST** — Acc **0.901**, Macro-F1 **0.850**, AUROC **0.984**, time ≈ **3.7s**
- **MLP** — Acc **0.897**, Macro-F1 **0.847**, AUROC **0.978**, time ≈ **0.8s**

**Takeaways**
- Enhanced variants (BiLSTM-Proj, PatchTST-CLS, MLPMixer) → **higher cost**, no consistent gains.  
- Choose model based on deployment needs:
  - **Accuracy priority** → LSTM  
  - **Efficiency priority** → MLP  
  - **Balanced** → PatchTST

---

## 9) Repro tips

- Fix the RNG seed (already set to 42) for repeatable splits.  
- If you want to evaluate **only** Round-20 confusion matrices, ensure `simulate_fedprox.py` saves CM on `server_round == NUM_ROUNDS` (already implemented in `get_evaluate_fn`).  
- CPU vs GPU: The code runs CPU-only; enable CUDA by installing the right torch build.

---

## 10) License

MIT License. See `LICENSE`.

---

## 11) Citation

If you use this repo, please cite:

```
@misc{shafiei2025fedproxhar,
  title = {Federated Learning with FedProx for Human Activity Recognition: A Comparative Benchmark of Neural Architectures},
  author = {Shafiei, Mohammad},
  year = {2025},
  note = {University of Lincoln MSc Dissertation Project},
}
```

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_curves(history_csv_paths, labels, metric="accuracy", out_path=None, title=None):
    plt.figure(figsize=(6,4), dpi=130)
    for p, lab in zip(history_csv_paths, labels):
        df = pd.read_csv(p)
        plt.plot(df["round"], df[metric], label=lab)
    plt.xlabel("Round")
    plt.ylabel(metric)
    if title: plt.title(title)
    plt.legend()
    plt.tight_layout()
    if out_path: plt.savefig(out_path)
    plt.close()

def bar_at_round20(history_csv_paths, labels, metrics=("accuracy","macro_f1","macro_auroc"), out_path=None, title=None):
    vals = []
    for p in history_csv_paths:
        df = pd.read_csv(p)
        row = df.iloc[df["round"].idxmax()]
        vals.append([row[m] for m in metrics])
    vals = np.array(vals)  # (runs, metrics)
    x = np.arange(len(labels))
    w = 0.25
    plt.figure(figsize=(7,4), dpi=130)
    for i, m in enumerate(metrics):
        plt.bar(x + i*w - w, vals[:, i], width=w, label=m)
    plt.xticks(x, labels)
    if title: plt.title(title)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()
    if out_path: plt.savefig(out_path)
    plt.close()

def radar_summary(history_csv_paths, labels, out_path=None, title=None, efficiency=None):
    # efficiency: dict {label: value in [0,1]} (higher is better). If None, skip.
    metrics = ("accuracy","macro_f1","macro_auroc")
    angles = np.linspace(0, 2*np.pi, len(metrics)+(1 if efficiency else 0), endpoint=False)
    if efficiency:
        metrics = (*metrics, "efficiency")

    plt.figure(figsize=(5,5), dpi=130)
    ax = plt.subplot(111, polar=True)
    for p, lab in zip(history_csv_paths, labels):
        df = pd.read_csv(p)
        row = df.iloc[df["round"].idxmax()]
        vals = [row["accuracy"], row["macro_f1"], row["macro_auroc"]]
        if efficiency:
            vals.append(efficiency.get(lab, 0.0))
        vals = np.array(vals)
        vals = np.concatenate([vals, vals[:1]])   # close the loop
        ang = np.concatenate([angles, angles[:1]])
        ax.plot(ang, vals, label=lab)
        ax.fill(ang, vals, alpha=0.1)
    ax.set_xticks(angles)
    ax.set_xticklabels(metrics)
    if title: plt.title(title, y=1.08)
    plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10))
    plt.tight_layout()
    if out_path: plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def plot_time_bars(avg_time_dict, out_path=None, title=None):
    # avg_time_dict: {label: seconds}
    labels = list(avg_time_dict.keys())
    vals = [avg_time_dict[k] for k in labels]
    plt.figure(figsize=(5,3.2), dpi=130)
    plt.bar(labels, vals)
    plt.ylabel("Avg client train time (s)")
    if title: plt.title(title)
    plt.tight_layout()
    if out_path: plt.savefig(out_path)
    plt.close()

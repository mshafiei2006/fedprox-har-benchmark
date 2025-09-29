import os, glob
import pandas as pd

def load_histories(pattern="results/history_*.csv"):
    paths = glob.glob(pattern)
    frames = []
    for p in paths:
        tag = os.path.basename(p).replace("history_","").replace(".csv","")
        df = pd.read_csv(p)
        df["run_tag"] = tag
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def last_round_summary(df: pd.DataFrame):
    # For each run_tag, take the last round
    idx = df.groupby("run_tag")["round"].idxmax()
    return df.loc[idx].sort_values("run_tag")

if __name__ == "__main__":
    df = load_histories()
    if df.empty:
        print("No histories found.")
    else:
        print(last_round_summary(df)[["run_tag","round","accuracy","macro_f1","macro_auroc"]])

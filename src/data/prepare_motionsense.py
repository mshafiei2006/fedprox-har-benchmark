import os
import pandas as pd
import numpy as np

# === Config ===

DATA_DIR = os.path.join("datasets", "MotionSense Dataset")
CLEANED_DATA_DIR = os.path.join(DATA_DIR, "Cleaned data")
os.makedirs(CLEANED_DATA_DIR, exist_ok=True)

Activity_MAP = {
    "dws_2": "Downstairs",
    "jog_9": "Jogging",
    "sit_5": "Sitting",
    "std_6": "Standing",
    "ups_4": "Upstairs",
    "wlk_8": "Walking"
}
FEATURE_COLUMNS = [
    "attitude.roll", "attitude.pitch", "attitude.yaw",
    "gravity.x", "gravity.y", "gravity.z",
    "rotationRate.x", "rotationRate.y", "rotationRate.z",
    "userAcceleration.x", "userAcceleration.y", "userAcceleration.z"
]
TARGET_COLUMN = "Activity"

# === Load, clean, and save per-client (ordered by client then activity)
print("Loading and cleaning raw files (Client → Activity order)...\n")

for client_id in map(str, range(1, 25)):  # i.e., "1" to "24":
    client_frames = []

    for folder, activity in Activity_MAP.items():
        folder_path = os.path.join(DATA_DIR, folder)
        file_name = f"sub_{int(client_id)}.csv"
        file_path = os.path.join(folder_path, file_name)

        if not os.path.isfile(file_path):
            continue  # File might be missing for some classes

        df = pd.read_csv(file_path)

        # Fix timestamp column
        if "Timestamp" not in df.columns:
            if df.columns[0].startswith("Unnamed"):
                df.rename(columns={df.columns[0]: "Timestamp"}, inplace=True)
            else:
                df.insert(0, "Timestamp", range(len(df)))

        # Keep only relevant columns and label
        df = df[["Timestamp"] + FEATURE_COLUMNS]
        df[TARGET_COLUMN] = activity

        # Drop NaNs
        before = len(df)
        df.dropna(inplace=True)
        after = len(df)
        nan_msg = f"⚠️ Dropped {before - after} NaNs" if before != after else "✔️ No NaNs"


        # Fix dtypes
        fix_log = []
        if df["Timestamp"].dtype != "int32":
            df["Timestamp"] = df["Timestamp"].astype("int32")
            fix_log.append("Timestamp")

        for col in FEATURE_COLUMNS:
            if df[col].dtype != "float64":
                df[col] = df[col].astype("float64")
                fix_log.append(col)

        if df[TARGET_COLUMN].dtype != "object":
            df[TARGET_COLUMN] = df[TARGET_COLUMN].astype("object")
            fix_log.append("Activity")

        fix_status = f"✔️ Fixed dtypes: {fix_log}" if fix_log else "✔️ All dtypes OK"
        
        print(f"[Client {client_id} - {activity}] {nan_msg} | {fix_status} | Number of rows: {len(df)}")

        client_frames.append(df)

# === Save one file per client ===
    if client_frames:
        merged = pd.concat(client_frames, ignore_index=True)
        merged["Timestamp"] = np.arange(1, len(merged) + 1) 
        print(f"→ [Client {client_id}] Merged rows: {len(merged)}\n")
        out_path = os.path.join(CLEANED_DATA_DIR, f"Client_{client_id}.csv")
        merged.to_csv(out_path, index=False)
    else:
        print(f"⚠️ No data found for Client {client_id}\n")

print("🎉 All clients cleaned and saved.")
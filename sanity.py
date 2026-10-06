from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Scan current directory for subfolders containing `<foldername>_preprocessed.csv`
current_dir = Path(".")
dataset_files = sorted([
    folder / f"{folder.name}_preprocessed.csv"
    for folder in current_dir.iterdir()
    if folder.is_dir() and (folder / f"{folder.name}_preprocessed.csv").exists()
])

# Fallback to any `*_preprocessed.csv` files in the current folder if no subfolders match
if not dataset_files:
    dataset_files = sorted(list(current_dir.glob("*_preprocessed.csv")))

if not dataset_files:
    raise FileNotFoundError("No matching preprocessed CSV files found.")

print(f"Found {len(dataset_files)} dataset(s) to evaluate independently.\n")

# 2. Iterate and train a separate RandomForestRegressor per dataset
for fpath in dataset_files:
    print("=" * 50)
    print(f"Dataset: {fpath}")
    print("=" * 50)
    
    # Load dataset & clean column whitespace
    df = pd.read_csv(fpath)
    df.columns = df.columns.str.strip()
    
    # Identify target columns starting with 'target_'
    target_cols = sorted([col for col in df.columns if col.startswith("target_")])
    if not target_cols:
        print(f"Skipping {fpath}: No columns starting with 'target_' found.\n")
        continue

    X = df.drop(columns=target_cols)
    y = df[target_cols]
    
    print(f"Samples: {len(df):,} | Features: {X.shape[1]} | Targets ({len(target_cols)}): {target_cols}")

    # Initialize and fit a new Random Forest model on this dataset
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)

    # Evaluate predictions on full dataset
    preds = model.predict(X)

    mae = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))

    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

    # Compute Euclidean distance if targets represent 2D coordinates (e.g. lat/lon)
    if len(target_cols) == 2 and any("lat" in c.lower() or "lon" in c.lower() for c in target_cols):
        euc_dist = np.sqrt(np.sum((preds - y.values) ** 2, axis=1)).mean()
        print(f"Mean Positioning Error (Euclidean Distance): {euc_dist:.2f}m")
    
    print()
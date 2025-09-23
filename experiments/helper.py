from pathlib import Path
import pandas as pd
import numpy as np

def _extract_acquisition_from_path(p):
    """
    Extract acquisition token {acq} from filename: {label}_{acq}_{mode}_{idx}.npy
    Example: data/.../B_15_1_95.npy -> 15
    """
    fname = Path(str(p)).name
    parts = fname.split("_")
    if len(parts) < 4:
        raise ValueError(f"Unexpected file name format for grouping: {fname}")
    acq = parts[1]
    try:
        return int(acq)
    except ValueError:
        return acq  # fallback to string if not numeric


def load_xy_groups(csv_path):
    """
    Load X, y, groups from CSV:
      - features: columns starting with 'f'
      - labels: 'label' (int) or 'label_str' (mapped to int)
      - groups: acquisition parsed from 'path'
    """
    df = pd.read_csv(csv_path)

    if "path" not in df.columns:
        raise ValueError(f"CSV must contain 'path' column: {csv_path}")

    feat_cols = [c for c in df.columns if c.startswith("f")]
    if not feat_cols:
        raise ValueError(f"No feature columns starting with 'f' in {csv_path}")

    if "label" in df.columns:
        y = df["label"].values
    elif "label_str" in df.columns:
        mapping = {s: i for i, s in enumerate(sorted(df["label_str"].astype(str).unique()))}
        y = df["label_str"].map(mapping).values
    else:
        raise ValueError(f"No 'label' or 'label_str' column in {csv_path}")

    groups = df["path"].apply(_extract_acquisition_from_path).values
    X = df[feat_cols].to_numpy(dtype=np.float32)
    return X, y, groups


def load_best_params(json_path):
    """
    Load best hyperparameters from JSON file.
    """
    import json

    # Load parameters
    with open(json_path, "r") as f:
        params = json.load(f)
    return params
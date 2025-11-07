# -*- coding: utf-8 -*-
"""
mirror3d_cleaner.py
-------------------------------------------------------------
Processes the raw tracking CSV output from the tracker to smooth noisy 
data and reject outliers based on motion and velocity thresholds.

The program produces a cleaned, analysis-ready dataset for 
visualization or quantitative study.

It loads parameters from mirror3d_config.ini via mirror3d_config.py.

[cleaner]
input_csv = ./output/mirror3d_track_raw.csv
output_csv = ./output/mirror3d_track_clean.csv

MIT License
Copyright (c) 2025 Thomas Zimmerman
See https://opensource.org/licenses/MIT for full license text.
-------------------------------------------------------------
"""

import os
import csv
import numpy as np
from collections import defaultdict
from mirror3d_config import get_cleaner_params


# -------------------------- Load Configuration --------------------------
cfg = get_cleaner_params() or {}

smooth_radius = int(cfg.get("smooth_radius", 5))
motion_threshold = float(cfg.get("motion_threshold", 3.2))
force_overwrite = bool(cfg.get("force_overwrite", False))

input_csv = cfg.get("input_csv", "./output/mirror3d_track_raw.csv")
output_csv = cfg.get("output_csv", "./output/mirror3d_track_clean.csv")
save_path = os.path.dirname(output_csv)
os.makedirs(save_path, exist_ok=True)

print(f"[INFO] Loaded cleaner config:")
print(f"       smooth_radius={smooth_radius}")
print(f"       motion_threshold={motion_threshold}")
print(f"       force_overwrite={force_overwrite}")
print(f"       input_csv={input_csv}")
print(f"       output_csv={output_csv}")


# -------------------------- Utility Functions --------------------------
def moving_average(data, radius):
    """Apply a simple centered moving average smoothing."""
    if radius < 1:
        return data
    kernel = np.ones(2 * radius + 1) / (2 * radius + 1)
    return np.convolve(data, kernel, mode="same")


def detect_motion_jumps(z_series, threshold):
    """Return indices of motion jumps that exceed the threshold."""
    diffs = np.abs(np.diff(z_series, prepend=z_series[0]))
    return np.where(diffs > threshold)[0]


# -------------------------- Main Cleaning Logic --------------------------
def clean_tracks():
    if not os.path.exists(input_csv):
        print(f"[ERROR] Input file not found: {input_csv}")
        return

    if os.path.exists(output_csv) and not force_overwrite:
        print(f"[WARN] Output file exists: {output_csv}")
        print("       Use force_overwrite=True to overwrite.")
        return

    with open(input_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("[ERROR] No rows found in input CSV.")
        return

    header = reader.fieldnames
    cleaned_rows = []

    # Group by pair_id
    grouped = defaultdict(list)
    for r in rows:
        for k in ["x1", "y1", "x2", "y2", "Z", "areaA", "areaB"]:
            r[k] = float(r[k])
        r["frame"] = int(r["frame"])
        r["pair_id"] = int(r["pair_id"])
        grouped[r["pair_id"]].append(r)

    # --- Smooth each pair and mark motion jumps ---
    for pid, items in grouped.items():
        items.sort(key=lambda r: r["frame"])
        Z_series = np.array([r["Z"] for r in items])
        Z_smooth = moving_average(Z_series, smooth_radius)
        jump_idx = detect_motion_jumps(Z_smooth, motion_threshold)
        for i, r in enumerate(items):
            r["Z_smooth"] = Z_smooth[i]
            r["motion_jump"] = (i in jump_idx)
            cleaned_rows.append(r)

    # =============================================================
    # Write cleaned CSV (deduplicated header)
    # =============================================================
    print("[INFO] Saving cleaned CSV...")

    # Create unique header list
    out_fields = list(header) + ["Z_smooth", "motion_jump"]
    seen = set()
    out_fields = [f for f in out_fields if not (f in seen or seen.add(f))]

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(cleaned_rows)

    print(f"[INFO] Saved cleaned CSV to: {output_csv}")
    print(f"[INFO] Total records written: {len(cleaned_rows)}")


# -------------------------- Entry Point --------------------------
if __name__ == "__main__":
    clean_tracks()

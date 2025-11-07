# -*- coding: utf-8 -*-
"""
mirror3d_test_config.py
-------------------------------------------------------------
Quick diagnostic tool to verify that mirror3d_config.ini
and mirror3d_config.py are correctly synchronized across
the suite modules (tracker, cleaner, movie).
-------------------------------------------------------------
MIT License
Copyright (c) 2025 Thomas Zimmerman
See https://opensource.org/licenses/MIT for full license text.
"""

from mirror3d_config import (
    get_tracker_params,
    get_cleaner_params,
    get_movie_params,
    DEFAULT_CONFIG_FILE,
)
import os

print("\n[ Mirror3D Configuration Self-Test ]")
print("=====================================")
print(f"Config file path: {DEFAULT_CONFIG_FILE}")
print(f"Exists: {os.path.exists(DEFAULT_CONFIG_FILE)}\n")

# -------------------------- Tracker --------------------------
print("[tracker]")
tracker_cfg = get_tracker_params()
if tracker_cfg:
    for k, v in tracker_cfg.items():
        print(f"  {k:<15} = {v}")
else:
    print("  ⚠️  No tracker section found.")
print()

# -------------------------- Cleaner --------------------------
print("[cleaner]")
cleaner_cfg = get_cleaner_params()
if cleaner_cfg:
    for k, v in cleaner_cfg.items():
        print(f"  {k:<15} = {v}")
else:
    print("  ⚠️  No cleaner section found.")
print()

# -------------------------- Movie --------------------------
print("[movie]")
movie_cfg = get_movie_params()
if movie_cfg:
    for k, v in movie_cfg.items():
        print(f"  {k:<15} = {v}")
else:
    print("  ⚠️  No movie section found.")
print()

# -------------------------- Summary --------------------------
if all([tracker_cfg, cleaner_cfg, movie_cfg]):
    print("[✅] All sections loaded successfully!\n")
else:
    print("[⚠️] One or more sections missing — check mirror3d_config.ini.\n")

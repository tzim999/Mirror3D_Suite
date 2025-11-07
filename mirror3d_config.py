# -*- coding: utf-8 -*-
"""
mirror3d_config.py
-------------------------------------------------------------
Shared configuration loader for:
  - mirror3d_tracker.py
  - mirror3d_cleaner.py
  - mirror3d_movie.py

Uses standard INI format (mirror3d_config.ini)
All section and key names are lowercase for consistency.

MIT License
Copyright (c) 2025 Thomas Zimmerman
See https://opensource.org/licenses/MIT for full license text.
-------------------------------------------------------------
"""

import os
import configparser
from typing import Dict, Any

# =============================================================
# Default config file path (same folder as this script)
# =============================================================
DEFAULT_CONFIG_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "mirror3d_config.ini"
)

# =============================================================
# Utility: safely convert strings to numbers or booleans
# =============================================================
def _convert_value(val: str) -> Any:
    """Convert INI string values to int, float, bool, or leave as str."""
    val = val.strip()
    if val.lower() in ("true", "yes", "on"):
        return True
    if val.lower() in ("false", "no", "off"):
        return False
    try:
        if "." in val:
            return float(val)
        return int(val)
    except ValueError:
        return val

# =============================================================
# Generic section loader
# =============================================================
def load_config(section: str = None, path: str = DEFAULT_CONFIG_FILE) -> Dict[str, Any]:
    """Load configuration from mirror3d_config.ini (case-insensitive keys)."""
    config = configparser.ConfigParser()
    config.optionxform = str.lower  # ensure lowercase keys

    if not os.path.exists(path):
        print(f"[WARN] Config file not found: {path}")
        return {}

    config.read(path)

    if section and section not in config:
        print(f"[WARN] Section [{section}] not found in {os.path.basename(path)}")
        return {}

    vals = {}
    for key, val in config[section].items():
        vals[key.lower()] = _convert_value(val)
    return vals

# =============================================================
# Section-specific accessors
# =============================================================
def get_tracker_params(path: str = DEFAULT_CONFIG_FILE) -> Dict[str, Any]:
    """Return parameters from the [tracker] section."""
    return load_config("tracker", path)

def get_cleaner_params(path: str = DEFAULT_CONFIG_FILE) -> Dict[str, Any]:
    """Return parameters from the [cleaner] section."""
    return load_config("cleaner", path)

def get_movie_params(path: str = DEFAULT_CONFIG_FILE) -> Dict[str, Any]:
    """Return parameters from the [movie] section."""
    return load_config("movie", path)

# =============================================================
# Self-test utility
# =============================================================
if __name__ == "__main__":
    print("[DEBUG] Testing configuration loader...\n")
    for section in ["tracker", "cleaner", "movie"]:
        cfg = load_config(section)
        print(f"[{section}] -> {len(cfg)} keys")
        for k, v in cfg.items():
            print(f"   {k} = {v}")
        print()

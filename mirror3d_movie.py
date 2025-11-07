"""
Mirror3D Movie Generator (NumPy Version)
----------------------------------------
Renders a rotating 3D trail animation of object motion using the cleaned tracking data.
It loads parameters from mirror3d_config.ini through mirror3d_config.py.
Reads configuration from mirror3d_config.ini:
[movie]
input_csv   = ./output/mirror3d_track_clean.csv
output_mp4  = ./output/mirror3d_plot_movie.mp4

MIT License
Copyright (c) 2025 Thomas Zimmerman
See https://opensource.org/licenses/MIT for full license text.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from mirror3d_config import get_movie_params


# -------------------------- Load configuration --------------------------
cfg = get_movie_params()

input_csv = cfg.get("input_csv", "./output/mirror3d_track_clean.csv")
output_mp4 = cfg.get("output_mp4", "./output/mirror3d_plot_movie.mp4")
rotate_cube = str(cfg.get("rotate_cube", "true")).lower() == "true"
rotate_speed = float(cfg.get("rotate_speed", 0.3))
num_trails = int(cfg.get("num_trails", 30))
fps = int(cfg.get("fps", 15))

os.makedirs(os.path.dirname(output_mp4), exist_ok=True)

print("[INFO] Configuration loaded:")
print(f"   Input CSV:  {input_csv}")
print(f"   Output MP4: {output_mp4}")
print(f"   Rotate:     {rotate_cube}, Speed: {rotate_speed}")
print(f"   Trails:     {num_trails}, FPS: {fps}")


# -------------------------- Load data (NumPy) --------------------------
if not os.path.exists(input_csv):
    raise FileNotFoundError(f"Cannot find input CSV: {input_csv}")

data = np.genfromtxt(input_csv, delimiter=",", names=True, dtype=None, encoding="utf-8")

required_cols = ["frame", "pair_id", "x2", "y2", "Z"]
for col in required_cols:
    if col not in data.dtype.names:
        raise ValueError(f"Missing required column: {col}")

mask_valid = np.isfinite(data["x2"]) & np.isfinite(data["y2"]) & np.isfinite(data["Z"])
data = data[mask_valid]
if len(data) == 0:
    raise ValueError("No valid data rows found in CSV.")

print(f"[INFO] Loaded {len(data)} rows from CSV.")


# -------------------------- Rank trails --------------------------
pair_ids, counts = np.unique(data["pair_id"], return_counts=True)
sorted_idx = np.argsort(-counts)
top_ids = pair_ids[sorted_idx[:num_trails]]
if len(top_ids) == 0:
    raise RuntimeError("No valid trails found for animation.")

mask_top = np.isin(data["pair_id"], top_ids)
data = data[mask_top]


# -------------------------- Normalize Z --------------------------
Z = data["Z"]
valid_Z = Z[np.isfinite(Z)]
if valid_Z.size < 2:
    raise ValueError("Z column contains insufficient valid data.")

z_min, z_max = np.percentile(valid_Z, (2, 98))
Z_norm = np.clip((Z - z_min) / (z_max - z_min), 0, 1)

print(f"[INFO] Z normalization range: {z_min:.2f} to {z_max:.2f}")


# -------------------------- Prepare grouped trails --------------------------
trails = {}
for pid in top_ids:
    subset = data[data["pair_id"] == pid]
    trails[pid] = {
        "x2": subset["x2"],
        "y2": subset["y2"],
        "Z": subset["Z"],
    }


# -------------------------- Setup plot --------------------------
fig = plt.figure(figsize=(12, 12))

# Ensure figure size is integer pixels to avoid float->int deprecation warnings
fig.canvas.manager.set_window_title("Mirror3D Movie")
fig.set_size_inches(round(fig.get_size_inches()[0]), round(fig.get_size_inches()[1]))


ax = fig.add_subplot(111, projection="3d")
ax.set_facecolor("black")
fig.patch.set_facecolor("black")
ax.grid(False)
ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

# # --- Zoom / scale so cube fills the view ---
# # Compute equal axis limits and tighten them
# x_range = np.ptp(data["x2"])
# y_range = np.ptp(data["y2"])
# z_range = np.ptp(data["Z"])
# max_range = max(x_range, y_range, z_range)

# # Find midpoints of data to center the cube
# x_mid = (np.max(data["x2"]) + np.min(data["x2"])) / 2
# y_mid = (np.max(data["y2"]) + np.min(data["y2"])) / 2
# z_mid = (np.max(data["Z"]) + np.min(data["Z"])) / 2

# # Zoom factor: 1.0 = fit, <1 zooms out, >1 zooms in (e.g. 1.3 makes cube larger)
# zoom = 1.3

# ax.set_xlim3d([x_mid - (max_range/2)/zoom, x_mid + (max_range/2)/zoom])
# ax.set_ylim3d([y_mid - (max_range/2)/zoom, y_mid + (max_range/2)/zoom])
# ax.set_zlim3d([z_mid - (max_range/2)/zoom, z_mid + (max_range/2)/zoom])

# Color map for trails
cmap = plt.get_cmap("turbo", len(top_ids))
colors = {pid: cmap(i / len(top_ids)) for i, pid in enumerate(top_ids)}

# Plot each trail
scatters = {}
for i, pid in enumerate(top_ids):
    t = trails[pid]
    color = colors[pid]
    #scat = ax.plot(t["x2"], t["y2"], t["Z"], lw=1.5, color=color, label=f"ID {pid}")[0]
    x_stretch = 1   # >1 makes cube wider, <1 narrower
    scat = ax.plot(t["x2"] * x_stretch, t["y2"], t["Z"],lw=2, color=color, label=f"ID {pid}")[0]


    scatters[pid] = scat

# -------------------------- Add XY grid floor --------------------------
def add_xy_grid(ax, zlevel=None, spacing=100, color="gray", alpha=0.15):
    """Draws a translucent XY grid at given Z level."""
    if zlevel is None:
        zlevel = np.min(Z)
    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    xs = np.arange(xlim[0], xlim[1], spacing)
    ys = np.arange(ylim[0], ylim[1], spacing)

    for x in xs:
        ax.plot([x, x], [ylim[0], ylim[1]], [zlevel, zlevel], color=color, alpha=alpha, lw=0.8)
    for y in ys:
        ax.plot([xlim[0], xlim[1]], [y, y], [zlevel, zlevel], color=color, alpha=alpha, lw=0.8)

add_xy_grid(ax, zlevel=np.min(Z), spacing=(np.ptp(data["x2"]) / 10))

# -------------------------- Static legend --------------------------
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.08),
    ncol=min(len(top_ids), 6),
    fontsize=12,
    frameon=False,
    labelcolor="white"
)

ax.view_init(elev=25, azim=35)
angle = 35

# -------------------------- Animation function --------------------------
# Store original get_proj before overriding
if not hasattr(ax, "_orig_get_proj"):
    ax._orig_get_proj = ax.get_proj  # save the original method

def update(frame_idx):
    global angle
    if rotate_cube:
        angle += rotate_speed
        ax.view_init(elev=25, azim=angle)

    # --- Apply zoom per frame without accumulation or recursion ---
    zoom_factor = 1.2  # >1 zooms in, <1 zooms out

    def get_proj_zoom():
        proj = ax._orig_get_proj()  # use the saved original
        proj_zoom = proj.copy()
        proj_zoom[:3, :3] *= zoom_factor
        return proj_zoom

    ax.get_proj = get_proj_zoom  # safe override

    return scatters.values()


# -------------------------- Optional interactive rotation --------------------------
def on_key(event):
    global rotate_cube
    if event.key.lower() == "r":
        rotate_cube = not rotate_cube
        print(f"[INFO] Rotation toggled: {rotate_cube}")

fig.canvas.mpl_connect("key_press_event", on_key)


# -------------------------- Create animation --------------------------
writer = FFMpegWriter(fps=fps, bitrate=1800)
print("[INFO] Generating animation... (this may take a while)")

full_rotation_degrees = 360
frames_per_rotation = int(full_rotation_degrees / rotate_speed)

ani = FuncAnimation(fig, update, frames=frames_per_rotation, interval=1000/fps)

#ani = FuncAnimation(fig, update, frames=360, interval=1000 / fps, blit=False)
ani.save(output_mp4, writer=writer)
print(f"[INFO] Movie saved: {output_mp4}")

plt.close(fig)

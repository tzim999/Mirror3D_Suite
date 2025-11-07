"""
=============================================================
mirror3d_tracker.py  –  Tracker module (V14+ config compatible)
-------------------------------------------------------------
Tracks mirrored object pairs in video to compute 3D positions 
through optical geometry and motion analysis.

The program provides a live Tkinter interface for tuning threshold, 
area, and velocity settings.

It exports raw tracking data to a CSV file for later cleaning.

It loads parameters from mirror3d_config.ini using mirror3d_config.py.

MIT License
Copyright (c) 2025 Thomas Zimmerman
See https://opensource.org/licenses/MIT for full license text.
=============================================================
"""


import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from mirror3d_config import get_tracker_params
import math
import csv

# -------------------------- Load configuration --------------------------
cfg = get_tracker_params()

# Always resolve paths relative to this script’s directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Core parameters
params = {
    "THRESH":   int(cfg.get("thresh", 235)),
    "MIN_AREA": int(cfg.get("min_area", 0)),
    "MAX_AREA": int(cfg.get("max_area", 490)),
    "MAX_VEL":  int(cfg.get("max_vel", 70)),
    "MAX_LOS":  int(cfg.get("max_los", 10)),
    "FG":       int(cfg.get("fg", 25)),
    "FPS":      int(cfg.get("fps", 15)),
    "MAX_DIST": int(cfg.get("max_dist", 80)),
    "ANGLE":    int(cfg.get("angle", 20)),
}

# Optical geometry constants
W = float(cfg.get("w", 1.0))
opticalx = int(cfg.get("opticalx", 683))
opticaly = int(cfg.get("opticaly", 501))
optical_center = (opticalx, opticaly)

# Legacy uppercase compatibility (for old V13 code)
opticalX = opticalx
opticalY = opticaly

# File paths (resolved safely)
video_path  = os.path.join(BASE_DIR, cfg.get("video_path", "videos/mirror3d_video.mp4"))
output_csv  = os.path.join(BASE_DIR, cfg.get("output_csv", "output/mirror3d_track_raw.csv"))

# Overlay selection
overlay_mode = str(cfg.get("overlay", "video")).lower()

print(f"[INFO] Loaded tracker parameters from config (lowercase mode).")
print(f"[INFO] Video path: {video_path}")
print(f"[INFO] Output CSV: {output_csv}")
print(f"[INFO] Params: {params}  OC=({opticalX},{opticalY})  W={W}  Overlay={overlay_mode}")

# -------------------------- CSV schema --------------------------
csv_header = ["frame", "pair_id", "x1", "y1", "x2", "y2", "Z", "areaA", "areaB", "missed"]
csv_rows = []

# -------------------------- Helpers --------------------------
def angle_between(v1, v2):
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    mag1, mag2 = math.hypot(*v1), math.hypot(*v2)
    if mag1 == 0 or mag2 == 0: return 180.0
    cosang = abs(dot / (mag1 * mag2))
    cosang = max(min(cosang, 1.0), -1.0)
    return math.degrees(math.acos(cosang))

def find_pairs(detections, max_dist, max_angle, ocx, ocy):
    pairs, used = [], set()
    for i, (x1, y1) in enumerate(detections):
        if i in used: continue
        best_j, best_score = None, max_angle
        for j, (x2, y2) in enumerate(detections):
            if i == j or j in used: continue
            d = math.hypot(x1 - x2, y1 - y2)
            if d > max_dist: continue
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ang = angle_between((x2 - x1, y2 - y1), (ocx - mx, ocy - my))
            if ang <= best_score:
                best_score = ang; best_j = j
        if best_j is not None:
            pairs.append((i, best_j)); used.update((i, best_j))
    return pairs

def ray_intersect_rect(ax, ay, vx, vy, w, h):
    eps = 1e-9; ts = []
    if abs(vx) > eps:
        for x in [0, w-1]:
            t = (x - ax) / vx; y = ay + t * vy
            if t >= 0 and 0 <= y <= h-1: ts.append((t, x, y))
    if abs(vy) > eps:
        for y in [0, h-1]:
            t = (y - ay) / vy; x = ax + t * vx
            if t >= 0 and 0 <= x <= w-1: ts.append((t, x, y))
    if not ts: return int(ax), int(ay)
    _, x_hit, y_hit = min(ts, key=lambda x: x[0])
    return int(round(x_hit)), int(round(y_hit))

def extend_outer_through_inner_to_edge(x_outer, y_outer, x_inner, y_inner, w, h):
    vx, vy = x_inner - x_outer, y_inner - y_outer
    if vx == 0 and vy == 0:
        return x_outer, y_outer, x_inner, y_inner, x_inner, y_inner
    xe, ye = ray_intersect_rect(x_inner, y_inner, vx, vy, w, h)
    return x_outer, y_outer, x_inner, y_inner, xe, ye

def update_pair_tracks(pair_tracks, pairs, detections, areas,
                       opticalX, opticalY, W, max_vel, max_los, next_id):
    assigned = set(); pair_positions = []
    for (i, j) in pairs:
        x1a, y1a = detections[i]; x2a, y2a = detections[j]
        areaA, areaB = areas[i], areas[j]
        d1 = math.hypot(x1a - opticalX, y1a - opticalY)
        d2 = math.hypot(x2a - opticalX, y2a - opticalY)
        if d1 < d2:
            xi, yi, ai, xo, yo, ao, A, B = x1a, y1a, areaA, x2a, y2a, areaB, d1, d2
        else:
            xi, yi, ai, xo, yo, ao, A, B = x2a, y2a, areaB, x1a, y1a, areaA, d2, d1
        Z = W*(B - A)/(B + A) if (B + A) != 0 else 0.0
        cx, cy = (xi + xo)/2, (yi + yo)/2
        pair_positions.append((cx, cy, xi, yi, ai, xo, yo, ao, Z))
    new_tracks = {}
    for pid, (x1, y1, a1, x2, y2, a2, Z, missed) in list(pair_tracks.items()):
        px, py = (x1 + x2)/2, (y1 + y2)/2
        best_d, best_idx = 1e9, None
        for idx, (cx, cy, *_) in enumerate(pair_positions):
            if idx in assigned: continue
            d = math.hypot(cx - px, cy - py)
            if d < best_d and d <= max_vel:
                best_d, best_idx = d, idx
        if best_idx is not None:
            cx, cy, xi, yi, ai, xo, yo, ao, Z = pair_positions[best_idx]
            new_tracks[pid] = (xi, yi, ai, xo, yo, ao, Z, 0)
            assigned.add(best_idx)
        else:
            missed += 1
            if missed <= max_los:
                new_tracks[pid] = (x1, y1, a1, x2, y2, a2, Z, missed)
    for idx, (cx, cy, xi, yi, ai, xo, yo, ao, Z) in enumerate(pair_positions):
        if idx not in assigned:
            new_tracks[next_id] = (xi, yi, ai, xo, yo, ao, Z, 0)
            next_id += 1
    return new_tracks, next_id

# -------------------------- GUI --------------------------
root = tk.Tk()
root.title("3D Mirror Tracker 14")
root.geometry("1200x800+100+50")
root.rowconfigure(0, weight=1)
root.columnconfigure(0, weight=1)

video_label = ttk.Label(root)
video_label.grid(row=0, column=0, sticky="nsew")
controls = ttk.Frame(root); controls.grid(row=1, column=0, sticky="ew", pady=8)

style = ttk.Style()
style.configure("TButton", padding=6)
style.configure("Selected.TButton", padding=6, background="#88f")

running = True
freeze_mode = False
selected_param = None
param_buttons = {}
detections_last = []; areas_last = []
pair_tracks = {}
next_id = 1
frame_job = None
current_frame_number = 0
frames_processed = 0

# define these BEFORE creating buttons
def select_param(name):
    global selected_param
    selected_param = name
    for n, b in param_buttons.items():
        b.config(style="Selected.TButton" if n == name else "TButton")

def adjust_param(step):
    if not selected_param: return
    name = selected_param
    val = params[name] + step
    limits = {"THRESH":(0,255),"MIN_AREA":(0,2000),"MAX_AREA":(0,2000),
              "MAX_VEL":(1,200),"MAX_LOS":(1,100),"FG":(5,80),
              "FPS":(0,30),"MAX_DIST":(1,300),"ANGLE":(1,90)}
    lo, hi = limits[name]
    params[name] = max(lo, min(hi, val))
    param_buttons[name].config(text=f"{name} {params[name]}")

def toggle_optical_center():
    global freeze_mode
    freeze_mode = not freeze_mode
    if freeze_mode:
        print("Optical Center mode: frozen — click in video to set new center.")
    else:
        print("Optical Center mode: resumed.")

vars_frame = ttk.Frame(controls); vars_frame.pack(side="top", pady=4)
frame_label = ttk.Label(vars_frame, text="Frame 0", width=10); frame_label.pack(side="left", padx=4, pady=4)

# Build buttons
for name in ["THRESH","MIN_AREA","MAX_AREA","MAX_VEL","MAX_LOS","FG","FPS","MAX_DIST","ANGLE"]:
    b = ttk.Button(vars_frame, text=f"{name} {params[name]}", width=14, command=lambda n=name: select_param(n))
    b.pack(side="left", padx=3, pady=4); param_buttons[name] = b

ttk.Button(vars_frame, text="Optical Center", width=14, command=toggle_optical_center)\
   .pack(side="left", padx=6, pady=4)
   
   
# --- Toggle overlay mode button ---
def toggle_overlay_mode():
    global OVERLAY_VIDEO
    OVERLAY_VIDEO = not OVERLAY_VIDEO
    state = "Video" if OVERLAY_VIDEO else "Binary"
    print(f"[INFO] Overlay switched to {state} mode.")
    overlay_btn.config(text=f"Overlay: {state}")

OVERLAY_VIDEO = True  # or False, depending on your default mode
overlay_btn = ttk.Button(vars_frame, text="Overlay: Video", width=14, command=toggle_overlay_mode)
overlay_btn.pack(side="left", padx=6, pady=4)
overlay_btn.config(text=f"Overlay: {'Video' if OVERLAY_VIDEO else 'Binary'}") 
   
   

steps_frame = ttk.Frame(controls); steps_frame.pack(side="top", pady=8)
for s in [-100,-10,-1,+1,+10,+100]:
    ttk.Button(steps_frame, text=f"{s:+d}", width=6, command=lambda v=s: adjust_param(v)).pack(side="left", padx=3)

def on_click(event):
    global freeze_mode, opticalX, opticalY, optical_center
    if not freeze_mode or cap is None: return
    w_disp, h_disp = video_label.winfo_width(), video_label.winfo_height()
    fw, fh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fw == 0 or fh == 0 or w_disp == 0 or h_disp == 0: return
    opticalX = int(event.x * fw / w_disp)
    opticalY = int(event.y * fh / h_disp)
    optical_center = (opticalX, opticalY)
    print(f"New optical center: ({opticalX}, {opticalY})")
    freeze_mode = False  # auto-unfreeze after selection

video_label.bind("<Button-1>", on_click)

# -------------------------- Video & BG model --------------------------

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"[WARNING] Cannot open video at:\n  {video_path}")
    cap = None
    total_frames = 0
else:
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Video loaded successfully:\n  {video_path}")
    print(f"[INFO] Total frames: {total_frames}")

fgbg = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=params["FG"],
    detectShadows=False
)


# -------------------------- Main loop --------------------------
OVERLAY_VIDEO = (overlay_mode == "video") # True = show original video, False = show binary mask

def process_frame():
    global frame_job, detections_last, areas_last, pair_tracks, next_id, current_frame_number, frames_processed
    if not running or cap is None:
        return

    # Optical center freeze
    if freeze_mode:
        pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        ok, frame = cap.read()
        if not ok:
            save_and_exit(complete=True); return
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        h, w = frame.shape[:2]
        overlay = frame.copy()
        pairs = find_pairs(detections_last, params["MAX_DIST"], params["ANGLE"], opticalX, opticalY)
        for (i, j) in pairs:
            (x1, y1), (x2, y2) = detections_last[i], detections_last[j]
            d1 = math.hypot(x1 - opticalX, y1 - opticalY)
            d2 = math.hypot(x2 - opticalX, y2 - opticalY)
            xo, yo, xi, yi = (x1, y1, x2, y2) if d1 > d2 else (x2, y2, x1, y1)
            x_outer, y_outer, x_inner, y_inner, xe, ye = extend_outer_through_inner_to_edge(xo, yo, xi, yi, w, h)
            cv2.line(overlay, (x_outer, y_outer), (xe, ye), (0, 255, 255), 2)
        cv2.circle(overlay, (int(optical_center[0]), int(optical_center[1])), 8, (255, 0, 0), 2)
        wd, hd = max(video_label.winfo_width(),1), max(video_label.winfo_height(),1)
        disp_img = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)).resize((wd, hd), Image.Resampling.LANCZOS)
        imgtk = ImageTk.PhotoImage(disp_img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
        frame_job = root.after(80, process_frame)
        return

    # Regular playback
    ok, frame = cap.read()
    if not ok:
        save_and_exit(complete=True); return

    current_frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    frame_label.config(text=f"Frame {current_frame_number}")
    frames_processed = max(frames_processed, current_frame_number)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fgbg.setVarThreshold(params["FG"])
    fgmask = fgbg.apply(gray)
    fgmask = cv2.medianBlur(fgmask, 5)
    _, fgmask = cv2.threshold(fgmask, params["THRESH"], 255, cv2.THRESH_BINARY)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), 2)
    cnts, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections_last, areas_last = [], []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < params["MIN_AREA"] or a > params["MAX_AREA"]: continue
        M = cv2.moments(c)
        if M["m00"] == 0: continue
        cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
        detections_last.append((cx, cy)); areas_last.append(a)

    pairs = find_pairs(detections_last, params["MAX_DIST"], params["ANGLE"], opticalX, opticalY)
    pair_tracks, next_id = update_pair_tracks(
        pair_tracks, pairs, detections_last, areas_last,
        opticalX, opticalY, W, params["MAX_VEL"], params["MAX_LOS"], next_id
    )

    if pair_tracks:
        for pid, (x1, y1, a1, x2, y2, a2, Z, missed) in pair_tracks.items():
            csv_rows.append([current_frame_number, pid, int(x1), int(y1), int(x2), int(y2),
                             round(Z, 6), round(a1, 2), round(a2, 2), missed])

    # --- Overlay choice ---
    overlay = frame.copy() if OVERLAY_VIDEO else cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)

    # Draw optical center & yellow pair lines
    cv2.circle(overlay, (int(optical_center[0]), int(optical_center[1])), 8, (255, 0, 0), 2)
    for pid, (x1, y1, a1, x2, y2, a2, Z, missed) in pair_tracks.items():
        cx, cy = int((x1 + x2)/2), int((y1 + y2)/2)
        cv2.putText(overlay, str(pid), (cx + 8, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.circle(overlay, (cx, cy), 5, (0, 0, 255), 1)
        cv2.line(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 1)

    # --- Display scaled ---
    wd, hd = max(video_label.winfo_width(),1), max(video_label.winfo_height(),1)
    disp_img = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)).resize((wd, hd), Image.Resampling.LANCZOS)
    imgtk = ImageTk.PhotoImage(disp_img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)
    
    # --- FPS control (pause with heartbeat if FPS==0) ---
    if params["FPS"] <= 0:
        # Rewind one frame so we re-display the same frame next tick
        pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, pos-1))
        # Keep a slow UI heartbeat so optical-center clicks & UI still work
        frame_job = root.after(80, process_frame)
        return
    
    delay = int(1000 / params["FPS"])
    frame_job = root.after(delay, process_frame)

# -------------------------- Exit & saving --------------------------
def print_params():
    print("[INFO] Parameters at exit:")
    for k in ["THRESH","MIN_AREA","MAX_AREA","MAX_VEL","MAX_LOS","FG","FPS","MAX_DIST","ANGLE"]:
        print(f"  {k} {params[k]}")
    print(f"  opticalX {opticalX}")
    print(f"  opticalY {opticalY}")

def save_and_exit(complete=False):
    """Save all tracked data to CSV and exit cleanly."""
    print("\n[INFO] Saving CSV and exiting...")

    # --- define consistent local variables ---
    videoFileName = os.path.basename(video_path)  # short name for printout
    status = "complete" if complete else "interrupted"

    # --- safe check in case frames_processed isn't yet defined ---
    #frames = globals().get("frames_processed", 0)
    frames = globals().get("frames_processed", 0) or globals().get("current_frame_number", 0) 

    try:
        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(csv_header)
            writer.writerows(csv_rows)
        print(f"[INFO] Saved {frames} of {int(total_frames)} frames from video: {videoFileName} ({status})")
        print(f"[INFO] Output CSV: {output_csv}")
    except Exception as e:
        print(f"[ERROR] Could not save CSV: {e}")

    # --- print current parameter summary ---
    print("=" * 20)
    print("[INFO] Parameters at exit:")
    for k, v in params.items():
        print(f"  {k} {v}")
    print(f"  opticalX {opticalX}")
    print(f"  opticalY {opticalY}")
    print("=" * 20)

    # --- clean shutdown of Tkinter UI ---
    try:
        cap.release()
        root.destroy()
    except Exception as e:
        print(f"[WARN] Could not destroy Tkinter root cleanly: {e}")



def on_close():
    save_and_exit(complete=False)

root.protocol("WM_DELETE_WINDOW", on_close)

# -------------------------- Kickoff --------------------------
if cap is not None:
    root.after(0, process_frame)
root.mainloop()

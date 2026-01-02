import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("analysis/RMSE")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DURATION = 10.0
FPS = 30
HEIGHT = 10.0
TOTAL_FRAMES = int(DURATION * FPS)

GEOMETRIES = ["2_mics", "rect_4", "rect_16", "rect_64", "ring_32", "spiral_64", "sunflower_64"]
TRAJECTORIES = ["linear", "diagonal", "circle"]

def get_true_position(traj_type, frame_idx):
    t = frame_idx / FPS
    if t > DURATION: t = DURATION

    x, y, z = 0.0, 0.0, HEIGHT

    if traj_type == "linear":
        start_pos = -9.0
        end_pos = 9.0
        velocity = (end_pos - start_pos) / DURATION
        x = start_pos + velocity * t
        y = 0.0

    elif traj_type == "diagonal":
        start_x, start_y = -8.0, -8.0
        end_x, end_y = 8.0, 8.0
        progress = t / DURATION
        x = start_x + (end_x - start_x) * progress
        y = start_y + (end_y - start_y) * progress

    elif traj_type == "circle":
        radius = 8.0
        omega = 2 * np.pi / DURATION
        angle = omega * t + np.pi
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)

    return np.array([x, y, z])

stats_list = []

for geo in GEOMETRIES:
    for traj in TRAJECTORIES:

        folder_name = f"{geo}_{traj}"
        file_name = f"{geo}_{traj}_g10_focuspoints.npy"

        file_path = RESULTS_DIR / geo / folder_name / "g10" / "data" / file_name

        if not file_path.exists():
            print(f"[SKIP] Nie znaleziono pliku: {file_path}")
            continue

        print(f"Przetwarzanie: {geo} - {traj}...")

        try:
            est_points = np.load(file_path)
        except Exception as e:
            print(f"Błąd odczytu {file_name}: {e}")
            continue

        n_frames = min(TOTAL_FRAMES, len(est_points))
        est_points = est_points[:n_frames]

        true_points = np.array([get_true_position(traj, i) for i in range(n_frames)])

        jumps = np.linalg.norm(np.diff(est_points, axis=0, prepend=est_points[0:1]), axis=1)

        valid_mask = (jumps < 2.0) & (np.abs(est_points[:, 0]) < 20.0) & (np.abs(est_points[:, 1]) < 20.0)

        errors = []
        for i in range(n_frames):
            if valid_mask[i]:
                dist = np.linalg.norm(est_points[i, :2] - true_points[i, :2])
                errors.append(dist)
            else:
                errors.append(np.nan)

        errors = np.array(errors)
        valid_errors = errors[~np.isnan(errors)]

        if len(valid_errors) > 0:
            rmse = np.sqrt(np.mean(valid_errors ** 2))
            mean_err = np.mean(valid_errors)
            median_err = np.median(valid_errors)
            std_err = np.std(valid_errors)
            valid_percent = (len(valid_errors) / n_frames) * 100
        else:
            rmse, mean_err, median_err, std_err, valid_percent = np.nan, np.nan, np.nan, np.nan, 0

        # Zapis do listy
        stats_list.append({
            "Geometry": geo,
            "Trajectory": traj,
            "RMSE": rmse,
            "Median Error": median_err,
            "Std Dev": std_err,
            "Valid Points %": valid_percent
        })

        # WYKRES 1: Trajektoria
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(true_points[:, 0], true_points[:, 1], 'k--', linewidth=2, label='Trajektoria')
        ax.scatter(est_points[valid_mask, 0], est_points[valid_mask, 1], c='blue', s=10, alpha=0.6, label='FBF')
        ax.scatter(est_points[~valid_mask, 0], est_points[~valid_mask, 1], c='red', marker='x', s=20, label='Outliery')

        ax.set_title(f"Trajektoria: {geo} - {traj}\nRMSE: {rmse:.3f}m")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.axis('equal')
        ax.legend()
        ax.grid(True)
        plt.savefig(OUTPUT_DIR / f"traj_{geo}_{traj}.png")
        plt.close()

        # WYKRES 2: Błąd w czasie
        fig, ax = plt.subplots(figsize=(10, 4))
        time_axis = np.arange(n_frames) / FPS
        ax.stem(time_axis, errors, markerfmt='.', basefmt=" ", linefmt='C0-')
        ax.set_xlabel("Czas [s]")
        ax.set_ylabel("Błąd lokalizacji [m]")
        ax.set_title(f"Błąd w czasie: {geo} - {traj}")
        ax.grid(True, alpha=0.3)
        plt.savefig(OUTPUT_DIR / f"error_{geo}_{traj}.png")
        plt.close()

df = pd.DataFrame(stats_list)

custom_order = [
    "2_mics",
    "rect_4",
    "rect_16",
    "rect_64",
    "ring_32",
    "spiral_64",
    "sunflower_64"
]

df['Geometry'] = pd.Categorical(df['Geometry'], categories=custom_order, ordered=True)

df = df.sort_values(by=["Geometry", "Trajectory"])

formatters = {
    "RMSE": "{:.4f}".format,
    "Median Error": "{:.4f}".format,
    "Std Dev": "{:.4f}".format,
    "Valid Points %": "{:.1f}%".format
}

latex_code = df.to_latex(
    index=False,
    formatters=formatters,
    caption="Porównanie błędów RMSE dla badanych geometrii",
    label="tab:rmse_results",
    position="h",
    column_format="lccccc"
)

latex_file = OUTPUT_DIR / "tabela_rmse.tex"
with open(latex_file, "w") as f:
    f.write(latex_code)

# WYKRES RMSE
if not df.empty:
    plt.figure(figsize=(12, 6))
    pivot_df = df.pivot(index="Geometry", columns="Trajectory", values="RMSE")

    pivot_df.plot(kind='bar', figsize=(12, 6), width=0.8)
    plt.title("Porównanie RMSE dla różnych geometrii")
    plt.ylabel("RMSE [m]")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "comparison_rmse_bar.png")
    plt.show()

print(f"\nAnaliza zakończona. Wyniki zapisano w folderze: {OUTPUT_DIR.resolve()}")
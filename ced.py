import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path

RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("analysis/CED")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DURATION = 10.0
FPS = 30
HEIGHT = 10.0
TOTAL_FRAMES = int(DURATION * FPS)

GEOMETRIES = [
    "2_mics",
    "rect_4",
    "rect_16",
    "rect_64",
    "ring_32",
    "spiral_64",
    "sunflower_64"
]

TRAJECTORIES = ["linear", "diagonal", "circle"]

def get_true_position(traj_type, frame_idx):
    t = frame_idx / FPS
    if t > DURATION: t = DURATION
    x, y, z = 0.0, 0.0, HEIGHT

    if traj_type == "linear":
        start_pos, end_pos = -9.0, 9.0
        velocity = (end_pos - start_pos) / DURATION
        x = start_pos + velocity * t
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

for traj in TRAJECTORIES:
    print(f"Generowanie wykresu CED dla trajektorii: {traj}...")

    fig, ax = plt.subplots(figsize=(10, 6))

    for geo in GEOMETRIES:
        file_path = RESULTS_DIR / geo / f"{geo}_{traj}" / "g10" / "data" / f"{geo}_{traj}_g10_focuspoints.npy"

        if not file_path.exists():
            print(f"  [SKIP] Brak pliku dla {geo}")
            continue

        try:
            est_points = np.load(file_path)
        except:
            continue

        n_frames = min(TOTAL_FRAMES, len(est_points))
        est_points = est_points[:n_frames]
        true_points = np.array([get_true_position(traj, i) for i in range(n_frames)])

        jumps = np.linalg.norm(np.diff(est_points, axis=0, prepend=est_points[0:1]), axis=1)
        valid_mask = (jumps < 2.0) & (np.abs(est_points[:, 0]) < 20.0)

        errors = []
        for i in range(n_frames):
            if valid_mask[i]:
                dist = np.linalg.norm(est_points[i, :2] - true_points[i, :2])
                errors.append(dist)

        errors = np.array(errors)

        if len(errors) == 0:
            continue

        sorted_errors = np.sort(errors)
        yvals = np.arange(len(sorted_errors)) / float(len(sorted_errors))

        ax.plot(sorted_errors, yvals, label=geo, linewidth=2)

    ax.set_title(f"Dystrybuanta błędu (CED) - Trajektoria {traj.capitalize()}")
    ax.set_xlabel("Błąd lokalizacji [m]")
    ax.set_ylabel("Prawdopodobieństwo (Skumulowane)")

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))

    ax.set_xlim(0, 0.5)

    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(title="Geometria")

    save_path = OUTPUT_DIR / f"ced_{traj}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"  Zapisano: {save_path}")

print("\nGotowe! Sprawdź folder analysis_output/ced")
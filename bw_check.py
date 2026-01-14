import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("analysis/GAMMA_CHECK")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

geometries = ["rect_64", "sunflower_64"]
gammas_order = [-1, 1, 4, 10, 50, 300]
gamma_labels = ["MVDR (g=-1)", "DAS (g=1)", "g=4", "g=10", "g=50", "g=300"]
trajectory = "circle"

stats = {geo: {'means': [], 'stds': []} for geo in geometries}

print("Generowanie wykresu szerokości wiązki...")

for geom in geometries:
    for g in gammas_order:
        npy_path = RESULTS_DIR / geom / f"{geom}_{trajectory}" / f"g{g}" / "data" / f"{geom}_{trajectory}_widths.npy"

        if npy_path.exists():
            widths = np.load(npy_path)
            avg_frame_width = np.mean(widths, axis=1)
            valid_widths = avg_frame_width[avg_frame_width > 0.01]

            if len(valid_widths) > 0:
                mean_val = np.mean(valid_widths)
                std_val = np.std(valid_widths)
            else:
                mean_val = 0
                std_val = 0
        else:
            mean_val = 0
            std_val = 0

        stats[geom]['means'].append(mean_val)
        stats[geom]['stds'].append(std_val)

plt.figure(figsize=(12, 7))

x = np.arange(len(gammas_order))

styles = {'rect_64': 's--', 'sunflower_64': 'o-'}
colors = {'rect_64': 'blue', 'sunflower_64': 'orange'}

for geom in geometries:
    means = np.array(stats[geom]['means'])
    stds = np.array(stats[geom]['stds'])

    mask = means > 0

    plt.plot(x[mask], means[mask], styles[geom], label=geom, color=colors[geom], linewidth=2, markersize=8)
    plt.fill_between(x[mask], means[mask] - stds[mask], means[mask] + stds[mask], color=colors[geom], alpha=0.15)

    y_offset = 15 if geom == "sunflower_64" else -20

    for xi, yi in zip(x[mask], means[mask]):
        plt.annotate(
            f"{yi:.3f} m",
            (xi, yi),
            textcoords="offset points",
            xytext=(0, y_offset),
            ha='center',
            color=colors[geom],
            fontsize=9,
            fontweight='bold'
        )

plt.xticks(x, gamma_labels)
plt.xlabel("Algorytm / Wartość Gamma")
plt.ylabel("Szerokość wiązki głównej (-3dB) [m]")
plt.title("Wpływ parametru Gamma na rozdzielczość przestrzenną")
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()

save_path = OUTPUT_DIR / "beamwidth_comparison_labels.png"
plt.savefig(save_path, dpi=150)
print(f"Zapisano wykres: {save_path}")
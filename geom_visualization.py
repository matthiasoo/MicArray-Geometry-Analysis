import acoular as ac
import matplotlib.pyplot as plt
from pathlib import Path

geom_folder = Path('geom')
output_folder = Path('geom/visuals')
output_folder.mkdir(parents=True, exist_ok=True)

files = [
    "rect_4.xml",
    "rect_16.xml",
    "rect_64.xml",
    "ring_32.xml",
    "spiral_64.xml",
    "sunflower_64.xml",
]

for filename in files:
    fpath = geom_folder / filename

    if not fpath.exists():
        print(f"No such file: {filename}. Skipping.")
        continue

    try:
        mg = ac.MicGeom(from_file=fpath)

        plt.figure(figsize=(8, 6))

        plt.scatter(mg.mpos[0], mg.mpos[1], s=50, c='tab:blue', alpha=0.7, edgecolors='k')

        plt.scatter(mg.mpos[0][0], mg.mpos[1][0], s=80, c='red', edgecolors='k', label='Mic 1 (Start)')

        offset = 0.02 * max(mg.mpos[0].max() - mg.mpos[0].min(), 0.1)

        for i in range(mg.num_mics):
            plt.text(mg.mpos[0][i] + offset / 2, mg.mpos[1][i] + offset / 2,
                     str(i + 1),
                     fontsize=9, color='darkslategray')

        plt.title(f"Geometry: {filename}\nTotal Mics: {mg.num_mics}", fontsize=14, fontweight='bold')
        plt.xlabel("X Position [m]")
        plt.ylabel("Y Position [m]")
        plt.axis('equal')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()

        output_filename = output_folder / f"{fpath.stem}.png"
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"-> Saved: {output_filename.name}")

    except Exception as e:
        print(f"Error with file {filename}: {e}")

print("Done")
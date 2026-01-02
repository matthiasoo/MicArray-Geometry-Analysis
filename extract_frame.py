import cv2
import os
from pathlib import Path

RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("analysis", "SNAPSHOTS_5s")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_TIME = 5.0
FPS = 30

GEOMETRIES = [
    "2_mics", "rect_4", "rect_16", "rect_64",
    "ring_32", "spiral_64", "sunflower_64"
]
TRAJECTORIES = ["linear", "diagonal", "circle"]

FRAME_NO = int(TARGET_TIME * FPS)

print(f"--- EKSTRAKCJA KLATEK (Czas: {TARGET_TIME}s, Klatka: {FRAME_NO}) ---")

count = 0
for geo in GEOMETRIES:
    for traj in TRAJECTORIES:

        folder_name = f"{geo}_{traj}"
        video_name = f"{geo}_{traj}_g10.mp4"
        video_path = RESULTS_DIR / geo / folder_name / "g10" / "maps" / video_name

        if not video_path.exists():
            print(f"[SKIP] Nie znaleziono: {video_name}")
            continue

        cap = cv2.VideoCapture(str(video_path))

        cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_NO)

        success, frame = cap.read()

        if success:
            output_filename = f"{geo}_{traj}_t{TARGET_TIME}.png"
            output_path = OUTPUT_DIR / output_filename

            cv2.imwrite(str(output_path), frame)
            print(f"[OK] Zapisano: {output_filename}")
            count += 1
        else:
            print(f"[ERROR] Nie udało się odczytać klatki z {video_name}")

        cap.release()

print(f"\nZakończono! Wyciągnięto {count} obrazów.")
print(f"Pliki znajdują się w folderze: {OUTPUT_DIR.resolve()}")
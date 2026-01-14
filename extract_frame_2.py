import cv2
import os
from pathlib import Path

RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("analysis", "SNAPSHOTS_gamma")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_TIME = 5.0
FPS = 30
FRAME_NO = int(TARGET_TIME * FPS)

GEOMETRIES = ["rect_64", "sunflower_64"]
TRAJECTORIES = ["circle"]
GAMMAS = [-1, 1, 4, 10, 50, 300]

print(f"--- EKSTRAKCJA KLATEK DO ANALIZY GAMMA ---")
print(f"Czas: {TARGET_TIME}s (Klatka #{FRAME_NO})")
print(f"Geometrie: {GEOMETRIES}")
print(f"Gammy: {GAMMAS}")
print("-" * 50)

count = 0
for geo in GEOMETRIES:
    for traj in TRAJECTORIES:
        for g in GAMMAS:

            folder_name = f"{geo}_{traj}"
            gamma_folder = f"g{g}"
            video_name = f"{geo}_{traj}_g{g}.mp4"

            video_path = RESULTS_DIR / geo / folder_name / gamma_folder / "maps" / video_name

            if not video_path.exists():
                print(f"[SKIP] Nie znaleziono pliku: {video_path}")
                continue

            cap = cv2.VideoCapture(str(video_path))

            cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_NO)
            success, frame = cap.read()

            if success:
                output_filename = f"{geo}_{traj}_g{g}.png"
                output_path = OUTPUT_DIR / output_filename

                cv2.imwrite(str(output_path), frame)
                print(f"[OK] Zapisano: {output_filename}")
                count += 1
            else:
                print(f"[ERROR] Nie udało się odczytać klatki z {video_name}")

            cap.release()

print("-" * 50)
print(f"Zakończono! Wyciągnięto {count} obrazów.")
print(f"Pliki gotowe do raportu znajdują się w: {OUTPUT_DIR.resolve()}")
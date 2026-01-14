import xml.etree.ElementTree as ET
from pathlib import Path

INPUT_DIR = Path("geom_3d")
OUTPUT_DIR = Path("geom_3d_export")
OUTPUT_DIR.mkdir(exist_ok=True)


def convert_xml_to_ply(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    mics = []
    for pos in root.findall("pos"):
        x = float(pos.get("x"))
        y = float(pos.get("y"))
        z = float(pos.get("z"))
        mics.append((x, y, z))

    n_verts = len(mics)
    ply_path = OUTPUT_DIR / f"{xml_path.stem}.ply"

    header = f"""ply
format ascii 1.0
comment Exported from Acoular XML
element vertex {n_verts}
property float x
property float y
property float z
end_header
"""

    with open(ply_path, "w") as f:
        f.write(header)
        for x, y, z in mics:
            f.write(f"{x} {y} {z}\n")

    print(f"[OK] Wyeksportowano: {ply_path} ({n_verts} punktów)")


if __name__ == "__main__":
    xml_files = list(INPUT_DIR.glob("*.xml"))

    if not xml_files:
        print(f"Nie znaleziono plików XML w folderze {INPUT_DIR}")
    else:
        print(f"Znaleziono {len(xml_files)} plików. Konwertowanie...")
        for xml_file in xml_files:
            convert_xml_to_ply(xml_file)
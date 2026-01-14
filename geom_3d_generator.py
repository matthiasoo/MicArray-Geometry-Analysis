import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET

OUTPUT_DIR = Path("geom_3d")
OUTPUT_DIR.mkdir(exist_ok=True)


def save_to_xml(coords, filename):
    root = ET.Element("MicGeom")
    root.set("Name", filename.stem)

    for i in range(coords.shape[1]):
        pos = ET.SubElement(root, "pos")
        pos.set("Name", f"Mic{i + 1}")
        pos.set("x", f"{coords[0, i]:.4f}")
        pos.set("y", f"{coords[1, i]:.4f}")
        pos.set("z", f"{coords[2, i]:.4f}")

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ", level=0)
    tree.write(filename, encoding="utf-8", xml_declaration=True)
    print(f"Generated: {filename} ({coords.shape[1]} mics)")

def gen_double_layer():
    x = np.linspace(-0.5, 0.5, 4)
    y = np.linspace(-0.5, 0.5, 4)
    xx, yy = np.meshgrid(x, y)
    z1 = np.zeros_like(xx)

    z2 = np.full_like(xx, -0.5)

    l1 = np.array([xx.flatten(), yy.flatten(), z1.flatten()])
    l2 = np.array([xx.flatten(), yy.flatten(), z2.flatten()])

    coords = np.concatenate([l1, l2], axis=1)
    save_to_xml(coords, OUTPUT_DIR / "double_minidsp_32.xml")


def gen_cube(side=1.0):
    half = side / 2
    coords = []

    # wierzchołki
    for x in [-half, half]:
        for y in [-half, half]:
            for z in [-half, half]:
                coords.append([x, y, z])

    # przecięcia przekątnych bocznych
    coords.append([half, 0, 0])
    coords.append([-half, 0, 0])

    coords.append([0, half, 0])
    coords.append([0, -half, 0])

    coords.append([0, 0, half])
    coords.append([0, 0, -half])

    coords_arr = np.array(coords).T
    save_to_xml(coords_arr, OUTPUT_DIR / "cube_14.xml")


def gen_fibonacci_sphere(n_mics=64, radius=0.5):
    phi = np.pi * (3. - np.sqrt(5.))

    coords = np.zeros((3, n_mics))

    for i in range(n_mics):
        y = 1 - (i / float(n_mics - 1)) * 2
        radius_at_y = np.sqrt(1 - y * y)

        theta = phi * i

        x = np.cos(theta) * radius_at_y
        z = np.sin(theta) * radius_at_y

        coords[0, i] = x * radius
        coords[1, i] = y * radius
        coords[2, i] = z * radius

    save_to_xml(coords, OUTPUT_DIR / "sphere_fib_64.xml")


if __name__ == "__main__":
    gen_double_layer()
    gen_cube(side=1.0)
    gen_fibonacci_sphere(n_mics=64, radius=0.5)
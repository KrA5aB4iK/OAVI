import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import csv

INPUT_DIR = 'alphabet/direct'
OUTPUT_DIR = 'alphabet/features'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_binary_image(filepath):
    img = Image.open(filepath).convert('L')
    arr = np.array(img)
    return (arr == 0).astype(np.uint8)  # 1 для чёрного, 0 для белого

def compute_mass_quarters(binary):
    h, w = binary.shape
    qh, qw = h // 2, w // 2
    quarters = [
        binary[0:qh, 0:qw],
        binary[0:qh, qw:],
        binary[qh:, 0:qw],
        binary[qh:, qw:]
    ]
    masses = [np.sum(q) for q in quarters]
    areas = [q.size for q in quarters]
    mass_norm = [m / a for m, a in zip(masses, areas)]
    return masses, mass_norm

def compute_centroid(binary):
    h, w = binary.shape
    y_indices, x_indices = np.nonzero(binary)
    if len(x_indices) == 0:
        return (0, 0), (0, 0)
    cx = np.mean(x_indices)
    cy = np.mean(y_indices)
    return (cx, cy), (cx / w, cy / h)

def compute_moments(binary, cx, cy):
    y_indices, x_indices = np.nonzero(binary)
    Ix = np.sum((y_indices - cy) ** 2)
    Iy = np.sum((x_indices - cx) ** 2)
    return Ix, Iy

def compute_normalized_moments(Ix, Iy, h, w):
    total = Ix + Iy
    if total == 0:
        return 0.0, 0.0
    return Ix / total, Iy / total



def compute_profiles(binary):
    profile_x = np.sum(binary, axis=0)
    profile_y = np.sum(binary, axis=1)
    return profile_x, profile_y

def plot_profile(profile, axis, name):
    plt.figure(figsize=(6, 3))
    plt.bar(range(len(profile)), profile)
    plt.title(f"Profile {axis} - {name}")
    plt.xlabel(f"{axis}-axis")
    plt.ylabel("Black pixel count")
    plt.xticks(range(0, len(profile), max(1, len(profile)//10)))
    plt.yticks(np.arange(0, profile.max() + 1, max(1, profile.max()//10)))
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"profile_{axis}_{name}.png"))
    plt.close()

csv_path = os.path.join(OUTPUT_DIR, "features.csv")

with open(csv_path, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=';')
    header = [
        "name",
        "mass_q1", "mass_q2", "mass_q3", "mass_q4",
        "norm_mass_q1", "norm_mass_q2", "norm_mass_q3", "norm_mass_q4",
        "cx", "cy", "cx_norm", "cy_norm",
        "Ix", "Iy", "Ix_norm", "Iy_norm"
    ]
    writer.writerow(header)

    for filename in sorted(os.listdir(INPUT_DIR)):
        if not filename.lower().endswith(".png"):
            continue

        filepath = os.path.join(INPUT_DIR, filename)
        name = os.path.splitext(filename)[0]

        binary = load_binary_image(filepath)
        h, w = binary.shape

        mass, norm_mass = compute_mass_quarters(binary)
        (cx, cy), (cx_n, cy_n) = compute_centroid(binary)
        Ix, Iy = compute_moments(binary, cx, cy)
        Ix_n, Iy_n = compute_normalized_moments(Ix, Iy, h, w)  # где новая функция даёт Ix / (Ix + Iy)
        profile_x, profile_y = compute_profiles(binary)

        # Сохраняем профили
        plot_profile(profile_x, 'X', name)
        plot_profile(profile_y, 'Y', name)

        # Пишем в CSV
        row = [name] + mass + norm_mass + [cx, cy, cx_n, cy_n, Ix, Iy, Ix_n, Iy_n]
        writer.writerow(row)

print(f"✅ Готово! Файл признаков: {csv_path}\n📊 Профили: {OUTPUT_DIR}/profile_X_*.png, profile_Y_*.png")

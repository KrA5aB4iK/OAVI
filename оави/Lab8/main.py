import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def get_lightness_channel(img_rgb):
    img_hls = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HLS)
    L = img_hls[:, :, 1]
    return L

def linear_contrast(img):
    min_val = np.min(img)
    max_val = np.max(img)
    if max_val == min_val:
        return np.zeros_like(img)
    stretched = (img - min_val) / (max_val - min_val) * 255
    return np.clip(stretched, 0, 255).astype(np.uint8)

def compute_hog_features(img_gray, cell_size=8):
    gx = cv2.filter2D(img_gray.astype(np.float32), -1, np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]))
    gy = cv2.filter2D(img_gray.astype(np.float32), -1, np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]))
    mag = np.abs(gx) + np.abs(gy)

    h, w = mag.shape
    features = []
    for y in range(0, h, cell_size):
        for x in range(0, w, cell_size):
            cell = mag[y:y+cell_size, x:x+cell_size]
            features.append(np.sum(cell))
    return np.array(features)

def compute_ngtdm_matrix(img, num_levels=16):
    img_quantized = np.floor(img / (256 / num_levels)).astype(np.uint8)
    h, w = img.shape
    ngtdm = np.zeros((num_levels,), dtype=np.float32)
    counts = np.zeros((num_levels,), dtype=np.int32)

    for y in range(1, h-1):
        for x in range(1, w-1):
            g = img_quantized[y, x]
            neighborhood = img_quantized[y-1:y+2, x-1:x+2].copy()
            neighborhood[1, 1] = 0  # исключаем центр
            avg = (np.sum(neighborhood) - 0) / 8.0
            diff = abs(g * 1.0 - avg)
            ngtdm[g] += diff
            counts[g] += 1

    # Чтобы не было деления на 0
    with np.errstate(divide='ignore', invalid='ignore'):
        ngtdm[counts > 0] /= counts[counts > 0]

    return ngtdm

def process_image(path):
    filename = os.path.splitext(os.path.basename(path))[0]
    out_dir = f"{filename}_output"
    os.makedirs(out_dir, exist_ok=True)

    img = cv2.imread(path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    L = get_lightness_channel(img_rgb)
    L = np.clip(L, 0, 150).astype(np.uint8)  # искусственное сжатие контраста
    L_contrasted = linear_contrast(L)

    hog_orig = compute_hog_features(L)
    hog_contr = compute_hog_features(L_contrasted)

    ngtdm_orig = compute_ngtdm_matrix(L)
    ngtdm_contr = compute_ngtdm_matrix(L_contrasted)

    # Сохраняем изображения
    cv2.imwrite(os.path.join(out_dir, f"{filename}_L_original.png"), L)
    cv2.imwrite(os.path.join(out_dir, f"{filename}_L_contrasted.png"), L_contrasted)

    # Визуализация
    fig, axs = plt.subplots(4, 2, figsize=(12, 14))

    axs[0, 0].imshow(L, cmap='gray')
    axs[0, 0].set_title("Полутоновое изображение (до)")
    axs[0, 0].axis('off')

    axs[0, 1].imshow(L_contrasted, cmap='gray')
    axs[0, 1].set_title("Контрастированное изображение (после)")
    axs[0, 1].axis('off')

    axs[1, 0].hist(L.ravel(), bins=256, color='gray')
    axs[1, 0].set_title("Гистограмма яркости (до)")

    axs[1, 1].hist(L_contrasted.ravel(), bins=256, color='gray')
    axs[1, 1].set_title("Гистограмма яркости (после)")

    axs[2, 0].bar(np.arange(len(hog_orig)), hog_orig, color='black')
    axs[2, 0].set_title("HOG признаки - до")

    axs[2, 1].bar(np.arange(len(hog_contr)), hog_contr, color='black')
    axs[2, 1].set_title("HOG признаки - после")

    axs[3, 0].bar(np.arange(len(ngtdm_orig)), ngtdm_orig, color='green')
    axs[3, 0].set_title("NGTDM матрица - до")

    axs[3, 1].bar(np.arange(len(ngtdm_contr)), ngtdm_contr, color='green')
    axs[3, 1].set_title("NGTDM матрица - после")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{filename}_analysis.png"))
    plt.close()

    # Отчёт в консоль
    print(f"\nДо контрастирования (сумма HOG признаков): {np.sum(hog_orig):.2f}")
    print(f"После контрастирования (сумма HOG признаков): {np.sum(hog_contr):.2f}")
    print("min/max L:", np.min(L), np.max(L))
    print("min/max L_contrasted:", np.min(L_contrasted), np.max(L_contrasted))
    print("\nNGTDM матрица (до контрастирования):\n", ngtdm_orig)
    print("\nNGTDM матрица (после контрастирования):\n", ngtdm_contr)

# Запуск
process_image("nature.png")

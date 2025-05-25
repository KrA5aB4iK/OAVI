from os import path

from PIL import Image
import numpy as np

def save_image(image_array, filename):
    img = Image.fromarray(image_array.astype(np.uint8))
    img.save(path.join('pictures_results', filename))


# Функция для преобразования RGB в HSI
def rgb_to_hsi(image_name):
    img = image_to_np_array(image_name).astype(np.float32) / 255.0
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    I = (R + G + B) / 3

    min_rgb = np.minimum(np.minimum(R, G), B)
    S = 1 - (min_rgb / (I + 1e-10))


    num = 0.5 * ((R - G) + (R - B))
    den = np.sqrt((R - G) ** 2 + (R - B) * (G - B)) + 1e-10
    theta = np.arccos(num / den)  # В радианах

    H = np.where(B > G, 2 * np.pi - theta, theta)  # Коррекция угла
    H = H / (2 * np.pi)  # Нормализация в диапазон [0,1]


    save_image((H * 255).astype(np.uint8), f'H_{image_name}')
    save_image((S * 255).astype(np.uint8), f'S_{image_name}')
    save_image((I * 255).astype(np.uint8), f'I_{image_name}')

    H = np.clip(H, 0, 1)
    S = np.clip(S, 0, 1)
    I = np.clip(I, 0, 1)

    hsi_image = np.stack([H, S, I], axis=2) * 255
    save_image(hsi_image.astype(np.uint8), f'HSI_{image_name}')
    return H, S, I

# Функция для растяжения изображения
def stretch_image(image, M):
    width, height = image.size
    new_width = int(width * M)
    new_height = int(height * M)

    # Создаем новый пустой массив пикселей
    new_image = Image.new("RGB", (new_width, new_height))
    pixels = new_image.load()
    original_pixels = image.load()

    for y in range(new_height):
        for x in range(new_width):
            orig_x = int(x / M)
            orig_y = int(y / M)
            pixels[x, y] = original_pixels[orig_x, orig_y]

    return new_image

# Функция для сжатия изображения
def compress_image(image, N):
    width, height = image.size
    new_width = int(width / N)
    new_height = int(height / N)

    new_image = Image.new("RGB", (new_width, new_height))
    pixels = new_image.load()
    original_pixels = image.load()

    for y in range(new_height):
        for x in range(new_width):
            orig_x = int(x * N)
            orig_y = int(y * N)
            pixels[x, y] = original_pixels[orig_x, orig_y]

    return new_image

# Функция для передискретизации за один проход
def resample_image(image, K):
    width, height = image.size
    new_width = int(width * K)
    new_height = int(height * K)

    new_image = Image.new("RGB", (new_width, new_height))
    pixels = new_image.load()
    original_pixels = image.load()

    for y in range(new_height):
        for x in range(new_width):
            orig_x = int(x / K)
            orig_y = int(y / K)
            pixels[x, y] = original_pixels[orig_x, orig_y]

    return new_image

def image_to_np_array(image_name: str) -> np.array:
    img_src = Image.open(path.join('pictures_src', image_name)).convert('RGB')
    return np.array(img_src)

def extract_rgb_components(image_name):
    img = image_to_np_array(image_name)

    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    save_image(np.stack([R, np.zeros_like(R), np.zeros_like(R)], axis=2), f'R_{image_name}')
    save_image(np.stack([np.zeros_like(G), G, np.zeros_like(G)], axis=2), f'G_{image_name}')
    save_image(np.stack([np.zeros_like(B), np.zeros_like(B), B], axis=2), f'B_{image_name}')

# Функция для интвертирования интенсивности
def invert_intensity(image_name):
    img = image_to_np_array(image_name).astype(np.float32) / 255.0
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    I = (R + G + B) / 3
    inverted_I = 1 - I

    factor = inverted_I / (I + 1e-10)  # Избегаем деления на 0
    R_new, G_new, B_new = R * factor, G * factor, B * factor

    R_new = np.clip(R_new, 0, 1)
    G_new = np.clip(G_new, 0, 1)
    B_new = np.clip(B_new, 0, 1)

    inverted_img = np.stack([R_new, G_new, B_new], axis=2) * 255

    save_image(inverted_img.astype(np.uint8), f'Inverted_{image_name}')

# Загрузка изображения
image = 'image1.png'
image_data = np.array(image)
def RGB():
    # 1. Цветовые модели
    # 1) Выделение компонент R, G, B
    extract_rgb_components(image)

    # 2) Преобразование в HSI и сохранение яркостной компоненты
    rgb_to_hsi(image)

    # 3) Инвертирование яркостной компоненты
    invert_intensity(image)

def Oversampling():
    img = Image.open('square.png')
    M = 3
    N = 2
    # 2. Передискретизация
    # 1) Растяжение изображения в M раз
    stretched_image = stretch_image(img, M)
    stretched_image.save('stretched_image.bmp')

    # 2) Сжатие изображения в N раз
    compressed_image = compress_image(img, N)
    compressed_image.save('compressed_image.bmp')

    # 3) Передискретизация в K = M/N раз (два прохода)
    K = M / N
    temp_image = stretch_image(img, M)
    resampled_image_two_pass = compress_image(temp_image, N)
    resampled_image_two_pass.save('resampled_image_two_pass.bmp')

    # 4) Передискретизация в K раз за один проход
    resampled_image_one_pass = resample_image(img, K)
    resampled_image_one_pass.save('resampled_image_one_pass.bmp')

if __name__ == "__main__":
    #RGB()
    Oversampling()

print("Все операции выполнены и результаты сохранены.")
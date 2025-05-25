import cv2
import numpy as np
import os

# Пути к папкам
input_folder = 'input_images'
output_folder = 'output_images'

# Создание папок, если не существуют
os.makedirs(input_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

# Порог для бинаризации — подбирается вручную (можно менять)
THRESHOLD = 30

# Прюитт-операторы
pruitt_kernel_x = np.array([[1, 0, -1],
                            [1, 0, -1],
                            [1, 0, -1]], dtype=np.float32)

pruitt_kernel_y = np.array([[1,  1,  1],
                            [0,  0,  0],
                            [-1, -1, -1]], dtype=np.float32)

# Обработка всех изображений в папке
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        image_path = os.path.join(input_folder, filename)
        img_color = cv2.imread(image_path)

        if img_color is None:
            print(f'Не удалось открыть {filename}')
            continue

        # Сохраняем исходное изображение
        cv2.imwrite(os.path.join(output_folder, f'{filename}_1_original.png'), img_color)

        # Преобразуем в полутоновое изображение
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(output_folder, f'{filename}_2_gray.png'), img_gray)

        # Вычисляем градиенты
        Gx = cv2.filter2D(img_gray, -1, pruitt_kernel_x)
        Gy = cv2.filter2D(img_gray, -1, pruitt_kernel_y)
        G = np.abs(Gx.astype(np.float32)) + np.abs(Gy.astype(np.float32))

        # Нормализация Gx, Gy, G до диапазона 0-255
        def normalize(mat):
            norm = cv2.normalize(mat, None, 0, 255, cv2.NORM_MINMAX)
            return norm.astype(np.uint8)

        Gx_norm = normalize(Gx)
        Gy_norm = normalize(Gy)
        G_norm = normalize(G)

        cv2.imwrite(os.path.join(output_folder, f'{filename}_3_Gx.png'), Gx_norm)
        cv2.imwrite(os.path.join(output_folder, f'{filename}_3_Gy.png'), Gy_norm)
        cv2.imwrite(os.path.join(output_folder, f'{filename}_3_G.png'), G_norm)

        # Бинаризация по порогу
        _, G_binary = cv2.threshold(G_norm, THRESHOLD, 255, cv2.THRESH_BINARY)
        cv2.imwrite(os.path.join(output_folder, f'{filename}_4_binary.png'), G_binary)

        print(f'Обработано: {filename}')

print('Готово. Результаты сохранены в output_images/')

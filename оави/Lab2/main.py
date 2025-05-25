from PIL import Image
import numpy as np

# Функция для приведения полноцветного изображения к полутоновому
def rgb_to_grayscale(image):
    image_np = np.array(image)  # Преобразуем изображение в NumPy-массив
    # Взвешенное усреднение по каналам R, G, B
    grayscale = (0.299 * image_np[:, :, 0] + 0.587 * image_np[:, :, 1] + 0.114 * image_np[:, :, 2]).astype(np.uint8)
    return Image.fromarray(grayscale, mode="L")

# Функция для пороговой обработки (бинаризации)
def threshold_binarization(image_data, threshold):
    # Применяем порог
    binary = np.where(image_data > threshold, 255, 0)
    return binary.astype(np.uint8)

# Функция для адаптивной бинаризации (метод NICK)
def adaptive_threshold_nick(image, k=-0.2, window_size=5):
    grayscale_np = np.array(image, dtype=np.float32)
    height, width = grayscale_np.shape
    half_window = window_size // 2

    # Создаём пустое бинарное изображение
    binary_np = np.zeros((height, width), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            # Определяем границы окна
            x_min, x_max = max(0, x - half_window), min(width, x + half_window + 1)
            y_min, y_max = max(0, y - half_window), min(height, y + half_window + 1)

            # Извлекаем окно 5×5
            window = grayscale_np[y_min:y_max, x_min:x_max]

            # Среднее и стандартное отклонение
            mean = np.mean(window)
            std = np.std(window)

            # Порог по NICK
            threshold = mean + k * std

            # Бинаризация
            binary_np[y, x] = 255 if grayscale_np[y, x] > threshold else 0

    return Image.fromarray(binary_np, mode="L")

# Загрузка изображения
image = Image.open('cart.png')
image_data = np.array(image)

# 1. Приведение полноцветного изображения к полутоновому
grayscale_image = rgb_to_grayscale(image_data)
grayscale_image.save('mike_grayscale_image.bmp')

# 2. Приведение полутонового изображения к монохромному
adaptive_binary_image = adaptive_threshold_nick(grayscale_image, window_size=5, k=0.2)
adaptive_binary_image.save('cart_binary_image_adaptive_nick.bmp')

print("Все операции выполнены и результаты сохранены.")
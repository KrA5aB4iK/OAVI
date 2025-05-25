import numpy as np
from PIL import Image

def dilate(image_array, window_size=3):
    """ Дилатация (расширение) – берём максимум в окне 3×3 """
    height, width = image_array.shape
    pad = window_size // 2
    padded_image = np.pad(image_array, pad, mode='edge')  # Заполняем края
    new_image = np.zeros_like(image_array)

    for y in range(height):
        for x in range(width):
            new_image[y, x] = np.max(padded_image[y:y+window_size, x:x+window_size])

    return new_image

def erode(image_array, window_size=3):
    """ Эрозия (сужение) – берём минимум в окне 3×3 """
    height, width = image_array.shape
    pad = window_size // 2
    padded_image = np.pad(image_array, pad, mode='edge')
    new_image = np.zeros_like(image_array)

    for y in range(height):
        for x in range(width):
            new_image[y, x] = np.min(padded_image[y:y+window_size, x:x+window_size])

    return new_image

def morphological_closing(image):
    """ Морфологическое закрытие: дилатация → эрозия """
    image_array = np.array(image, dtype=np.uint8)
    dilated = dilate(image_array)
    closed = erode(dilated)
    return Image.fromarray(closed)

def difference_image(original, processed):
    """ Разностное изображение: XOR для монохрома, разность для полутона """
    original_array = np.array(original, dtype=np.uint8)
    processed_array = np.array(processed, dtype=np.uint8)

    if original.mode == "1":  # Если монохромное (1-битное)
        diff_array = np.bitwise_xor(original_array, processed_array) * 255  # XOR
    else:  # Если полутоновое (8-битное)
        diff_array = np.abs(original_array.astype(int) - processed_array.astype(int))  # Модуль разности

    return Image.fromarray(diff_array.astype(np.uint8))


if __name__ == "__main__":

    input_path = "img_1.png"  # Укажи путь к изображению
    output_closed_path = "closed_cartoon.bmp"
    output_diff_path = "difference_cartoon.bmp"

    # Загружаем изображение и приводим к полутону, если оно цветное
    image = Image.open(input_path).convert("L")  # "L" = 8-битное полутоновое

    # Применяем морфологическое закрытие
    closed_image = morphological_closing(image)
    closed_image.save(output_closed_path)

    # Вычисляем разностное изображение
    diff_image = difference_image(image, closed_image)
    diff_image.save(output_diff_path)

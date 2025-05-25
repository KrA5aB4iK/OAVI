import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from os import path, makedirs
from helpers import calculate_profile, cut_black


def image_to_np_array(image_path):
    return np.array(Image.open(image_path).convert('L'))


def bar(data, bins, axis, save_path):
    plt.figure(figsize=(10, 2) if axis == 1 else (2, 10))
    if axis == 1:
        plt.bar(x=bins, height=data)
        plt.xlabel("X (columns)")
        plt.ylabel("Sum")
    elif axis == 0:
        plt.barh(y=bins, width=data)
        plt.ylabel("Y (rows)")
        plt.xlabel("Sum")
    else:
        raise ValueError('Invalid axis')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def segment_by_profile(img, profile):
    symbols = []
    rects = []

    in_char = False
    start = 0
    for x in range(len(profile)):
        if profile[x] > 0 and not in_char:
            in_char = True
            start = x
        elif profile[x] == 0 and in_char:
            in_char = False
            end = x
            symbols.append((start, end))
    if in_char:
        symbols.append((start, len(profile)))

    for start, end in symbols:
        cropped = img[:, start:end]
        # Обрезаем по вертикали
        prof_y = np.sum(cropped, axis=1)
        y_indices = np.where(prof_y > 0)[0]
        if len(y_indices) == 0:
            continue
        top, bottom = y_indices[0], y_indices[-1] + 1
        final_crop = cropped[top:bottom, :]
        rects.append((start, top, end, bottom))

    return rects


if __name__ == '__main__':
    input_path = 'results/sentence_white2.bmp'
    output_dir = 'results/symbols_white'
    rect_img_path = 'results/outlined_white.png'
    profile_dir = 'results'

    makedirs(output_dir, exist_ok=True)

    img = image_to_np_array(input_path)

    # Построение профилей
    profile_x = np.sum(img, axis=0)
    profile_y = np.sum(img, axis=1)
    bar(profile_y, np.arange(len(profile_y)), 0, path.join(profile_dir, 'profile_x_white.png'))
    bar(profile_x, np.arange(len(profile_x)), 1, path.join(profile_dir, 'profile_y_white.png'))

    # Сегментация
    rects = segment_by_profile(img, profile_x)
    print(f"Найдено символов: {len(rects)}")

    # Отрисовка прямоугольников
    rgb_img = Image.open(input_path).convert("RGB")
    draw = ImageDraw.Draw(rgb_img)
    for i, (left, top, right, bottom) in enumerate(rects):
        draw.rectangle([left, top, right, bottom], outline="red", width=1)
        letter_img = img[top:bottom, left:right]
        out_img = Image.fromarray(letter_img).convert("L")
        out_img.save(path.join(output_dir, f"letter_{i:02d}.png"))
    rgb_img.save(rect_img_path)

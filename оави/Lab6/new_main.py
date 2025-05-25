import os
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

def load_binary_image(path):
    img = Image.open(path).convert('L')
    binary = (np.array(img) < 128).astype(np.uint8)  # Чёрный = 1
    return binary

def compute_profiles(binary):
    profile_y = np.sum(binary, axis=1)
    profile_x = np.sum(binary, axis=0)
    return profile_x, profile_y

def save_profile_plot(profile, axis, name):
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(profile)), profile)
    plt.title(f"{axis}-профиль")
    plt.xlabel(axis)
    plt.ylabel("Сумма чёрных пикселей")
    plt.tight_layout()
    plt.savefig(f"results/profile_{name}_{axis}.png")
    plt.close()

def segment_by_vertical_profile(binary, min_width=2):
    vertical_profile = np.sum(binary, axis=0)
    segments = []
    in_char = False
    start = 0
    for i, value in enumerate(vertical_profile):
        if value > 0 and not in_char:
            in_char = True
            start = i
        elif value == 0 and in_char:
            in_char = False
            end = i
            if end - start >= min_width:
                segments.append((start, end))
    if in_char and binary.shape[1] - start >= min_width:
        segments.append((start, binary.shape[1]))
    return segments

def extract_bounding_boxes(binary):
    row_profile = np.sum(binary, axis=1)
    boxes = []
    in_line = False
    for y, val in enumerate(row_profile):
        if val > 0 and not in_line:
            in_line = True
            start_y = y
        elif val == 0 and in_line:
            in_line = False
            end_y = y
            line_img = binary[start_y:end_y, :]
            segments = segment_by_vertical_profile(line_img)
            for left, right in segments:
                boxes.append((left, start_y, right, end_y))
    return boxes

def draw_boxes(image_path, boxes, out_path):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for (x1, y1, x2, y2) in boxes:
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
    img.save(out_path)

def main():
    image_path = 'results/sentence_black2.bmp'
    binary = load_binary_image(image_path)

    # Профили
    profile_x, profile_y = compute_profiles(binary)
    save_profile_plot(profile_x, 'X', 'sentence_black2')
    save_profile_plot(profile_y, 'Y', 'sentence_black2')

    # Рамки
    boxes = extract_bounding_boxes(binary)
    draw_boxes(image_path, boxes, 'results/sentence_with_boxes.png')
    print(f"Найдено символов: {len(boxes)}")

if __name__ == '__main__':
    main()

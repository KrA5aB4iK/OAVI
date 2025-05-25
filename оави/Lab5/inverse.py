import os
from PIL import Image, ImageOps

input_dir = 'alphabet/direct'
output_dir = 'alphabet/inverse'
os.makedirs(output_dir, exist_ok=True)

for filename in sorted(os.listdir(input_dir)):
    if filename.lower().endswith('.png'):
        img_path = os.path.join(input_dir, filename)
        img = Image.open(img_path).convert('L')
        inverted_img = ImageOps.invert(img)
        inverted_img.save(os.path.join(output_dir, filename))

print("✅ Все изображения инвертированы и сохранены в 'alphabet/inverse'")

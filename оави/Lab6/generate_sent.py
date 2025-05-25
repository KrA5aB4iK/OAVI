from PIL import Image
from PIL.ImageOps import invert
from gen import FontDrawer
from helpers import calculate_profile, cut_black

sentence = "I love you"

if __name__ == '__main__':
    util = FontDrawer()
    result = util.render_binarized(sentence, 170)

    initial = Image.fromarray(result,'L')
    initial.save(f"results/initial_sentence_white2.bmp")
    initial = invert(initial)
    initial.save(f"results/initial_sentence_black2.bmp")

    for axis in (0, 1):
        text_profile = calculate_profile(result, axis)
        result, _ = cut_black(result, text_profile, axis)

    string = Image.fromarray(result,'L')
    string.save(f"results/sentence_white2.bmp")

    string = invert(string)
    string.save(f"results/sentence_black2.bmp")
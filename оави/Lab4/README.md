# Лабораторная работа №4. Выделение контуров на изображении.
Использовались Оператор Прюитт 3 × 3  и градиентная матрица $`G = |G_x| + |G_y|`$
## Кадр из мультфильма
Исходное изображение:

<img src="input_images/mike_grayscale_image.bmp" width="512">

Градиент по Х:

<img src="output_images/mike_grayscale_image.bmp_3_Gx.png" width="512">

Градиент по Y:

<img src="output_images/mike_grayscale_image.bmp_3_Gy.png" width="512">

Градиентная матрица G:

<img src="output_images/mike_grayscale_image.bmp_3_G.png" width="512">

Бинаризованная градиентная матрица G (Глобальная бинаризация, порог 40):

<img src="output_images/mike_grayscale_image.bmp_4_binary.png" width="512">


## Стикер
Исходное изображение:

<img src="output_images/Love.png_1_original.png" width="512">

Градиент по Х:

<img src="output_images/Love.png_3_Gx.png" width="512">

Градиент по Y:

<img src="output_images/Love.png_3_Gy.png" width="512">

Градиентная матрица G:

<img src="output_images/Love.png_3_G.png" width="512">

Бинаризованная градиентная матрица G(Глобальная бинаризация, порог 40):

<img src="output_images/Love.png_4_binary.png" width="512">


## Выводы
Алгоритм выделения контуров оператором КрПрюиттуна хорошо себя показывает и качественно выделяет все границы
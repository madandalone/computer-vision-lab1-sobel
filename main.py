import sys
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

# Ядро свёртки для градиента Gx (яркость слева направо) для вертикальных границ
KERNEL_X = np.array(
    [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
    ],
    dtype=np.float32,
)

# Ядро свёртки для градиента Gy (яркость сверху вниз) для горизонтальных границ
KERNEL_Y = np.array(
    [
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1],
    ],
    dtype=np.float32,
)


def normalize_gradient(gradient: np.ndarray) -> np.ndarray:
    # Переводим значения градиента в диапазон от 0 до 255
    max_value = gradient.max()

    if max_value == 0:
        return np.zeros_like(gradient, dtype=np.uint8)

    normalized = gradient / max_value * 255
    return normalized.astype(np.uint8)


def sobel_opencv(image: np.ndarray) -> np.ndarray:
    # OpenCV для свёртки изображения
    gradient_x = cv2.Sobel(
        image,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
        borderType=cv2.BORDER_REPLICATE,
    )
    gradient_y = cv2.Sobel(
        image,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
        borderType=cv2.BORDER_REPLICATE,
    )

    gradient = np.sqrt(gradient_x**2 + gradient_y**2)
    return normalize_gradient(gradient)


def sobel_manual(image: np.ndarray) -> np.ndarray:
    # Добавляем рамку вокруг изображения, путём повторения краевых вариантов
    source = image.astype(np.float32)
    padded = np.pad(source, 1, mode="edge")

    gradient_x = np.zeros_like(source)
    gradient_y = np.zeros_like(source)

    height, width = source.shape

    # Берем окно 3 на 3 и умножаем его на ядра Собеля
    for y in range(height):
        for x in range(width):
            window = padded[y : y + 3, x : x + 3]
            gradient_x[y, x] = np.sum(window * KERNEL_X)
            gradient_y[y, x] = np.sum(window * KERNEL_Y)

    gradient = np.sqrt(gradient_x**2 + gradient_y**2)
    return normalize_gradient(gradient)


def measure_time(function, image: np.ndarray, runs: int = 3):
    # Измерение времени работы ручного и OpenCV реализаций алгоритмов
    function(image)

    start = perf_counter()
    result = None

    for _ in range(runs):
        result = function(image)

    elapsed_ms = (perf_counter() - start) * 1000 / runs
    return result, elapsed_ms


def main():
    if len(sys.argv) != 2:
        print("Запуск python main.py images/photo.jpg")
        raise SystemExit(1)

    image_path = Path(sys.argv[1])
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"Не удалось открыть изображение  {image_path}")
        raise SystemExit(1)

    opencv_result, opencv_time = measure_time(sobel_opencv, image)
    manual_result, manual_time = measure_time(sobel_manual, image)

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    name = image_path.stem
    cv2.imwrite(str(results_dir / f"{name}_opencv.png"), opencv_result)
    cv2.imwrite(str(results_dir / f"{name}_manual.png"), manual_result)

    difference = cv2.absdiff(opencv_result, manual_result)
    cv2.imwrite(str(results_dir / f"{name}_difference.png"), difference)

    mean_difference = float(difference.mean())
    speedup = manual_time / opencv_time if opencv_time > 0 else 0

    print(f"OpenCV  {opencv_time:.3f} мс")
    print(f"Ручная версия  {manual_time:.3f} мс")
    print(f"OpenCV быстрее примерно в {speedup:.1f} раза")
    print(f"Средняя разница результатов  {mean_difference:.3f}")
    print(f"Результаты сохранены в {results_dir}")


if __name__ == "__main__":
    main()

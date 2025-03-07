import cv2
from matplotlib import pyplot as plt
import numpy as np

from resizeImage import resizeImageScale


def imageEnhancement(image, show_visualization=False):
    try:
        resized_img = resizeImageScale(image, 3500)

        gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        denoised = cv2.bilateralFilter(gray, 11, 17, 17)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        contrast_enhanced = clahe.apply(denoised)
        average_brightness = np.mean(contrast_enhanced)
        brightness_threshold = 100
        # print(average_brightness)

        if average_brightness < brightness_threshold:
            increase = 125
            brightness_matrix = np.full_like(
                contrast_enhanced, increase, dtype=np.uint8
            )
            contrast_enhanced = cv2.add(contrast_enhanced, brightness_matrix)

        _, binary = cv2.threshold(
            contrast_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        kernel_sharpen = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(cleaned, -1, kernel_sharpen)

        if show_visualization:
            plt.figure(figsize=(15, 10))

            plt.subplot(2, 3, 1)
            plt.title("1. Original Image")
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.axis("off")

            plt.subplot(2, 3, 2)
            plt.title("2. Grayscale Conversion")
            plt.imshow(gray, cmap="gray")
            plt.axis("off")

            plt.subplot(2, 3, 3)
            plt.title("3. Bilateral Filtering")
            plt.imshow(denoised, cmap="gray")
            plt.axis("off")

            plt.subplot(2, 3, 4)
            plt.title("4. Contrast Enhancement")
            plt.imshow(contrast_enhanced, cmap="gray")
            plt.axis("off")

            plt.subplot(2, 3, 5)
            plt.title("5. Thresholding")
            plt.imshow(binary, cmap="gray")
            plt.axis("off")

            plt.subplot(2, 3, 6)
            plt.title("6. Final Sharpened Image")
            plt.imshow(sharpened, cmap="gray")
            plt.axis("off")

            plt.tight_layout()
            plt.show()

        return sharpened

    except Exception as e:
        print(f"Error during enhancement: {str(e)}")
        return image

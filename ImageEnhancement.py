import cv2
from matplotlib import pyplot as plt
import numpy as np


def imageEnhancement(image, show_visualization=True):
    """
    Apply specialized enhancement techniques for license plate images
    """
    try:

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply bilateral filter to reduce noise while preserving edges
        denoised = cv2.bilateralFilter(gray, 11, 17, 17)

        # Improve contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        contrast_enhanced = clahe.apply(denoised)

        # Thresholding to separate text from background
        _, binary = cv2.threshold(
            contrast_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Remove small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Sharpen the image
        kernel_sharpen = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(cleaned, -1, kernel_sharpen)

        if show_visualization:
            # Create a figure with subplots for each enhancement step
            plt.figure(figsize=(15, 10))

            # Original Image
            plt.subplot(2, 3, 1)
            plt.title("1. Original Image")
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.axis("off")

            # Grayscale Conversion
            plt.subplot(2, 3, 2)
            plt.title("2. Grayscale Conversion")
            plt.imshow(gray, cmap="gray")
            plt.axis("off")

            # Bilateral Filtering (Denoising)
            plt.subplot(2, 3, 3)
            plt.title("3. Bilateral Filtering")
            plt.imshow(denoised, cmap="gray")
            plt.axis("off")

            # Contrast Enhancement with CLAHE
            plt.subplot(2, 3, 4)
            plt.title("4. Contrast Enhancement")
            plt.imshow(contrast_enhanced, cmap="gray")
            plt.axis("off")

            # Thresholding
            plt.subplot(2, 3, 5)
            plt.title("5. Thresholding")
            plt.imshow(binary, cmap="gray")
            plt.axis("off")

            # Final Sharpened Image
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

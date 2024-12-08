import cv2
import numpy as np


def enhance_image(image):
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

        return sharpened

    except Exception as e:
        print(f"Error during enhancement: {str(e)}")
        return image

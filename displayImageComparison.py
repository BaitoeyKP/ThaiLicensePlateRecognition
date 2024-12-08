import numpy as np
import cv2
from PIL import Image, ImageDraw


def display_comparison(image1, image2):
    """
    Display two images side by side.

    Parameters:
    image1 (np.ndarray): First image as a NumPy array.
    image2 (np.ndarray): Second image as a NumPy array.
    """
    try:
        # Check if inputs are valid
        if image1 is None or image2 is None:
            raise ValueError("One or both images are None")
        if not isinstance(image1, np.ndarray) or not isinstance(image2, np.ndarray):
            raise TypeError("Inputs must be NumPy arrays")

        # Convert from BGR to RGB for PIL
        image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

        # Get dimensions
        height = max(image1_rgb.shape[0], image2_rgb.shape[0])
        width = image1_rgb.shape[1] + image2_rgb.shape[1]

        # Create comparison image
        comparison = np.zeros((height, width, 3), dtype=np.uint8)

        # Copy images side by side
        comparison[: image1_rgb.shape[0], : image1_rgb.shape[1]] = image1_rgb
        comparison[: image2_rgb.shape[0], image1_rgb.shape[1] :] = image2_rgb

        # Convert to PIL for displaying
        comparison_pil = Image.fromarray(comparison)

        # Add labels
        draw = ImageDraw.Draw(comparison_pil)
        draw.text((10, 10), "Original Image", fill=(255, 255, 255))
        draw.text(
            (image1_rgb.shape[1] + 10, 10), "Enhanced Image", fill=(255, 255, 255)
        )

        # Display the comparison
        comparison_pil.show(title="Image Comparison")

    except Exception as e:
        print(f"Error displaying comparison: {str(e)}")

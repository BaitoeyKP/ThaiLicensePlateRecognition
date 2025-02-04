import cv2
from matplotlib import pyplot as plt


def resizeImageScale(image, scale):
    width = int(image.shape[1] * scale / 100)
    height = int(image.shape[0] * scale / 100)
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
    return resized


def resizeImageFix(image, width, height, show_visualization=False):
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    if show_visualization:
        # Convert BGR to RGB for proper display with matplotlib
        original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Display original image
        ax1.imshow(original_rgb)
        ax1.set_title(f"Original ({image.shape[1]}x{image.shape[0]})")
        ax1.axis("off")

        # Display resized image
        ax2.imshow(resized_rgb)
        ax2.set_title(f"Resized ({width}x{height})")
        ax2.axis("off")

        plt.tight_layout()
        plt.show()

    return resized

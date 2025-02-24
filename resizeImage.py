import cv2
from matplotlib import pyplot as plt


def resizeImageScale(image, scale):
    try:
        width = int(image.shape[1] * scale / 100)
        height = int(image.shape[0] * scale / 100)
        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
        return resized
    except Exception as e:
        print(f"Error during resizeImageScale: {str(e)}")
        return None


def resizeImageFix(image, width, height, show_visualization=False):
    try:
        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

        if show_visualization:
            original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            ax1.imshow(original_rgb)
            ax1.set_title(f"Original ({image.shape[1]}x{image.shape[0]})")
            ax1.axis("off")

            ax2.imshow(resized_rgb)
            ax2.set_title(f"Resized ({width}x{height})")
            ax2.axis("off")

            plt.tight_layout()
            plt.show()

        return resized

    except Exception as e:
        print(f"Error during resizeImageFix: {str(e)}")
        return None

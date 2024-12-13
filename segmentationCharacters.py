import numpy as np
import cv2
import matplotlib.pyplot as plt
from imageEnhancement import imageEnhancement
from resizeImage import resizeImage


def segmentationCharacters(img, show_visualization=True):
    try:
        if img is None or img.size == 0:
            print("Invalid input image")
            return None

        # Store original before any processing
        original_img = img.copy()

        # Resize image first and use this size consistently
        resized_img = resizeImage(img, 500)
        improved_image = imageEnhancement(resized_img, False)

        # Analysis on improved image - using column sums
        column_sums = np.sum(improved_image, axis=0)
        column_sums_normalized = column_sums / np.max(column_sums)
        column_sums_inverted = 1 - column_sums_normalized

        # Find regions of high intensity
        threshold = 0.05
        high_intensity_cols = np.where(column_sums_inverted > threshold)[0]

        if len(high_intensity_cols) == 0:
            print("No high intensity regions found")
            return None

        # Find continuous high intensity regions
        high_regions = []
        start_idx = high_intensity_cols[0]

        for i in range(1, len(high_intensity_cols)):
            if high_intensity_cols[i] != high_intensity_cols[i - 1] + 1:
                high_regions.append([start_idx, high_intensity_cols[i - 1]])
                start_idx = high_intensity_cols[i]
        high_regions.append([start_idx, high_intensity_cols[-1]])

        # Process high intensity regions
        cropped_images = []
        crop_regions = []

        h, w = resized_img.shape[:2]
        for region in high_regions:
            min_col = region[0]
            max_col = region[1]

            # Expand crop region
            min_crop = max(int(min_col) - 20, 0)  # Reduced padding for tighter crops
            max_crop = min(int(max_col) + 20, w)

            if min_crop >= max_crop or max_crop > w:
                continue

            # Calculate mean intensity for this region
            region_intensity = np.mean(column_sums_inverted[min_col:max_col])
            print(
                np.min(column_sums_inverted[min_col:max_col]),
                np.max(column_sums_inverted[min_col:max_col]),
                np.mean(column_sums_inverted[min_col:max_col]),
                np.median(column_sums_inverted[min_col:max_col]),
            )
            # Include regions with significant content
            if (
                region_intensity > threshold
                and 0.4 < np.max(column_sums_inverted[min_col:max_col]) < 0.95
            ):
                cropped_image = resized_img[:, min_crop:max_crop]
                if (
                    max_col - min_col > 500
                ):  # Reduced minimum width for character detection
                    cropped_images.append(cropped_image)
                    crop_regions.append((min_crop, max_crop))

        if show_visualization:
            # Calculate layout
            n_crops = len(cropped_images)
            if n_crops == 0:
                return None

            # Create figure with top row for analysis and bottom rows for characters
            fig = plt.figure(figsize=(15, 10))

            # Top row: analysis plots (3 plots in row 1)
            plt.subplot(2, 3, 1)
            plt.title("Original Image")
            plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
            plt.axis("off")

            plt.subplot(2, 3, 2)
            plt.title("Improved Image")
            plt.imshow(cv2.cvtColor(improved_image, cv2.COLOR_BGR2RGB))
            plt.axis("off")

            plt.subplot(2, 3, 3)
            plt.title("Column Analysis")
            plt.plot(
                range(len(column_sums_inverted)),
                column_sums_inverted,
                "k-",
                label="Column Sums",
            )
            plt.axhline(y=threshold, color="r", linestyle="--", label="Threshold")

            # Highlight all detected regions
            for i, region in enumerate(crop_regions):
                color = "blue" if i % 2 == 0 else "green"  # Alternating colors
                plt.axvspan(
                    region[0],
                    region[1],
                    color=color,
                    alpha=0.3,
                    label=f"Character {i+1}",
                )

            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.ylabel("Inverted Normalized Sum")
            plt.xlabel("Column Number")

            # Bottom row: characters (3 plots in row 2)
            # Display first 6 characters in the bottom row
            for i in range(n_crops):
                plt.subplot(
                    2, n_crops, i + n_crops + 1
                )  # Start from position 7 (second row)
                plt.title(f"Char {i+1}")
                plt.imshow(cv2.cvtColor(cropped_images[i], cv2.COLOR_BGR2RGB))
                plt.axis("off")

            plt.tight_layout()
            plt.show()

        return cropped_images

    except Exception as e:
        print(f"Error during segmentation: {str(e)}")
        return None

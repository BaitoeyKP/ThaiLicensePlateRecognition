import numpy as np
import cv2
import matplotlib.pyplot as plt
from imageEnhancement import imageEnhancement
from resizeImage import resizeImage


def segmentationRow(img, show_visualization=True):
    try:
        if img is None or img.size == 0:
            print("Invalid input image")
            return None, None

        # Store original before any processing
        original_img = img.copy()

        # Resize image first and use this size consistently
        resized_img = resizeImage(img, 500)
        improved_image = imageEnhancement(resized_img, False)

        # Analysis on improved image
        row_sums = np.sum(improved_image, axis=1)
        row_sums_normalized = row_sums / np.max(row_sums)
        row_sums_inverted = 1 - row_sums_normalized

        # First, find regions of high intensity
        threshold = 0.2
        high_intensity_rows = np.where(row_sums_inverted > threshold)[0]

        if len(high_intensity_rows) == 0:
            print("No high intensity regions found")
            return None, None

        # Find continuous high intensity regions
        high_regions = []
        start_idx = high_intensity_rows[0]

        for i in range(1, len(high_intensity_rows)):
            if high_intensity_rows[i] != high_intensity_rows[i - 1] + 1:
                high_regions.append([start_idx, high_intensity_rows[i - 1]])
                start_idx = high_intensity_rows[i]
        high_regions.append([start_idx, high_intensity_rows[-1]])

        # Process high intensity regions
        cropped_images = []
        heights = []
        crop_regions = []
        region_intensities = []

        h, w = resized_img.shape[:2]
        for region in high_regions:
            min_row = region[0]
            max_row = region[1]

            # Expand crop region
            min_crop = max(int(min_row) - 100, 0)
            max_crop = min(int(max_row) + 100, h)

            if min_crop >= max_crop or max_crop > h:
                continue

            # Calculate mean intensity for this region
            region_intensity = np.mean(row_sums_inverted[min_row:max_row])

            # Only include regions with significant content
            if region_intensity > threshold:
                cropped_image = resized_img[min_crop:max_crop, :, :]
                if (
                    max_row - min_row > 100
                    and np.max(row_sums_inverted[min_row:max_row]) < 0.7
                ):
                    print(np.max(row_sums_inverted[min_row:max_row]))
                    cropped_images.append(cropped_image)
                    heights.append(max_row - min_row)
                    crop_regions.append((min_crop, max_crop))
                    region_intensities.append(region_intensity)

        # if len(cropped_images) < 4:
        #     print(f"Not enough regions found: {len(cropped_images)}")
        #     return None, None

        # Select data (high 2) and province (high 4) regions
        # high 1 is index 0, high 2 is index 1, etc.
        data_idx = min(1, len(cropped_images) - 1)  # high 2
        province_idx = min(3, len(cropped_images) - 1)  # high 4

        data_img = cropped_images[0]
        province_img = cropped_images[1]
        data_region = crop_regions[0]
        province_region = crop_regions[1]

        if show_visualization:
            fig = plt.figure(figsize=(15, 10))

            plt.subplot(2, 3, 1)
            plt.title("Original Image")
            plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
            plt.axis("off")

            plt.subplot(2, 3, 2)
            plt.title("Improved Image")
            plt.imshow(cv2.cvtColor(improved_image, cv2.COLOR_BGR2RGB))
            plt.axis("off")

            plt.subplot(2, 3, 3)
            plt.title("Row Sums Analysis")

            plt.plot(
                row_sums_inverted, range(len(row_sums_inverted)), "k-", label="Row Sums"
            )
            plt.axvline(x=threshold, color="r", linestyle="--", label="Threshold")

            # Highlight detected regions
            colors = [
                "green",
                "blue",
            ]  # green for data, blue for province
            for i, region in enumerate(crop_regions):
                color = colors[i % len(colors)]
                alpha = 0.4 if (i == data_idx or i == province_idx) else 0.1
                label = "Province" if i == data_idx else ("Data")
                plt.axhspan(
                    region[0],
                    region[1],
                    color=color,
                    alpha=alpha,
                    label=label,
                )

            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.xlabel("Inverted Normalized Sum")
            plt.ylabel("Row Number")
            plt.gca().invert_yaxis()

            plt.subplot(2, 3, 5)
            plt.title("Data")
            if data_img is not None and data_img.size > 0:
                plt.imshow(cv2.cvtColor(data_img, cv2.COLOR_BGR2RGB))
            plt.axis("off")

            plt.subplot(2, 3, 6)
            plt.title("Province")
            if province_img is not None and province_img.size > 0:
                plt.imshow(cv2.cvtColor(province_img, cv2.COLOR_BGR2RGB))
            plt.axis("off")

            plt.tight_layout()
            plt.show()

        return data_img, province_img

    except Exception as e:
        print(f"Error during segmentation: {str(e)}")
        return None, None

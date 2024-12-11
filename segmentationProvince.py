import numpy as np
import cv2
import matplotlib.pyplot as plt
from imageEnhancement import imageEnhancement
from resizeImage import resizeImage


def segmentationProvince(img, show_visualization=True):
    """
    Segment an image to extract individual text elements.

    Args:
        img (np.ndarray): The input image to be segmented.
        show_visualization (bool): Whether to display a visualization of the segmentation process.

    Returns:
        list: A list of cropped image regions corresponding to the extracted text elements.
    """
    try:
        if img is None or img.size == 0:
            print("Invalid input image")
            return None

        # Store original image before any processing
        original_img = img.copy()

        # Resize image and use this size consistently
        resized_img = resizeImage(img, 500)
        enhanced_image = imageEnhancement(resized_img, False)

        # Analyze the enhanced image using column sums
        column_sums = np.sum(enhanced_image, axis=0)
        column_sums_normalized = column_sums / np.max(column_sums)
        column_sums_inverted = 1 - column_sums_normalized

        # Find regions of high intensity
        threshold = 0.1
        high_intensity_cols = np.where(column_sums_inverted < threshold)[0]

        if len(high_intensity_cols) == 0:
            print("No high intensity regions found")
            return None

        # Identify continuous high-intensity regions with expanded boundaries
        high_regions = []
        start_idx = high_intensity_cols[0]
        min_region_distance = 100  # Maximum allowed gap between high-intensity regions

        for i in range(1, len(high_intensity_cols)):
            if (
                high_intensity_cols[i] - high_intensity_cols[i - 1]
                <= min_region_distance
            ):
                # Continue the current region
                continue
            else:
                # End the current region and start a new one
                high_regions.append([start_idx, high_intensity_cols[i - 1]])
                start_idx = high_intensity_cols[i]

        # Add the last region
        high_regions.append([start_idx, high_intensity_cols[-1]])

        # Process the high-intensity regions
        cropped_images = []
        crop_regions = []

        h, w = resized_img.shape[:2]
        crop_regions_temp = []
        for region in high_regions:
            min_col = region[0]
            max_col = region[1]

            # Expand the crop region with padding
            min_crop = max(int(min_col) - 20, 0)
            max_crop = min(int(max_col) + 20, w)

            if min_crop >= max_crop or max_crop > w:
                continue

            # Calculate the mean intensity for this region
            region_intensity = np.mean(column_sums_inverted[min_col:max_col])

            # Include regions with significant content and reasonable width
            if region_intensity < threshold and max_col - min_col >= 500:
                crop_regions_temp.append([min_crop, max_crop])
        min_crop = max(crop_regions_temp[0][1] - 500, 0)
        max_crop = min(crop_regions_temp[-1][0] + 500, w)
        cropped_image = resized_img[:, min_crop:max_crop, :]
        cropped_images.append(cropped_image)
        crop_regions.append((min_crop, max_crop))

        if show_visualization:
            # Generate a visualization of the segmentation process
            n_crops = len(cropped_images)
            if n_crops == 0:
                return None

            fig = plt.figure(figsize=(15, 10))

            # Top row: analysis plots
            plt.subplot(2, 3, 1)
            plt.title("Original Image")
            plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
            plt.axis("off")

            plt.subplot(2, 3, 2)
            plt.title("Improved Image")
            plt.imshow(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))
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

            # Highlight the detected regions
            for i, region in enumerate(crop_regions):
                color = "blue" if i % 2 == 0 else "green"
                plt.axvspan(
                    region[0],
                    region[1],
                    color=color,
                    alpha=0.3,
                    label="Province",
                )

            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.ylabel("Inverted Normalized Sum")
            plt.xlabel("Column Number")

            # Bottom row: extracted characters
            for i in range(min(6, n_crops)):
                plt.subplot(2, 6, i + 7)
                plt.title("Province")
                plt.imshow(cv2.cvtColor(cropped_images[i], cv2.COLOR_BGR2RGB))
                plt.axis("off")

            plt.tight_layout()
            plt.show()

            # Display additional characters in separate pages
            if n_crops > 6:
                remaining_crops = n_crops - 6
                pages_needed = (remaining_crops + 11) // 12

                for page in range(pages_needed):
                    start_idx = 6 + page * 12
                    end_idx = min(start_idx + 12, n_crops)

                    fig = plt.figure(figsize=(15, 5))
                    plt.suptitle(f"Additional Characters (Page {page + 1})")

                    for i in range(start_idx, end_idx):
                        plt.subplot(2, 6, i - start_idx + 1)
                        plt.title(f"Char {i+1}")
                        plt.imshow(cv2.cvtColor(cropped_images[i], cv2.COLOR_BGR2RGB))
                        plt.axis("off")

                    plt.tight_layout()
                    plt.show()

        return cropped_images

    except Exception as e:
        print(f"Error during segmentation: {str(e)}")
        return None

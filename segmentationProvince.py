import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from resizeImage import resizeImageScale


def segmentationProvince(img, input_filename, show_visualization=False, save_path=None):
    try:
        if img is None or img.size == 0:
            print("Invalid input image")
            return None

        # Store original before any processing
        original_img = img.copy()
        # Resize image first and use this size consistently
        resized_img = resizeImageScale(img, 500)

        h, w = resized_img.shape[:2]
        # Analysis on improved image - using column sums
        column_sums = np.sum(resized_img, axis=0)
        column_sums_normalized = column_sums / np.max(column_sums)
        column_sums_inverted = 1 - column_sums_normalized

        # Find regions of high intensity
        threshold = 0.15

        high_intensity_cols = np.where(column_sums_inverted > threshold)[0]
        if len(high_intensity_cols) == 0:
            print("No high intensity regions found")
            return None

        # Find continuous high intensity regions
        high_regions = []
        start_idx = high_intensity_cols[0]

        left_border = 600
        right_border = w - left_border

        for i in range(1, len(high_intensity_cols)):
            current_range_max = 0

            # Check width of current region
            if start_idx < high_intensity_cols[i - 1]:
                current_range_max = np.max(
                    column_sums_inverted[start_idx : high_intensity_cols[i - 1]]
                )
            # Adjust min_region_distance based on current region width
            if (
                start_idx > left_border
                and high_intensity_cols[i - 1] < right_border
                and current_range_max < 0.9
            ):
                current_min_region_distance = 1000
            else:
                current_min_region_distance = 1

            if (
                high_intensity_cols[i] - high_intensity_cols[i - 1]
                <= current_min_region_distance
            ):
                # Continue the current region
                continue
            else:
                # End the current region and start a new one
                high_regions.append([start_idx, high_intensity_cols[i - 1]])
                start_idx = high_intensity_cols[i]
        high_regions.append([start_idx, high_intensity_cols[-1]])

        # Process high intensity regions
        cropped_images = []
        crop_regions = []
        for region in high_regions:
            min_col = region[0]
            max_col = region[1]

            # Expand crop region
            min_crop = max(int(min_col) - 50, 0)  # Reduced padding for tighter crops
            max_crop = min(int(max_col) + 50, w)
            if min_crop >= max_crop or max_crop > w:
                continue

            # Calculate mean intensity for this region
            region_intensity = np.mean(column_sums_inverted[min_col:max_col])
            if region_intensity > threshold:
                # print(
                #     "min_col:",
                #     min_col,
                #     " | max_col:",
                #     max_col,
                #     " | size:",
                #     max_col - min_col,
                # )
                cropped_image = resized_img[:, min_crop:max_crop]
                cropped_images.append(cropped_image)
                crop_regions.append((min_crop, max_crop))

            if len(cropped_images) < 1:
                cropped_image = resized_img[:, left_border:right_border]
                cropped_images.append(cropped_image)
                crop_regions.append((left_border, right_border))

        if show_visualization:
            # Find the largest crop
            largest_crop_idx = max(
                range(len(cropped_images)), key=lambda i: cropped_images[i].shape[1]
            )

            # Create figure with top row for analysis and bottom rows for characters
            fig = plt.figure(figsize=(15, 10))

            # Top row: analysis plots (3 plots in row 1)
            plt.subplot(2, 3, 1)
            plt.title("Original Image")
            plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
            plt.axis("off")

            plt.subplot(2, 3, 2)
            plt.title("Improved Image")
            plt.imshow(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))
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

            # Bottom row: Show only the largest character
            plt.subplot(2, 3, 5)  # Center position in bottom row
            plt.title("Province")
            plt.imshow(
                cv2.cvtColor(cropped_images[largest_crop_idx], cv2.COLOR_BGR2RGB)
            )
            plt.axis("off")

            plt.tight_layout()
            plt.show()

        # Add saving functionality
        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(save_path, exist_ok=True)

            # Find the largest crop - only once per input image
            if len(cropped_images) > 0:  # Make sure we have cropped images
                largest_crop_idx = max(
                    range(len(cropped_images)), key=lambda i: cropped_images[i].shape[1]
                )

                # Save only the largest cropped image once
                filename = f"{input_filename}_Province.png"
                filepath = os.path.join(save_path, filename)

                # Save the image
                cv2.imwrite(filepath, cropped_images[largest_crop_idx])
                print(f"Saved largest province crop to: {filepath}")

        return cropped_images

    except Exception as e:
        print(f"Error during segmentation: {str(e)}")
        return None

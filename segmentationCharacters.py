import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from resizeImage import resizeImageScale


def segmentationCharacters(
    img, input_filename, show_visualization=False, save_path=None
):
    try:
        if img is None or img.size == 0:
            print("Invalid input image")
            return None

        original_img = img.copy()
        resized_img = resizeImageScale(img, 500)

        h, w = resized_img.shape[:2]
        column_sums = np.sum(resized_img, axis=0)
        column_sums_normalized = column_sums / np.max(column_sums)
        column_sums_inverted = 1 - column_sums_normalized

        if np.mean(column_sums_inverted) > 0.36:
            threshold = 0.125
        else:
            threshold = 0.05
        # print("mean:", np.mean(column_sums_inverted), " | threshold:", threshold)

        high_intensity_cols = np.where(column_sums_inverted > threshold)[0]
        if len(high_intensity_cols) == 0:
            print("No high intensity regions found")
            return None

        high_regions = []
        start_idx = high_intensity_cols[0]
        left_border = 50
        right_border = w - left_border

        for i in range(1, len(high_intensity_cols)):
            current_width = high_intensity_cols[i - 1] - start_idx
            if start_idx < high_intensity_cols[i - 1]:
                current_range_max = np.max(
                    column_sums_inverted[start_idx : high_intensity_cols[i - 1]]
                )

            if (
                1000 < current_width < 1300
                and start_idx > left_border
                and high_intensity_cols[i - 1] < right_border
                and current_range_max > 0.5
            ):
                current_min_region_distance = 120
            elif (
                800 < current_width < 1000
                and start_idx > left_border
                and high_intensity_cols[i - 1] < right_border
                and current_range_max > 0.5
            ):
                current_min_region_distance = 450
            else:
                current_min_region_distance = 1

            if (
                high_intensity_cols[i] - high_intensity_cols[i - 1]
                <= current_min_region_distance
            ):
                continue
            else:
                high_regions.append([start_idx, high_intensity_cols[i - 1]])
                start_idx = high_intensity_cols[i]

        high_regions.append([start_idx, high_intensity_cols[-1]])

        cropped_images = []
        crop_regions = []
        for region in high_regions:
            min_col = region[0]
            max_col = region[1]

            min_crop = max(int(min_col) - 20, 0)
            max_crop = min(int(max_col) + 20, w)
            if min_crop >= max_crop or max_crop > w:
                continue

            region_intensity = np.mean(column_sums_inverted[min_col:max_col])
            # print(
            #     np.min(column_sums_inverted[min_col:max_col]),
            # np.max(column_sums_inverted[min_col:max_col]),
            # np.mean(column_sums_inverted[min_col:max_col]),
            # np.median(column_sums_inverted[min_col:max_col]),
            # )
            if (
                region_intensity > threshold
                and 0.4 < np.max(column_sums_inverted[min_col:max_col]) < 0.99
                and min_col > left_border
                and max_col < right_border
            ):
                # print(
                #     "min_col:",
                #     min_col,
                #     " | max_col:",
                #     max_col,
                # )
                cropped_image = resized_img[:, min_crop:max_crop]
                cropped_images.append(cropped_image)
                crop_regions.append((min_crop, max_crop))

        if show_visualization:
            n_crops = len(cropped_images)
            # if n_crops == 0:
            #     return None

            fig = plt.figure(figsize=(15, 10))

            plt.subplot(2, 2, 1)
            plt.title("Original Image")
            plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
            plt.axis("off")

            plt.subplot(2, 2, 2)
            plt.title("Column Analysis")
            plt.plot(
                range(len(column_sums_inverted)),
                column_sums_inverted,
                "k-",
                label="Column Sums",
            )
            plt.axhline(y=threshold, color="r", linestyle="--", label="Threshold")

            for i, region in enumerate(crop_regions):
                color = "blue" if i % 2 == 0 else "green"
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

            for i in range(n_crops):
                plt.subplot(2, n_crops, i + n_crops + 1)
                plt.title(f"Char {i+1}")
                plt.imshow(cv2.cvtColor(cropped_images[i], cv2.COLOR_BGR2RGB))
                plt.axis("off")

            plt.tight_layout()
            plt.show()

        if save_path:
            os.makedirs(save_path, exist_ok=True)

            for idx, (cropped_image) in enumerate(cropped_images):
                filename = f"{input_filename}_cropped_{idx}.png"
                filepath = os.path.join(save_path, filename)

                cv2.imwrite(filepath, cropped_image)
                print(f"Saved cropped image {idx+1} to: {filepath}")

        return cropped_images

    except Exception as e:
        print(f"Error during segmentation: {str(e)}")
        return None

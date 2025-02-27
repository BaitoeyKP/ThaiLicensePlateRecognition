import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from resizeImage import resizeImageScale


def segmentationProvince(
    img,
    input_filename,
    show_visualization=False,
    save_path=None,
    save_show_result_path=None,
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
        threshold = 0.15

        high_intensity_cols = np.where(column_sums_inverted > threshold)[0]
        if len(high_intensity_cols) == 0:
            print("No high intensity regions found")
            return None

        high_regions = []
        start_idx = high_intensity_cols[0]

        left_border = 600
        right_border = w - left_border

        for i in range(1, len(high_intensity_cols)):
            current_range_max = 0
            if start_idx < high_intensity_cols[i - 1]:
                current_range_max = np.max(
                    column_sums_inverted[start_idx : high_intensity_cols[i - 1]]
                )
            if (
                start_idx > left_border
                and high_intensity_cols[i - 1] < right_border
                and current_range_max < 0.9
            ):
                current_min_region_distance = 1000
            else:
                current_min_region_distance = 500

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

            min_crop = max(int(min_col) - 50, 0)
            max_crop = min(int(max_col) + 50, w)
            if min_crop >= max_crop or max_crop > w:
                continue

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

        largest_crop_idx = max(
            range(len(cropped_images)), key=lambda i: cropped_images[i].shape[1]
        )

        if show_visualization:
            fig = plt.figure(figsize=(15, 10))

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

            plt.subplot(2, 3, 5)
            plt.title("Province")
            plt.imshow(
                cv2.cvtColor(cropped_images[largest_crop_idx], cv2.COLOR_BGR2RGB)
            )
            plt.axis("off")

            plt.tight_layout()
            if save_show_result_path:
                filename = f"{input_filename}_Province.png"
                save_show_result_path = os.path.join(
                    os.path.dirname(save_show_result_path), filename
                )
                os.makedirs(os.path.dirname(save_show_result_path), exist_ok=True)
                plt.savefig(save_show_result_path, dpi=300, bbox_inches="tight")
            plt.show()

        if save_path:
            os.makedirs(save_path, exist_ok=True)

            if len(cropped_images) > 0:
                largest_crop_idx = max(
                    range(len(cropped_images)), key=lambda i: cropped_images[i].shape[1]
                )

                filename = f"{input_filename}_ProvinceCropped.png"
                filepath = os.path.join(save_path, filename)

                cv2.imwrite(filepath, cropped_images[largest_crop_idx])
                print(f"Saved largest province crop to: {filepath}")

        return cropped_images[largest_crop_idx]

    except Exception as e:
        print(f"Error during segmentation: {str(e)}")
        return None

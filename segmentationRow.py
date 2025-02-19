import numpy as np
import cv2
import matplotlib.pyplot as plt


def segmentationRow(img, show_visualization=False):
    try:
        if img is None or img.size == 0:
            print("Invalid input image")
            return None, None

        original_img = img.copy()
        row_sums = np.sum(img, axis=1)
        row_sums_normalized = row_sums / np.max(row_sums)
        row_sums_inverted = 1 - row_sums_normalized
        threshold = 0.18
        # print(
        #     "original - min : {:.5f} | max : {:.5f} | mean: {:.5f}".format(
        #         np.min(row_sums_inverted),
        #         np.max(row_sums_inverted),
        #         np.mean(row_sums_inverted),
        #     )
        # )
        high_intensity_rows = np.where(row_sums_inverted > threshold)[0]
        h, w = img.shape[:2]
        if len(high_intensity_rows) == 0:
            print("No high intensity regions found")
            return None, None

        high_regions = []
        start_idx = high_intensity_rows[0]

        for i in range(1, len(high_intensity_rows)):
            if high_intensity_rows[i] != high_intensity_rows[i - 1] + 1:
                high_regions.append([start_idx, high_intensity_rows[i - 1]])
                start_idx = high_intensity_rows[i]

        high_regions.append([start_idx, high_intensity_rows[-1]])

        cropped_images = []
        heights = []
        crop_regions = []
        region_intensities = []

        min_row_temp = 0
        for region in high_regions:
            min_row = region[0]
            max_row = region[1]

            min_crop = max(int(min_row) - 100, 0)
            max_crop = min(int(max_row) + 100, h)

            if min_crop >= max_crop or max_crop > h:
                continue

            region_intensity = np.mean(row_sums_inverted[min_row:max_row])

            if region_intensity > threshold:
                cropped_image = img[min_crop:max_crop, :]
                size = max_row - min_row
                if size > 200 and (
                    size / h >= 0.1 or np.max(row_sums_inverted[min_row:max_row]) < 0.75
                ):
                    min_row_temp = max_row + 100
                    # print(
                    #     "min : {:.5f} | max : {:.5f} | mean : {:.5f}".format(
                    #         np.min(row_sums_inverted[min_row:max_row]),
                    #         np.max(row_sums_inverted[min_row:max_row]),
                    #         np.mean(row_sums_inverted[min_row:max_row]),
                    #     ),
                    #     " | h : ",
                    #     size,
                    #     " | h-min : ",
                    #     h - min_row,
                    # )
                    cropped_images.append(cropped_image)
                    heights.append(max_row - min_row)
                    crop_regions.append((min_crop, max_crop))
                    region_intensities.append(region_intensity)

        if len(cropped_images) < 2:
            print(f"Not enough regions found: {len(cropped_images)}")
            cropped_image = img[h - 700 : h, :]
            cropped_images.append(cropped_image)
            heights.append(w - min_row)
            crop_regions.append((min_row_temp, w))
            region_intensities.append(region_intensity)

        data_img = cropped_images[0]
        province_img = cropped_images[1]

        data_idx = min(1, len(cropped_images) - 1)
        province_idx = min(3, len(cropped_images) - 1)

        if show_visualization:
            fig = plt.figure(figsize=(15, 10))

            plt.subplot(2, 2, 1)
            plt.title("Original Image")
            plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
            plt.axis("off")

            plt.subplot(2, 2, 2)
            plt.title("Row Sums Analysis")

            plt.plot(
                row_sums_inverted, range(len(row_sums_inverted)), "k-", label="Row Sums"
            )
            plt.axvline(x=threshold, color="r", linestyle="--", label="Threshold")
            colors = [
                "green",
                "blue",
            ] 
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

            plt.subplot(2, 2, 3)
            plt.title("Data")
            if data_img is not None and data_img.size > 0:
                plt.imshow(cv2.cvtColor(data_img, cv2.COLOR_BGR2RGB))
            plt.axis("off")

            plt.subplot(2, 2, 4)
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

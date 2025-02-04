import os

import cv2
from imageEnhancement import imageEnhancement
from loadedImage import loadImageFromFolder
from segmentationCharacters import segmentationCharacters
from segmentationProvince import segmentationProvince
from segmentationRow import segmentationRow

folder_path = "../dataset/internet"

# loaded_images, filenames = loadImageFromFolder(folder_path)
# for img, filename in zip(loaded_images, filenames):
#     enhance_image = imageEnhancement(img)
#     data, province = segmentationRow(enhance_image, True)
#     charactersCrop = segmentationCharacters(
#         data, filename, save_path="../output/characters"
#     )
#     provinceCrop = segmentationProvince(
#         province, filename, save_path="../output/province"
#     )

filename = "7809975aef6a2d2d78f36b8d0a27d090.jpg"
file_path = os.path.join(folder_path, filename)
img = cv2.imread(file_path)
enhance_image = imageEnhancement(img)
data, province = segmentationRow(enhance_image, True)
charactersCrop = segmentationCharacters(
    data,
    filename,
    # save_path="../output/characters",
)
provinceCrop = segmentationProvince(
    province, filename, True, save_path="../output/province"
)

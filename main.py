import cv2
import os

from imageEnhancement import imageEnhancement
from loadedImage import load_images_from_folder
from segmentationCharacters import segmentationCharacters
from segmentationProvince import segmentationProvince
from segmentationRow import segmentationRow

folder_path = "../new"

loaded_images = load_images_from_folder(folder_path)
for img in loaded_images:
    print("---------------")
    enhance_image = imageEnhancement(img)
    data, province = segmentationRow(enhance_image)
    charactersCrop = segmentationCharacters(data)
    provinceCrop = segmentationProvince(province)

# filename = "24_02_01_V00002.jpg"
# file_path = os.path.join(folder_path, filename)
# img = cv2.imread(file_path)
# enhance_image = imageEnhancement(img, False)
# data, province = segmentationRow(enhance_image, False)
# charactersCrop = segmentationCharacters(data)
# provinceCrop = segmentationProvince(province)

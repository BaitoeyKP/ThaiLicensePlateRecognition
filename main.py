import cv2
from dotenv import load_dotenv
import os

from imageEnhancement import imageEnhancement
from loadedImage import load_images_from_folder
from segmentationCharacters import segmentationCharacters
from segmentationProvince import segmentationProvince
from segmentationRow import segmentationRow

load_dotenv()
folder_path = os.getenv("FOLDER_PATH")

loaded_images = load_images_from_folder(folder_path)
for img in loaded_images:
    enhance_image = imageEnhancement(img, False)
    data, province = segmentationRow(enhance_image, False)
    charactersCrop = segmentationCharacters(data)
    # provinceCrop = segmentationProvince(province)

# filename = "24_02_01_V00117.jpg"
# file_path = os.path.join(folder_path, filename)
# img = cv2.imread(file_path)
# enhance_image = imageEnhancement(img)
# data, province = segmentationRow(enhance_image)
# charactersCrop = segmentationCharacters(data)
# provinceCrop = segmentationProvince(province)

# folder_path = "../New folder"
# loaded_images = load_images_from_folder(folder_path)
# for img in loaded_images:
#     enhance_image = imageEnhancement(img)
#     data, province = segmentationRow(enhance_image)
# charactersCrop = segmentationCharacters(data)
# provinceCrop = segmentationProvince(province)

from imageEnhancement import imageEnhancement
from loadedImage import loadImageFromFolder
from segmentationCharacters import segmentationCharacters
from segmentationProvince import segmentationProvince
from segmentationRow import segmentationRow

folder_path = "../dataset/20220531"

loaded_images, filenames = loadImageFromFolder(folder_path)
for img, filename in zip(loaded_images, filenames):
    enhance_image = imageEnhancement(img)
    data, province = segmentationRow(enhance_image)
    charactersCrop = segmentationCharacters(
        data, filename, save_path="../output/characters"
    )
    provinceCrop = segmentationProvince(
        province, filename, save_path="../output/province"
    )

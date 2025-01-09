from imageEnhancement import imageEnhancement
from loadedImage import loadImageFromFolder
from resizeImage import resizeImageFix
from runOnnxModel import runOnnxModel
from segmentationCharacters import segmentationCharacters
from segmentationProvince import segmentationProvince
from segmentationRow import segmentationRow

folder_path = "../autoTransformation"
model_path = "model/Characters_MobileNetV3Small.onnx"
characters_class_mapping = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "\u0e01",
    "\u0e02",
    "\u0e03",
    "\u0e04",
    "\u0e05",
    "\u0e06",
    "\u0e07",
    "\u0e08",
    "\u0e09",
    "\u0e0a",
    "\u0e0b",
    "\u0e0c",
    "\u0e0d",
    "\u0e0e",
    "\u0e0f",
    "\u0e10",
    "\u0e11",
    "\u0e12",
    "\u0e13",
    "\u0e14",
    "\u0e15",
    "\u0e16",
    "\u0e17",
    "\u0e18",
    "\u0e19",
    "\u0e1a",
    "\u0e1b",
    "\u0e1c",
    "\u0e1d",
    "\u0e1e",
    "\u0e1f",
    "\u0e20",
    "\u0e21",
    "\u0e22",
    "\u0e23",
    "\u0e25",
    "\u0e27",
    "\u0e28",
    "\u0e29",
    "\u0e2a",
    "\u0e2b",
    "\u0e2c",
    "\u0e2d",
    "\u0e2e",
]
loaded_images, filenames = loadImageFromFolder(folder_path)
for img, filename in zip(loaded_images, filenames):
    print(f"ParkingSlots_ID : {filename}")
    enhance_image = imageEnhancement(img)
    data, province = segmentationRow(enhance_image)

    # charactersCrop = segmentationCharacters(data, filename)
    # characters = []
    # if charactersCrop is not None:
    #     for img in charactersCrop:
    #         if img is not None and img.size > 0:
    #             height = 224
    #             width = height // 3  # 3:1 ratio
    #             img = resizeImageFix(img, width, height)
    #             character = runOnnxModel(
    #                 img,
    #                 model_path,
    #                 characters_class_mapping,
    #             )
    #             characters.append(character)
    # characters = "".join(characters)
    # print(f"License_ID : {characters}")

    provinceCrop = segmentationProvince(province, filename, save_path="../output/province")

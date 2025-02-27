import os
import cv2
import json

import numpy as np

from imageEnhancement import imageEnhancement
from resizeImage import resizeImageFix
from runOnnxModel import runOnnxModel
from segmentationCharacters import segmentationCharacters
from segmentationProvince import segmentationProvince
from segmentationRow import segmentationRow

character_model_path = "model/20250224_Characters_MobileNetV3Small.onnx"
province_model_path = "model/20250204_Province_MobileNetV3Small.onnx"
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
    "ก",
    "ข",
    "ฃ",
    "ค",
    "ฅ",
    "ฆ",
    "ง",
    "จ",
    "ฉ",
    "ช",
    "ซ",
    "ฌ",
    "ญ",
    "ฎ",
    "ฏ",
    "ฐ",
    "ฑ",
    "ฒ",
    "ณ",
    "ด",
    "ต",
    "ถ",
    "ท",
    "ธ",
    "น",
    "บ",
    "ป",
    "ผ",
    "ฝ",
    "พ",
    "ฟ",
    "ภ",
    "ม",
    "ย",
    "ร",
    "ล",
    "ว",
    "ศ",
    "ษ",
    "ส",
    "ห",
    "ฬ",
    "อ",
    "ฮ",
]
province_class_mapping = [
    "เชียงใหม่",
    "นครราชสีมา",
    "กาญจนบุรี",
    "ตาก",
    "อุบลราชธานี",
    "สุราษฎร์ธานี",
    "ชัยภูมิ",
    "แม่ฮ่องสอน",
    "เพชรบูรณ์",
    "ลำปาง",
    "อุดรธานี",
    "เชียงราย",
    "น่าน",
    "เลย",
    "ขอนแก่น",
    "พิษณุโลก",
    "บุรีรัมย์",
    "นครศรีธรรมราช",
    "สกลนคร",
    "นครสวรรค์",
    "ศรีสะเกษ",
    "กำแพงเพชร",
    "ร้อยเอ็ด",
    "สุรินทร์",
    "อุตรดิตถ์",
    "สงขลา",
    "สระแก้ว",
    "กาฬสินธุ์",
    "อุทัยธานี",
    "สุโขทัย",
    "แพร่",
    "ประจวบคีรีขันธ์",
    "จันทบุรี",
    "พะเยา",
    "เพชรบุรี",
    "ลพบุรี",
    "ชุมพร",
    "นครพนม",
    "สุพรรณบุรี",
    "มหาสารคาม",
    "ฉะเชิงเทรา",
    "ราชบุรี",
    "ตรัง",
    "ปราจีนบุรี",
    "กระบี่",
    "พิจิตร",
    "ยะลา",
    "ลำพูน",
    "นราธิวาส",
    "ชลบุรี",
    "มุกดาหาร",
    "บึงกาฬ",
    "พังงา",
    "ยโสธร",
    "หนองบัวลำภู",
    "สระบุรี",
    "ระยอง",
    "พัทลุง",
    "ระนอง",
    "อำนาจเจริญ",
    "หนองคาย",
    "ตราด",
    "พระนครศรีอยุธยา",
    "สตูล",
    "ชัยนาท",
    "นครปฐม",
    "นครนายก",
    "ปัตตานี",
    "กรุงเทพมหานคร",
    "ปทุมธานี",
    "สมุทรปราการ",
    "อ่างทอง",
    "สมุทรสาคร",
    "สิงห์บุรี",
    "นนทบุรี",
    "ภูเก็ต",
    "สมุทรสงคราม",
    "เบตง",
]

folder_path = "../dataset/20250227_hm"
filename = "lp_section3_20250227_190307.jpg"
file_path = os.path.join(folder_path, filename)
img = cv2.imread(file_path)
enhance_image = imageEnhancement(img, True)
data, province = segmentationRow(enhance_image, filename, True)

charactersCrop = segmentationCharacters(data, filename, True)
characters = []
charactersConfident = []
if charactersCrop is not None:
    for img in charactersCrop:
        if img is not None and img.size > 0:
            height = 224
            width = height // 3
            img = resizeImageFix(img, width, height)
            character, characterConfident = runOnnxModel(
                img,
                character_model_path,
                characters_class_mapping,
            )
            characters.append(character)
            charactersConfident.append(characterConfident)
characters = "".join(characters)

provinceCrop = segmentationProvince(province, filename, True)
width = 224
height = width // 3
provinceImage = resizeImageFix(provinceCrop, width, height, True)
province, provinceConfident = runOnnxModel(
    provinceImage,
    province_model_path,
    province_class_mapping,
)
# if provinceConfident > 50:
#     province = province
# else:
#     province = ""

output = {
    "License_ID": characters,
    "CharactersConfident": np.average(charactersConfident),
    "Province": province,
    "ProvinceConfident": provinceConfident,
}

json_output = json.dumps(output, ensure_ascii=False, indent=2)
print(json_output)

import json

from imageEnhancement import imageEnhancement
from loadedImage import loadImageFromFolder
from resizeImage import resizeImageFix
from runOnnxModel import runOnnxModel
from segmentationCharacters import segmentationCharacters
from segmentationProvince import segmentationProvince
from segmentationRow import segmentationRow

folder_path = "../update"
character_model_path = "model/Characters_MobileNetV3Small.onnx"
province_model_path = "model/Province_MobileNetV3Small.onnx"
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
]
loaded_images, filenames = loadImageFromFolder(folder_path)
for img, filename in zip(loaded_images, filenames):
    # print(f"ParkingSlots_ID : {filename}")
    enhance_image = imageEnhancement(img)
    data, province = segmentationRow(enhance_image)

    charactersCrop = segmentationCharacters(data, filename)
    characters = []
    if charactersCrop is not None:
        for img in charactersCrop:
            if img is not None and img.size > 0:
                height = 224
                width = height // 3  # 3:1 ratio
                img = resizeImageFix(img, width, height)
                character, confident = runOnnxModel(
                    img,
                    character_model_path,
                    characters_class_mapping,
                )
                # print(f"character : {character} | confident : {confident:.2f}%")
                characters.append(character)
    characters = "".join(characters)
    # print(f"License_ID : {characters}")

    provinceCrop = segmentationProvince(province, filename)
    width = 224
    height = width // 3  # 1:3 ratio
    if provinceCrop:
        provinceImage = resizeImageFix(provinceCrop[0], width, height)
        province, confident = runOnnxModel(
            provinceImage,
            province_model_path,
            province_class_mapping,
        )
        if confident > 50:
            province = province
        else:
            province = ""
        # print(f"province : {province} | confident : {confident:.2f}%")
        # print(f"Province : {province}")

output = {
    "ParkingSlots_ID": filename,
    "License_ID": characters,
    "Province": province,
}

# Convert results to JSON and print
json_output = json.dumps(output, ensure_ascii=False, indent=2)
print(json_output)

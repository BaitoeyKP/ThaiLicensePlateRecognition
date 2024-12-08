import cv2
from dotenv import load_dotenv
import os

from loadedImage import load_images_from_folder
from displayImageComparison import display_comparison
from segmentationRow import segmentationRow

load_dotenv()
folder_path = os.getenv("FOLDER_PATH")

# Load all images
loaded_images = load_images_from_folder(folder_path)
segmentationRowImages = [segmentationRow(img) for img in loaded_images]

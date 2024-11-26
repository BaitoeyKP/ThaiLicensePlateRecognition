from dotenv import load_dotenv
import os
from loadedImage import load_images_from_folder
from displayImage import display_images

load_dotenv()
folder_path = os.getenv("FOLDER_PATH")

# Load all images
loaded_images = load_images_from_folder(folder_path)

if loaded_images:
    display_images(loaded_images)

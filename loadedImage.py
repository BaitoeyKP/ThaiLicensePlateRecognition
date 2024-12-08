import cv2
import os


def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        # Load the image using cv2
        image = cv2.imread(file_path)  # Reads as a NumPy array
        if image is not None:
            images.append(image)
            print(f"Loaded: {filename}")
            print(f"Size: {image.shape}")
            print("-" * 50)
        else:
            print(f"Failed to load: {filename}")
    return images

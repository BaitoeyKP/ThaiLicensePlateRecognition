import cv2
import os


def loadImageFromFolder(folder_path):
    try:
        images = []
        filenames = []
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            image = cv2.imread(file_path)
            if image is not None:
                base_filename = os.path.splitext(filename)[0]
                images.append(image)
                filenames.append(base_filename)
                # print(f"Loaded: {filename}")
                # print(f"Size: {image.shape}")
                # print("-" * 50)
            else:
                print(f"Failed to load: {filename}")

        return images, filenames

    except Exception as e:
        print(f"Error during loadImageFromFolder: {str(e)}")
        return None, None

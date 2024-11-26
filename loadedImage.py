from PIL import Image
from pathlib import Path
import os


def load_images_from_folder(folder_path):
    """
    Load all images from the specified folder

    Parameters:
    folder_path (str): Path to the folder containing images

    Returns:
    dict: Dictionary of loaded images with their filenames as keys
    """
    try:
        # Check if folder exists
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found at {folder_path}")

        # Common image extensions
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}

        # Dictionary to store images
        images = {}

        # Counter for found images
        total_files = 0
        successful_loads = 0

        # Iterate through all files in the folder
        for filename in os.listdir(folder_path):
            total_files += 1
            file_path = os.path.join(folder_path, filename)

            # Check if the file has an image extension
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                try:
                    # Load the image
                    image = Image.open(file_path)

                    # Store image in dictionary
                    images[filename] = image

                    # Print image information
                    print(f"\nLoaded: {filename}")
                    print(f"Size: {image.size}")
                    print(f"Mode: {image.mode}")
                    print(f"Format: {image.format}")
                    print("-" * 50)

                    successful_loads += 1

                except Exception as e:
                    print(f"Error loading {filename}: {str(e)}")

        # Print summary
        print(f"\nSummary:")
        print(f"Total files found: {total_files}")
        print(f"Images successfully loaded: {successful_loads}")

        return images

    except Exception as e:
        print(f"Error accessing folder: {str(e)}")
        return None

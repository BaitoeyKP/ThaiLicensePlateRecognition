def display_images(images):
    """
    Display all loaded images

    Parameters:
    images (dict): Dictionary of images to display
    """
    if not images:
        print("No images to display")
        return

    for filename, image in images.items():
        print(f"\nDisplaying: {filename}")
        image.show()

import os
from pathlib import Path


def count_files_in_folders(base_path):
    """
    Count files in each subfolder of the specified base path.

    Args:
        base_path (str): Path to the base directory containing character folders

    Returns:
        dict: Dictionary with folder names as keys and file counts as values
    """
    # Convert the base path to a Path object
    base_dir = Path(base_path)

    # Dictionary to store counts
    folder_counts = {}

    try:
        # Iterate through all subfolders
        for folder in base_dir.iterdir():
            if folder.is_dir():
                # Count only files (not directories) in the current folder
                file_count = sum(1 for item in folder.iterdir() if item.is_file())
                # Get the folder name
                folder_name = folder.name
                folder_counts[folder_name] = file_count

        return folder_counts

    except FileNotFoundError:
        print(f"Error: Directory '{base_path}' not found")
        return None
    except PermissionError:
        print(f"Error: Permission denied accessing '{base_path}'")
        return None


def natural_sort_key(s):
    """
    Return a key for natural sorting of strings that may contain Thai characters
    """
    return s


def main():
    # Path to the characters directory
    characters_path = "../label/characters"

    # Get the counts
    counts = count_files_in_folders(characters_path)

    if counts:
        # Print results in a formatted way
        print("\nFile counts in each character folder:")
        print("-" * 50)
        print("Number of Files\t|\tFolder")
        print("-" * 50)

        total_files = 0
        # Sort using the folder names directly
        for folder in sorted(counts.keys(), key=natural_sort_key):
            count = counts[folder]
            # if count < 10:
            print(f"{count}\t|\t{folder}")
            total_files += count

        print("-" * 50)
        print(f"Total files: {total_files}")


if __name__ == "__main__":
    main()

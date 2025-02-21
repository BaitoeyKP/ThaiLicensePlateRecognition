import json

# Load the JSON file from the relative path
file_path = "../Report/all_license_results.json"

try:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Check if the loaded data is a list
    if isinstance(data, list):
        print(len(data))  # Output the length of the list
    else:
        print("Invalid JSON format: Expected a list")
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
except json.JSONDecodeError:
    print("Error: Invalid JSON format")

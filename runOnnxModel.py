import cv2
import onnxruntime as ort
import numpy as np
from PIL import Image


def runOnnxModel(image_input, model_path, class_mapping):
    session = ort.InferenceSession(model_path)

    # Handle both file paths and image objects
    if isinstance(image_input, str):
        # If input is a file path
        image = cv2.imread(image_input)
    elif isinstance(image_input, np.ndarray):
        # If input is already a numpy array (BGR format from OpenCV)
        image = image_input
    elif isinstance(image_input, Image.Image):
        # If input is a PIL Image, convert to OpenCV format
        image = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
    else:
        raise ValueError("Unsupported image input type")

    # Convert BGR to RGB and normalize to [0,1]
    input_data = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

    # Add batch dimension and convert to correct format for ONNX
    input_data = np.transpose(input_data, (2, 0, 1))  # HWC to CHW format
    input_data = np.expand_dims(input_data, axis=0).astype(np.float32)

    # Run inference
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_data.astype(np.float32)})
    probabilities = outputs[0][0]

    # # Get top predictions
    # top_indices = np.argsort(probabilities)[-3:][::-1]  # Get top 5

    # print("\nPredictions:")
    # for idx in top_indices:
    #     class_label = (
    #         class_mapping[idx] if idx < len(class_mapping) else f"Unknown({idx})"
    #     )
    #     print(
    #         f"Index {idx}: Class {class_label}, "
    #         f"Confidence: {probabilities[idx]:.2f}"
    #     )
    
    # 8.5

    result = class_mapping[np.argmax(probabilities)]

    return result

import cv2
import onnxruntime as ort
import numpy as np
from PIL import Image


def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()


def runOnnxModel(image_input, model_path, class_mapping):
    try:
        session = ort.InferenceSession(model_path)

        if isinstance(image_input, str):
            image = cv2.imread(image_input)
        elif isinstance(image_input, np.ndarray):
            image = image_input
        elif isinstance(image_input, Image.Image):
            image = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
        else:
            # raise ValueError("Unsupported image input type")
            return "", 0

        input_data = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        input_data = np.transpose(input_data, (2, 0, 1))
        input_data = np.expand_dims(input_data, axis=0).astype(np.float32)

        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: input_data.astype(np.float32)})
        logits = outputs[0][0]

        probabilities = softmax(logits)

        top_indices = np.argsort(probabilities)[-3:][::-1]

        # print("\nPredictions:")
        # for idx in top_indices:
        #     class_label = (
        #         class_mapping[idx] if idx < len(class_mapping) else f"Unknown({idx})"
        #     )
        #     print(
        #         f"Index {idx}: Class {class_label}, "
        #         f"Confidence: {probabilities[idx]*100:.2f}%"
        #     )

        class_label = class_mapping[top_indices[0]]
        confident = probabilities[top_indices[0]] * 100

        return class_label, confident

    except Exception as e:
        print(f"Error during runOnnxModel: {str(e)}")
        return "", 0

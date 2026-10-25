
# generate_models.py

import pickle
import numpy as np
import os

os.makedirs("./models", exist_ok=True)


class MockFaceRecognitionModel:
    def predict(self, image_tensor):
        if not isinstance(image_tensor, np.ndarray):
            raise TypeError(
                f"Model Error: Expected numpy.ndarray, got {type(image_tensor)}"
            )

        if image_tensor.shape != (1, 224, 224, 3):
            raise ValueError(
                f"Model Error: Invalid input shape. Expected "
                f"(1, 224, 224, 3), got {image_tensor.shape}"
            )

        return {"label": "authorized", "confidence": 0.98}


class MockObjectDetectionModel:
    def predict(self, image_tensor):
        if not isinstance(image_tensor, np.ndarray):
            raise TypeError(
                f"Model Error: Expected numpy.ndarray, got {type(image_tensor)}"
            )

        if image_tensor.shape != (1, 224, 224, 3):
            raise ValueError(
                f"Model Error: Invalid input shape. Expected "
                f"(1, 224, 224, 3), got {image_tensor.shape}"
            )

        return [
            {"label": "person", "confidence": 0.95},
            {"label": "knife", "confidence": 0.88},
            {"label": "backpack", "confidence": 0.75},
        ]


face_model = MockFaceRecognitionModel()
object_model = MockObjectDetectionModel()

with open("./models/face_recognition_model.pkl", "wb") as f:
    pickle.dump(face_model, f)

with open("./models/object_detection_model.pkl", "wb") as f:
    pickle.dump(object_model, f)

print("Mock models generated successfully in ./models/")

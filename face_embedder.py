import os
import cv2
import numpy as np
import onnxruntime


class FaceEmbedder:
    def __init__(self, model_path: str, input_size=(112, 112)):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Face embedding model not found at {model_path}")

        self.model_path = model_path
        self.input_size = input_size
        self.session = onnxruntime.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            raise ValueError("Empty image provided to embedder")

        face = cv2.resize(image, self.input_size, interpolation=cv2.INTER_LINEAR)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = face.astype(np.float32) / 255.0
        face = (face - 0.5) * 2.0
        face = np.transpose(face, (2, 0, 1))
        face = np.expand_dims(face, axis=0)
        return face

    def encode(self, face_image: np.ndarray) -> np.ndarray:
        tensor = self.preprocess(face_image)
        result = self.session.run([self.output_name], {self.input_name: tensor})
        embedding = np.array(result[0], dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

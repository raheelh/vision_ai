# Models

This project requires two models:

1. `models/face_detector.onnx`
   - DeepStream-compatible face detection model.
   - Common candidates are ONNX face detectors converted to TensorRT engines or used via `nvinfer`.

2. `models/face_recognizer.onnx`
   - A face embedding model such as a small ArcFace or FaceNet ONNX model.
   - The model should output a fixed-length embedding vector for a cropped face image.

Place the trained ONNX files in this directory before running the app.

If you need to produce a TensorRT engine for DeepStream, use the DeepStream model optimizer or TensorRT conversion tools.

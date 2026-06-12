# Vision AI Face Recognition with NVIDIA DeepStream

A DeepStream-based face recognition pipeline using camera input from a PC.

Features:
- Real-time face detection using NVIDIA DeepStream
- Face recognition with a simple `chromadb` vector database
- Bounding boxes and numeric labels overlayed on each detected face
- New faces are added to the vector DB and the best 5 embeddings are kept per person
- Reuses stored vectors to recognize people in future frames

## Contents
- `main.py` - entry point for the application
- `deepstream_face_recognizer.py` - DeepStream pipeline construction and metadata handling
- `face_embedder.py` - ONNX face embedding extractor
- `face_database.py` - simple `chromadb` backed vector storage and matching logic
- `config/face_detector_config.txt` - DeepStream detector config template
- `config/labels.txt` - label file for detector output
- `models/README.md` - model placement and expected files

## Requirements
- NVIDIA GPU with DeepStream SDK installed
- Python 3.10+ recommended
- `chromadb`, `onnxruntime`, `opencv-python`, `numpy`
- `pyds` and GStreamer Python bindings installed by DeepStream

## Install
1. Install DeepStream and confirm `pyds` is available.
2. Install Python dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run
```bash
python main.py --camera-index 0 --model-path models/face_recognizer.onnx
```

If your camera or device path differs, use `--camera-index` or update `config/face_detector_config.txt`.

For a working example, copy `config/face_detector_config_example.txt` to `config/face_detector_config.txt` and update `onnx-file` to the face detector model you are using.

To initialize the template automatically, run:

```bash
python setup_config.py
```

To create placeholder model files, run:

```bash
python setup_config.py --create-model-placeholders
```

Use `python setup_config.py --force --create-model-placeholders` to overwrite existing config and model placeholders.

## Notes
- Replace the placeholder models in `models/` with your own DeepStream-compatible face detector and recognition models.
- Example model names: `models/face_detector_resnet50.onnx` and `models/face_recognizer.onnx`.
- The first unknown face detected is added to the local ChromaDB store.
- The app retains only the best 5 saved vectors per person to keep matching performance stable.
- Faces are tracked across frames using a lightweight motion-based tracker, and each track receives a `T<ID>` label prefix.

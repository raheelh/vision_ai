import argparse
from deepstream_face_recognizer import DeepStreamFaceRecognizer
from face_embedder import FaceEmbedder
from face_database import FaceDatabase


def parse_args():
    parser = argparse.ArgumentParser(description="DeepStream face recognition with ChromaDB storage")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera device index")
    parser.add_argument("--model-path", default="models/face_recognizer.onnx", help="ONNX face embedding model path")
    parser.add_argument("--detector-config", default="config/face_detector_config.txt", help="DeepStream face detector configuration file")
    parser.add_argument("--db-path", default="db/chromadb", help="Local ChromaDB persistence directory")
    parser.add_argument("--threshold", type=float, default=0.55, help="Matching threshold for face embeddings")
    parser.add_argument("--max-vectors-per-person", type=int, default=5, help="Maximum vectors to keep per person")
    parser.add_argument("--preview-width", type=int, default=1280, help="Preview window width")
    parser.add_argument("--preview-height", type=int, default=720, help="Preview window height")
    return parser.parse_args()


def main():
    args = parse_args()
    face_db = FaceDatabase(
        persist_path=args.db_path,
        max_vectors_per_person=args.max_vectors_per_person,
        match_threshold=args.threshold,
    )

    embedder = FaceEmbedder(model_path=args.model_path)

    app = DeepStreamFaceRecognizer(
        detector_config=args.detector_config,
        face_embedder=embedder,
        face_db=face_db,
        camera_index=args.camera_index,
        preview_width=args.preview_width,
        preview_height=args.preview_height,
    )

    app.run()


if __name__ == "__main__":
    main()

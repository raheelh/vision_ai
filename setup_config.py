import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Copy example DeepStream config into active config file")
    parser.add_argument(
        "--source",
        default="config/face_detector_config_example.txt",
        help="Path to the example DeepStream config",
    )
    parser.add_argument(
        "--destination",
        default="config/face_detector_config.txt",
        help="Target DeepStream config file to write",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the destination file if it already exists",
    )
    parser.add_argument(
        "--create-model-placeholders",
        action="store_true",
        help="Create placeholder model files in the models directory",
    )
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Directory where placeholder model files are created",
    )
    args = parser.parse_args()

    source = Path(args.source)
    destination = Path(args.destination)
    if not source.exists():
        raise FileNotFoundError(f"Example config not found: {source}")

    if destination.exists() and not args.force:
        print(f"Destination already exists: {destination}")
        print("Use --force to overwrite the existing config.")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        print(f"Copied {source} to {destination}")

    if args.create_model_placeholders:
        _create_model_placeholders(Path(args.models_dir), force=args.force)


def _create_model_placeholders(models_dir: Path, force: bool = False):
    models_dir.mkdir(parents=True, exist_ok=True)
    placeholder_text = (
        "# Placeholder file for {name}\n"
        "# Replace this file with a valid DeepStream-compatible ONNX model.\n"
    )

    for model_name in ["face_detector.onnx", "face_recognizer.onnx"]:
        model_path = models_dir / model_name
        if model_path.exists() and not force:
            print(f"Model placeholder already exists: {model_path}")
            continue
        model_path.write_text(placeholder_text.format(name=model_name), encoding="utf-8")
        print(f"Created placeholder model: {model_path}")


if __name__ == "__main__":
    main()

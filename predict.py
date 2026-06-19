from __future__ import annotations

import argparse
from pathlib import Path

from fashion_mnist_utils import (
    BackgroundMode,
    PredictionMode,
    load_cnn_model,
    load_image,
    predict_image,
    top_k_predictions,
)


def print_predictions(image_path: Path, mode: PredictionMode, background: BackgroundMode, threshold: float, otsu: bool) -> None:
    """Run one image through the saved CNN and print confidence analysis."""
    model = load_cnn_model()
    image = load_image(image_path)
    probabilities, preprocess_result = predict_image(
        model=model,
        image=image,
        mode=mode,
        background=background,
        cleanup_threshold=threshold,
        use_otsu=otsu,
    )

    top3 = top_k_predictions(probabilities, k=3)
    predicted_class, confidence = top3[0]

    print(f"Image: {image_path}")
    print(f"Model type: {getattr(model, 'model_type', 'cnn')}")
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence * 100:.2f}%")
    print("Top 3 predictions:")
    for class_name, probability in top3:
        print(f"- {class_name}: {probability * 100:.2f}%")

    print("Preprocessing:")
    print(f"- Mode: {mode}")
    print(f"- Background option: {background}")
    print(f"- Inverted: {preprocess_result.was_inverted}")
    print(f"- Otsu thresholding used: {preprocess_result.used_otsu}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a Fashion-MNIST class from an image.")
    parser.add_argument("image", type=Path, help="Path to the image to classify.")
    parser.add_argument(
        "--mode",
        choices=["fashion_mnist", "real_photo", "auto"],
        default="auto",
        help="Preprocessing mode.",
    )
    parser.add_argument(
        "--background",
        choices=["auto", "light", "dark"],
        default="auto",
        help="Background assumption for inversion.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.18,
        help="Cleanup threshold used when Otsu is disabled or unavailable.",
    )
    parser.add_argument(
        "--no-otsu",
        action="store_true",
        help="Disable OpenCV Otsu thresholding.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        print_predictions(
            image_path=args.image,
            mode=args.mode,
            background=args.background,
            threshold=args.threshold,
            otsu=not args.no_otsu,
        )
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
    except ValueError as exc:
        print(f"Error: {exc}")
    except RuntimeError as exc:
        print(f"Error: {exc}")

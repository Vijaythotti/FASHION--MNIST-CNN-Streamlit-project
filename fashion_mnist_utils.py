from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, UnidentifiedImageError

try:
    import cv2
except ImportError:  # OpenCV is optional at import time; Otsu falls back gracefully.
    cv2 = None


CLASS_NAMES: list[str] = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "fashion_mnist_cnn.keras"
LEGACY_MODEL_PATH = MODEL_DIR / "fashion_mnist_softmax.npz"
OUTPUT_DIR = Path("outputs")

PredictionMode = Literal["fashion_mnist", "real_photo", "auto"]
BackgroundMode = Literal["auto", "light", "dark"]


@dataclass(frozen=True)
class PreprocessResult:
    """Container for the model-ready tensor and user-facing debug images."""

    tensor: np.ndarray
    processed_image: Image.Image
    grayscale_image: Image.Image
    foreground_crop: Image.Image
    background_mode: BackgroundMode
    was_inverted: bool
    used_otsu: bool


def load_image(path_or_file: str | Path | object) -> Image.Image:
    """Load an image and normalize EXIF orientation with a clear error message."""
    try:
        image = Image.open(path_or_file)
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The uploaded file is not a valid image.") from exc
    return ImageOps.exif_transpose(image).convert("RGB")


def detect_background(gray_array: np.ndarray) -> bool:
    """Return True when the image border indicates a light background."""
    border = np.concatenate(
        [
            gray_array[0, :],
            gray_array[-1, :],
            gray_array[:, 0],
            gray_array[:, -1],
        ]
    )
    return float(np.median(border)) > 0.5


def enhance_grayscale(image: Image.Image) -> Image.Image:
    """Convert to grayscale, stretch contrast, and make edges clearer."""
    gray = ImageOps.grayscale(image)
    gray = ImageOps.autocontrast(gray)
    gray = ImageEnhance.Contrast(gray).enhance(1.7)
    return gray.filter(ImageFilter.SHARPEN)


def should_invert(gray_array: np.ndarray, background: BackgroundMode) -> bool:
    if background == "light":
        return True
    if background == "dark":
        return False
    return detect_background(gray_array)


def create_foreground_mask(
    array: np.ndarray,
    cleanup_threshold: float,
    use_otsu: bool,
) -> tuple[np.ndarray, bool]:
    """Create a foreground mask from a normalized grayscale image."""
    if use_otsu and cv2 is not None:
        uint8 = np.uint8(np.clip(array, 0.0, 1.0) * 255)
        _, thresholded = cv2.threshold(uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = thresholded > 0
        return mask, True

    threshold = float(np.clip(cleanup_threshold, 0.01, 0.95))
    return array > threshold, False


def crop_to_foreground(image: Image.Image, mask: np.ndarray, padding: int = 6) -> Image.Image:
    """Crop around detected foreground pixels; fall back to the whole image."""
    if not mask.any():
        return image

    ys, xs = np.where(mask)
    left = max(int(xs.min()) - padding, 0)
    upper = max(int(ys.min()) - padding, 0)
    right = min(int(xs.max()) + padding + 1, image.width)
    lower = min(int(ys.max()) + padding + 1, image.height)
    return image.crop((left, upper, right, lower))


def resize_and_center(image: Image.Image, canvas_size: int = 28, object_size: int = 24) -> Image.Image:
    """Resize while preserving aspect ratio and paste into the center of a 28x28 canvas."""
    crop = image.copy()
    crop.thumbnail((object_size, object_size), Image.Resampling.LANCZOS)

    canvas = Image.new("L", (canvas_size, canvas_size), 0)
    left = (canvas_size - crop.width) // 2
    top = (canvas_size - crop.height) // 2
    canvas.paste(crop, (left, top))
    return canvas


def infer_preprocess_mode(image: Image.Image, mode: PredictionMode) -> PredictionMode:
    """Use Fashion-MNIST preprocessing for tiny grayscale inputs, real-photo otherwise."""
    if mode != "auto":
        return mode

    rgb = image.convert("RGB")
    array = np.asarray(rgb, dtype=np.float32)
    color_spread = float(np.mean(np.max(array, axis=2) - np.min(array, axis=2)))
    looks_like_dataset = image.width <= 64 and image.height <= 64 and color_spread < 8.0
    return "fashion_mnist" if looks_like_dataset else "real_photo"


def preprocess_for_cnn(
    image: Image.Image,
    mode: PredictionMode = "auto",
    background: BackgroundMode = "auto",
    cleanup_threshold: float = 0.18,
    use_otsu: bool = True,
) -> PreprocessResult:
    """Prepare any uploaded image for the Fashion-MNIST CNN.

    The CNN expects a single-channel 28x28 image with bright clothing pixels on a
    dark background. This function makes real photos closer to that training
    distribution without changing how Fashion-MNIST test images are handled.
    """
    selected_mode = infer_preprocess_mode(image, mode)
    gray = enhance_grayscale(image)

    max_side = 420 if selected_mode == "real_photo" else 96
    gray.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    array = np.asarray(gray, dtype=np.float32) / 255.0

    inverted = should_invert(array, background)
    if inverted:
        array = 1.0 - array
        gray = Image.fromarray(np.uint8(array * 255), mode="L")

    mask, used_otsu = create_foreground_mask(
        array=array,
        cleanup_threshold=cleanup_threshold,
        use_otsu=use_otsu and selected_mode == "real_photo",
    )

    crop = crop_to_foreground(gray, mask)
    processed = resize_and_center(crop)
    tensor = np.asarray(processed, dtype=np.float32).reshape(1, 28, 28, 1) / 255.0

    return PreprocessResult(
        tensor=tensor,
        processed_image=processed,
        grayscale_image=gray,
        foreground_crop=crop,
        background_mode=background,
        was_inverted=inverted,
        used_otsu=used_otsu,
    )


def top_k_predictions(probabilities: np.ndarray, k: int = 3) -> list[tuple[str, float]]:
    """Return top-k class names and probabilities from a model output vector."""
    vector = np.asarray(probabilities).reshape(-1)
    top_indices = vector.argsort()[::-1][:k]
    return [(CLASS_NAMES[int(index)], float(vector[index])) for index in top_indices]


def real_photo_prior(image: Image.Image) -> np.ndarray:
    """Estimate a simple prior for real photos when only the legacy model is available.

    This is not a replacement for the CNN. It keeps the app usable on real product
    photos in environments where TensorFlow cannot be installed yet.
    """
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    rgb.thumbnail((512, 512), Image.Resampling.LANCZOS)
    gray = np.asarray(rgb.convert("L"), dtype=np.float32) / 255.0

    border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
    background = float(np.median(border))
    if background > 0.5:
        mask = (background - gray) > 0.18
    else:
        mask = (gray - background) > 0.18

    mask_image = Image.fromarray(np.uint8(mask) * 255, mode="L")
    mask_image = mask_image.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(5))
    mask = np.asarray(mask_image) > 0

    scores = np.full(len(CLASS_NAMES), 0.01, dtype=np.float32)
    if not mask.any():
        return scores / scores.sum()

    ys, xs = np.where(mask)
    bbox_width = max(1, int(xs.max() - xs.min() + 1))
    bbox_height = max(1, int(ys.max() - ys.min() + 1))
    height, width = mask.shape
    aspect = bbox_width / bbox_height
    bbox_area = float((bbox_width * bbox_height) / max(1, width * height))
    bottom_position = float((ys.max() + 1) / height)
    top_position = float(ys.min() / height)
    fill_ratio = float(mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1].mean())

    long_horizontal = float(np.clip((aspect - 1.15) / 0.9, 0.0, 1.0))
    lower_object = float(np.clip((bottom_position - 0.55) / 0.35, 0.0, 1.0))
    raised_top = float(np.clip((top_position - 0.18) / 0.28, 0.0, 1.0))
    tall = float(np.clip((1.25 - aspect) / 0.75, 0.0, 1.0))

    footwear_shape = long_horizontal * lower_object * raised_top
    upper_body_shape = float(
        np.clip((0.28 - top_position) / 0.28, 0.0, 1.0)
        * np.clip((bottom_position - 0.62) / 0.32, 0.0, 1.0)
        * np.clip((1.65 - aspect) / 0.9, 0.0, 1.0)
        * np.clip((bbox_area - 0.18) / 0.42, 0.0, 1.0)
    )

    scores[CLASS_NAMES.index("Sneaker")] += 1.05 * footwear_shape + 0.25 * long_horizontal
    scores[CLASS_NAMES.index("Sandal")] += 0.35 * footwear_shape + 0.20 * (1.0 - fill_ratio)
    scores[CLASS_NAMES.index("Ankle boot")] += 0.25 * footwear_shape + 0.25 * tall * lower_object
    scores[CLASS_NAMES.index("Bag")] += 0.25 * tall + 0.08 * (1.0 - long_horizontal)
    scores[CLASS_NAMES.index("Shirt")] += 0.18 * tall * (1.0 - lower_object)
    scores[CLASS_NAMES.index("T-shirt/top")] += 0.14 * tall * (1.0 - lower_object)

    if upper_body_shape > 0.20 and footwear_shape < 0.25:
        scores[CLASS_NAMES.index("T-shirt/top")] += 1.30 * upper_body_shape
        scores[CLASS_NAMES.index("Shirt")] += 0.38 * upper_body_shape
        scores[CLASS_NAMES.index("Pullover")] += 0.24 * upper_body_shape
        scores[CLASS_NAMES.index("Coat")] += 0.12 * upper_body_shape
        scores[CLASS_NAMES.index("Bag")] *= 0.18
        scores[CLASS_NAMES.index("Sandal")] *= 0.35
        scores[CLASS_NAMES.index("Ankle boot")] *= 0.35
    return scores / scores.sum()


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


class LegacySoftmaxModel:
    """Compatibility adapter for the lightweight NumPy model used before CNN training."""

    model_type = "legacy_softmax"

    def __init__(self, weights: np.ndarray, bias: np.ndarray) -> None:
        self.weights = weights
        self.bias = bias

    def predict(self, tensor: np.ndarray, verbose: int = 0) -> np.ndarray:
        flat = np.asarray(tensor, dtype=np.float32).reshape(tensor.shape[0], -1)
        return _softmax(flat @ self.weights + self.bias)


def _load_legacy_model(model_path: Path = LEGACY_MODEL_PATH) -> LegacySoftmaxModel:
    if not model_path.exists():
        raise FileNotFoundError(
            f"No model found. Expected either {MODEL_PATH} or {LEGACY_MODEL_PATH}."
        )
    data = np.load(model_path)
    return LegacySoftmaxModel(weights=data["weights"], bias=data["bias"])


def load_cnn_model(model_path: Path = MODEL_PATH):
    """Load the trained Keras CNN, falling back to the legacy NumPy model if needed."""
    if not model_path.exists():
        return _load_legacy_model()

    try:
        from tensorflow.keras.models import load_model
    except ImportError as exc:
        if LEGACY_MODEL_PATH.exists():
            return _load_legacy_model()
        raise RuntimeError("TensorFlow is not installed and no legacy fallback model was found.") from exc

    model = load_model(model_path)
    model.model_type = "cnn"
    return model


def predict_image(
    model,
    image: Image.Image,
    mode: PredictionMode = "auto",
    background: BackgroundMode = "auto",
    cleanup_threshold: float = 0.18,
    use_otsu: bool = True,
) -> tuple[np.ndarray, PreprocessResult]:
    """Preprocess an image and run model inference."""
    result = preprocess_for_cnn(
        image=image,
        mode=mode,
        background=background,
        cleanup_threshold=cleanup_threshold,
        use_otsu=use_otsu,
    )
    probabilities = model.predict(result.tensor, verbose=0)[0]
    selected_mode = infer_preprocess_mode(image, mode)
    if getattr(model, "model_type", "cnn") == "legacy_softmax" and selected_mode == "real_photo":
        prior = real_photo_prior(image)
        probabilities = (0.15 * probabilities) + (0.85 * prior)
        probabilities = probabilities / probabilities.sum()
    return probabilities, result

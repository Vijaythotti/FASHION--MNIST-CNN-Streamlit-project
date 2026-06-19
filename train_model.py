from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

from fashion_mnist_utils import CLASS_NAMES, MODEL_PATH, OUTPUT_DIR


def prepare_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load Fashion-MNIST and return normalized tensors plus one-hot labels."""
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    y_train_cat = to_categorical(y_train, num_classes=len(CLASS_NAMES))
    y_test_cat = to_categorical(y_test, num_classes=len(CLASS_NAMES))
    return x_train, y_train, y_train_cat, x_test, y_test, y_test_cat


def build_model() -> Sequential:
    """Build the deeper CNN requested for Fashion-MNIST."""
    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(28, 28, 1)),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation="relu", padding="same"),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            Conv2D(64, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation="relu", padding="same"),
            MaxPooling2D((2, 2)),
            Dropout(0.30),
            Flatten(),
            Dense(256, activation="relu"),
            Dropout(0.5),
            Dense(len(CLASS_NAMES), activation="softmax"),
        ]
    )
    model.compile(
        optimizer=Adam(),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_augmenter() -> ImageDataGenerator:
    """Create augmentation that simulates small real-photo positioning changes."""
    return ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
    )


def save_history_plots(history, output_dir: Path) -> None:
    """Save accuracy and loss plots from Keras training history."""
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Train accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation accuracy")
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_plot.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Train loss")
    plt.plot(history.history["val_loss"], label="Validation loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_plot.png", dpi=160)
    plt.close()

    with (output_dir / "training_history.json").open("w", encoding="utf-8") as handle:
        json.dump({key: [float(value) for value in values] for key, values in history.history.items()}, handle, indent=2)


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_dir: Path) -> None:
    """Save a confusion matrix image and raw CSV values."""
    matrix = confusion_matrix(y_true, y_pred)
    np.savetxt(output_dir / "confusion_matrix.csv", matrix, delimiter=",", fmt="%d")

    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(CLASS_NAMES))
    plt.xticks(tick_marks, CLASS_NAMES, rotation=45, ha="right")
    plt.yticks(tick_marks, CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=180)
    plt.close()


def save_classification_report(y_true: np.ndarray, y_pred: np.ndarray, output_dir: Path) -> None:
    """Save sklearn's per-class precision, recall, and F1 report."""
    report_text = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
    report_dict = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)

    (output_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")
    with (output_dir / "classification_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report_dict, handle, indent=2)


def save_sample_predictions(
    x_test: np.ndarray,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    output_dir: Path,
    count: int = 25,
) -> None:
    """Save a grid of sample test predictions for quick visual QA."""
    predicted = probabilities.argmax(axis=1)
    sample_count = min(count, len(x_test))

    plt.figure(figsize=(12, 12))
    for index in range(sample_count):
        plt.subplot(5, 5, index + 1)
        plt.imshow(x_test[index].squeeze(), cmap="gray")
        color = "green" if predicted[index] == y_true[index] else "red"
        title = f"P: {CLASS_NAMES[predicted[index]]}\nT: {CLASS_NAMES[y_true[index]]}"
        plt.title(title, color=color, fontsize=8)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / "sample_predictions.png", dpi=180)
    plt.close()


def train_model(epochs: int, batch_size: int) -> None:
    """Train, evaluate, and save the CNN plus all requested artifacts."""
    np.random.seed(42)
    x_train, _y_train_raw, y_train, x_test, y_test_raw, y_test = prepare_dataset()
    model = build_model()
    augmenter = build_augmenter()

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        augmenter.flow(x_train, y_train, batch_size=batch_size),
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=1,
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    probabilities = model.predict(x_test, verbose=0)
    y_pred = probabilities.argmax(axis=1)

    save_history_plots(history, OUTPUT_DIR)
    save_confusion_matrix(y_test_raw, y_pred, OUTPUT_DIR)
    save_classification_report(y_test_raw, y_pred, OUTPUT_DIR)
    save_sample_predictions(x_test, y_test_raw, probabilities, OUTPUT_DIR)

    metadata = {
        "model_path": str(MODEL_PATH),
        "epochs_requested": epochs,
        "batch_size": batch_size,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "classes": CLASS_NAMES,
    }
    with (OUTPUT_DIR / "model_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved evaluation outputs: {OUTPUT_DIR}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an augmented Fashion-MNIST CNN.")
    parser.add_argument("--epochs", type=int, default=20, help="Maximum training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(epochs=args.epochs, batch_size=args.batch_size)

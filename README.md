# Fashion-MNIST CNN + Streamlit

This project trains a deeper CNN on Fashion-MNIST and serves predictions through a Streamlit app. It supports both classic Fashion-MNIST test images and real-world fashion photos by applying a stronger preprocessing pipeline before inference.

## Setup

Use Python 3.10, 3.11, or 3.12 for best TensorFlow compatibility.

```bash
python3 -m pip install -r requirements.txt
```

## Train

```bash
python3 train_model.py
```

The model is saved to:

```text
models/fashion_mnist_cnn.keras
```

Training uses:

- Conv2D(32) + BatchNormalization
- Conv2D(32)
- MaxPooling2D
- Dropout
- Conv2D(64) + BatchNormalization
- Conv2D(64)
- MaxPooling2D
- Dropout
- Flatten
- Dense(256)
- Dropout(0.5)
- Dense(10, softmax)
- Adam optimizer
- 20 epochs
- EarlyStopping
- ReduceLROnPlateau
- ImageDataGenerator augmentation with rotation, zoom, width shift, and height shift

## Evaluation Outputs

Training writes these files into `outputs/`:

- `accuracy_plot.png`
- `loss_plot.png`
- `confusion_matrix.png`
- `confusion_matrix.csv`
- `classification_report.txt`
- `classification_report.json`
- `sample_predictions.png`
- `training_history.json`
- `model_metadata.json`

## Predict From CLI

```bash
python3 predict.py path/to/image.jpg
```

Optional controls:

```bash
python3 predict.py path/to/image.jpg --mode real_photo --background light --threshold 0.18
python3 predict.py path/to/image.jpg --mode fashion_mnist --background dark --no-otsu
```

## Run Streamlit

```bash
python3 -m streamlit run app.py
```

The app displays:

- Original image
- Final processed 28x28 image actually fed into the CNN
- Predicted class
- Confidence percentage
- Top 3 predictions with probabilities
- Probability bar chart
- Enhanced grayscale and detected object crop

## Preprocessing Pipeline

Real photos and uploaded images are normalized before prediction:

1. Load image and fix EXIF orientation.
2. Convert to RGB, then grayscale.
3. Enhance contrast with autocontrast and contrast boosting.
4. Detect background brightness from border pixels.
5. Automatically invert images when the background is light.
6. Optionally apply OpenCV Otsu thresholding.
7. Detect the foreground object.
8. Crop around the object.
9. Resize while preserving aspect ratio.
10. Center the object on a black 28x28 canvas.
11. Feed the final `(1, 28, 28, 1)` tensor into the CNN.

## Prediction Modes

- `Fashion-MNIST Mode`: best for actual 28x28 dataset-style images.
- `Real Photo Mode`: best for product photos, phone photos, and images with backgrounds.
- `Auto Detect Mode`: chooses preprocessing based on image size and color characteristics.

Fashion-MNIST is still a small grayscale dataset, so real-world photos can remain challenging. The preprocessing pipeline reduces that gap, but the highest real-photo accuracy requires fine-tuning on a real fashion image dataset.


import os
import io
import logging
from threading import Lock

import cv2
import numpy as np
import tensorflow as tf

from PIL import Image, UnidentifiedImageError
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO


# ==========================================================
# 1. FLASK CONFIGURATION
# ==========================================================

app = Flask(__name__, template_folder="templates")

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")


# ==========================================================
# 2. EXACT MODEL PATHS
# ==========================================================

MODEL_PATHS = {
    "yolo": os.path.join(MODEL_DIR, "yolov8n.pt"),
    "behavior": os.path.join(
        MODEL_DIR, "driver_behavior_cnn.h5"
    ),
    "lane": os.path.join(
        MODEL_DIR, "lane_detection_unet.h5"
    ),
    "sign": os.path.join(
        MODEL_DIR, "traffic_sign_cnn.h5"
    ),
}


# ==========================================================
# 3. CLASS LABELS
# ==========================================================

DRIVER_STATES = [
    "Safe Driving",
    "Texting (Right)",
    "Talking on Phone (Right)",
    "Texting (Left)",
    "Talking on Phone (Left)",
    "Operating Radio",
    "Drinking",
    "Reaching Behind",
    "Fixing Hair/Makeup",
    "Talking to Passenger",
]

# Optional: define your actual traffic-sign class names
# using a Render environment variable named SIGN_CLASSES.
#
# Example value:
# Stop,Speed Limit,No Entry,Turn Left,Turn Right
#
# If not configured, the app returns the predicted class index.
SIGN_CLASSES = [
    name.strip()
    for name in os.environ.get("SIGN_CLASSES", "").split(",")
    if name.strip()
]


# ==========================================================
# 4. MODEL STATE
# ==========================================================

models = {
    "yolo": None,
    "behavior": None,
    "lane": None,
    "sign": None,
}

model_errors = {}
model_lock = Lock()
models_attempted = False


# ==========================================================
# 5. KERAS COMPATIBILITY HELPERS
# ==========================================================

class CompatibleDense(tf.keras.layers.Dense):
    @classmethod
    def from_config(cls, config):
        config = config.copy()
        config.pop("quantization_config", None)
        return super().from_config(config)


class CompatibleRescaling(tf.keras.layers.Rescaling):
    @classmethod
    def from_config(cls, config):
        config = config.copy()

        dtype = config.get("dtype")
        if isinstance(dtype, dict):
            dtype_config = dtype.get("config", {})
            config["dtype"] = (
                dtype_config.get("name", "float32")
                if isinstance(dtype_config, dict)
                else "float32"
            )

        return super().from_config(config)


class FixedInputLayer(tf.keras.layers.InputLayer):
    @classmethod
    def from_config(cls, config):
        config = config.copy()

        # Handle older serialized InputLayer configurations.
        if "batch_shape" in config and "batch_input_shape" not in config:
            config["batch_input_shape"] = config.pop("batch_shape")

        config.pop("optional", None)

        return super().from_config(config)


CUSTOM_OBJECTS = {
    "Dense": CompatibleDense,
    "Rescaling": CompatibleRescaling,
    "InputLayer": FixedInputLayer,
}


# ==========================================================
# 6. MODEL LOADING
# ==========================================================

def load_models():
    """
    Attempt to load each model separately.
    A failure in one model does not prevent the other
    models from being attempted.
    """
    global models_attempted

    with model_lock:
        if models_attempted:
            return

        models_attempted = True

        # A. YOLO object detection
        try:
            path = MODEL_PATHS["yolo"]

            if not os.path.isfile(path):
                raise FileNotFoundError(path)

            logger.info("Loading YOLO model...")
            models["yolo"] = YOLO(path)
            logger.info("YOLO model loaded successfully.")

        except Exception as exc:
            model_errors["yolo"] = str(exc)
            logger.exception("YOLO model loading failed.")

        # B, C and D. TensorFlow models
        #
        # Load one at a time and release temporary references.
        # This does not guarantee that the models fit in RAM.

        for key in ("behavior", "lane", "sign"):
            try:
                path = MODEL_PATHS[key]

                if not os.path.isfile(path):
                    raise FileNotFoundError(path)

                logger.info("Loading %s model...", key)

                model = tf.keras.models.load_model(
                    path,
                    custom_objects=CUSTOM_OBJECTS,
                    compile=False,
                )

                models[key] = model

                logger.info(
                    "%s model loaded. Input: %s, Output: %s",
                    key,
                    model.input_shape,
                    model.output_shape,
                )

            except Exception as exc:
                model_errors[key] = str(exc)
                logger.exception(
                    "%s model loading failed.", key
                )


# ==========================================================
# 7. IMAGE PREPROCESSING
# ==========================================================

def get_input_size(model, fallback=(224, 224)):
    """Return model width and height from its input shape."""
    shape = model.input_shape

    if isinstance(shape, list):
        shape = shape[0]

    if len(shape) != 4:
        raise ValueError(
            f"Expected a 4D image input, got {shape}"
        )

    height = shape[1]
    width = shape[2]

    if height is None or width is None:
        return fallback

    return int(width), int(height)


def has_rescaling_layer(model):
    """Check whether the model contains a Keras Rescaling layer."""
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Rescaling):
            return True

        # Check nested Sequential/Functional models.
        if isinstance(
            layer,
            (tf.keras.Sequential, tf.keras.Model),
        ):
            if has_rescaling_layer(layer):
                return True

    return False


def prepare_image(image, model, normalize=True):
    """
    Prepare an RGB PIL image for a Keras image classifier.

    Models containing a Rescaling layer receive 0-255 input.
    Otherwise, normalize=True scales pixels to 0-1.

    Confirm that this matches the preprocessing used during
    your models' training.
    """
    width, height = get_input_size(model)

    array = np.asarray(
        image.resize((width, height), Image.Resampling.BILINEAR),
        dtype=np.float32,
    )

    if normalize and not has_rescaling_layer(model):
        array /= 255.0

    return np.expand_dims(array, axis=0)


def predict_single(model, image, normalize=True):
    """Run inference and return one output array."""
    batch = prepare_image(
        image,
        model,
        normalize=normalize,
    )

    prediction = model.predict(batch, verbose=0)

    if isinstance(prediction, (list, tuple)):
        if len(prediction) != 1:
            raise ValueError("Multiple model outputs are unsupported.")
        prediction = prediction[0]

    prediction = np.asarray(prediction)

    if prediction.size == 0:
        raise ValueError("The model returned an empty prediction.")

    return prediction


# ==========================================================
# 8. OBJECT DETECTION — YOLOv8
# ==========================================================

def detect_objects(image):
    model = models["yolo"]

    if model is None:
        raise RuntimeError(
            model_errors.get("yolo", "YOLO is unavailable.")
        )

    results = model.predict(
        source=image,
        imgsz=640,
        conf=0.25,
        verbose=False,
        device="cpu",
    )

    vehicle_count = 0
    person_count = 0
    detected_objects = []

    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            label = results[0].names.get(
                class_id, str(class_id)
            )

            detected_objects.append({
                "label": label,
                "confidence": round(confidence * 100, 2),
            })

            # COCO labels used by the standard YOLOv8n model.
            if class_id in (2, 3, 5, 7):
                vehicle_count += 1
            elif class_id == 0:
                person_count += 1

    return {
        "status": "success",
        "vehicles": vehicle_count,
        "pedestrians": person_count,
        "objects": detected_objects,
        "summary": (
            f"Vehicles: {vehicle_count}, "
            f"Pedestrians: {person_count}"
        ),
    }


# ==========================================================
# 9. DRIVER BEHAVIOR CLASSIFICATION
# ==========================================================

def detect_behavior(image):
    model = models["behavior"]

    if model is None:
        raise RuntimeError(
            model_errors.get(
                "behavior", "Driver behavior model is unavailable."
            )
        )

    # Preserve the original preprocessing assumption:
    # the model's internal Rescaling layer handles normalization,
    # if it has one.
    prediction = predict_single(
        model,
        image,
        normalize=False,
    )

    probabilities = prediction[0].reshape(-1)

    class_idx = int(np.argmax(probabilities))
    confidence = float(probabilities[class_idx]) * 100

    if class_idx >= len(DRIVER_STATES):
        raise ValueError(
            f"Unexpected behavior class index: {class_idx}"
        )

    label = DRIVER_STATES[class_idx]

    return {
        "status": "success",
        "class_index": class_idx,
        "label": label,
        "confidence": round(confidence, 2),
        "summary": f"{label} ({confidence:.1f}%)",
    }


# ==========================================================
# 10. LANE DETECTION — U-NET
# ==========================================================

def detect_lanes(image):
    model = models["lane"]

    if model is None:
        raise RuntimeError(
            model_errors.get(
                "lane", "Lane detection model is unavailable."
            )
        )

    prediction = predict_single(
        model,
        image,
        normalize=True,
    )

    # Common U-Net output: [1, height, width, 1].
    # Also supports multi-channel segmentation outputs.
    mask = prediction[0]

    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]

    if mask.ndim == 3 and mask.shape[-1] > 1:
        # Multi-class segmentation: class zero is assumed
        # to be background.
        mask = np.argmax(mask, axis=-1)
        lane_pixels = mask != 0
    elif mask.ndim == 2:
        # Binary segmentation. Assumes the model outputs
        # probabilities or a binary mask.
        lane_pixels = mask > 0.5
    else:
        raise ValueError(
            f"Unsupported lane output shape: {prediction.shape}"
        )

    lane_coverage = float(np.mean(lane_pixels) * 100)

    if np.any(lane_pixels):
        summary = (
            f"Lane mask generated; "
            f"{lane_coverage:.1f}% of output pixels "
            "classified as lane"
        )
    else:
        summary = "No lane pixels detected in the output mask"

    return {
        "status": "success",
        "lane_pixels_percent": round(lane_coverage, 2),
        "output_width": int(mask.shape[1]),
        "output_height": int(mask.shape[0]),
        "summary": summary,
    }


# ==========================================================
# 11. TRAFFIC SIGN CLASSIFICATION
# ==========================================================

def detect_signs(image):
    model = models["sign"]

    if model is None:
        raise RuntimeError(
            model_errors.get(
                "sign", "Traffic sign model is unavailable."
            )
        )

    prediction = predict_single(
        model,
        image,
        normalize=True,
    )

    probabilities = prediction[0].reshape(-1)

    class_idx = int(np.argmax(probabilities))
    confidence = float(probabilities[class_idx]) * 100

    if SIGN_CLASSES and class_idx < len(SIGN_CLASSES):
        label = SIGN_CLASSES[class_idx]
    else:
        label = f"Class {class_idx}"

    return {
        "status": "success",
        "class_index": class_idx,
        "label": label,
        "confidence": round(confidence, 2),
        "summary": (
            f"{label} ({confidence:.1f}%)"
            if SIGN_CLASSES
            else (
                f"Predicted class index {class_idx} "
                f"({confidence:.1f}%). Configure SIGN_CLASSES "
                "to display your trained class names."
            )
        ),
    }


# ==========================================================
# 12. HOME PAGE
# ==========================================================

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


# ==========================================================
# 13. HEALTH CHECK
# ==========================================================

@app.route("/health", methods=["GET"])
def health():
    loaded = {
        key: value is not None
        for key, value in models.items()
    }

    return jsonify({
        "status": "ok",
        "models_loaded": all(loaded.values()),
        "models": loaded,
        "model_errors": model_errors,
    }), 200


# ==========================================================
# 14. ANALYZE BOTH IMAGES
# ==========================================================

@app.route("/analyze", methods=["POST"])
def analyze():
    dashcam_file = request.files.get("dashcam")
    cabin_file = request.files.get("cabin")

    if dashcam_file is None or cabin_file is None:
        return jsonify({
            "error": "Please upload both dashcam and cabin images."
        }), 400

    if not dashcam_file.filename or not cabin_file.filename:
        return jsonify({
            "error": "Please select both image files."
        }), 400

    try:
        dashcam_bytes = dashcam_file.read()
        cabin_bytes = cabin_file.read()

        if not dashcam_bytes or not cabin_bytes:
            return jsonify({
                "error": "One or both uploaded files are empty."
            }), 400

        with Image.open(io.BytesIO(dashcam_bytes)) as source:
            dashcam_img = source.convert("RGB")

        with Image.open(io.BytesIO(cabin_bytes)) as source:
            cabin_img = source.convert("RGB")

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
    ):
        return jsonify({
            "error": "One or both uploaded files are invalid images."
        }), 400

    # Model loading is attempted once per process.
    try:
        load_models()
    except Exception:
        logger.exception("Model initialization failed")
        return jsonify({
            "error": "Model initialization failed.",
            "model_errors": model_errors,
        }), 500

    results = {}

    # Run each model independently so one failed model
    # does not suppress results from the others.
    tasks = [
        ("objects", detect_objects, dashcam_img),
        ("behavior", detect_behavior, cabin_img),
        ("sign", detect_signs, dashcam_img),
        ("lane", detect_lanes, dashcam_img),
    ]

    for key, function, image in tasks:
        try:
            logger.info("Running %s inference...", key)
            results[key] = function(image)

        except Exception as exc:
            logger.exception("%s inference failed.", key)
            results[key] = {
                "status": "error",
                "error": str(exc),
            }

    successful = sum(
        value.get("status") == "success"
        for value in results.values()
    )

    # Keep the main result keys and summary strings compatible
    # with a frontend expecting objects, behavior, sign and lane.
    return jsonify({
        "status": (
            "success" if successful == 4
            else "partial" if successful > 0
            else "error"
        ),
        "objects": results["objects"].get(
            "summary", results["objects"].get("error", "Unavailable")
        ),
        "behavior": results["behavior"].get(
            "summary", results["behavior"].get("error", "Unavailable")
        ),
        "sign": results["sign"].get(
            "summary", results["sign"].get("error", "Unavailable")
        ),
        "lane": results["lane"].get(
            "summary", results["lane"].get("error", "Unavailable")
        ),
        "details": results,
    }), 200 if successful > 0 else 503


# ==========================================================
# 15. ERROR HANDLERS
# ==========================================================

@app.errorhandler(413)
def request_too_large(error):
    return jsonify({
        "error": "The total uploaded request exceeds 10 MB."
    }), 413


@app.errorhandler(500)
def internal_server_error(error):
    logger.exception("Internal server error")
    return jsonify({
        "error": "An internal server error occurred."
    }), 500


# ==========================================================
# 16. START SERVER
# ==========================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
    )

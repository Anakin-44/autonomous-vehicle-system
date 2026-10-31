
import os
import io
import logging

import cv2
import numpy as np
import tensorflow as tf

from PIL import Image, UnidentifiedImageError
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO


# ==================================================
# 1. FLASK CONFIGURATION
# ==================================================

app = Flask(__name__, template_folder="templates")

# Maximum total request size: 10 MB
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================================================
# 2. MODEL PATHS
# ==================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

YOLO_PATH = os.path.join(
    BASE_DIR, "models", "yolov8n.pt"
)

BEHAVIOR_PATH = os.path.join(
    BASE_DIR, "models", "driver_behavior_cnn.h5"
)


# ==================================================
# 3. DRIVER BEHAVIOR CLASSES
# ==================================================

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


# ==================================================
# 4. KERAS COMPATIBILITY LAYERS
# ==================================================

class CompatibleDense(tf.keras.layers.Dense):
    """
    Handle a newer Keras Dense configuration when the
    quantization_config field is unsupported by the runtime.
    """

    @classmethod
    def from_config(cls, config):
        config = config.copy()

        # This is safe only for an ordinary, non-quantized
        # model whose saved weights are already usable.
        config.pop("quantization_config", None)

        return super().from_config(config)


class CompatibleRescaling(tf.keras.layers.Rescaling):
    """
    Handle serialized dtype policies from newer Keras versions.
    """

    @classmethod
    def from_config(cls, config):
        config = config.copy()

        dtype = config.get("dtype")

        if isinstance(dtype, dict):
            dtype_config = dtype.get("config", {})

            if isinstance(dtype_config, dict):
                config["dtype"] = dtype_config.get(
                    "name", "float32"
                )
            else:
                config["dtype"] = "float32"

        return super().from_config(config)


class FixedInputLayer(tf.keras.layers.InputLayer):
    """
    Compatibility handling for older InputLayer configurations.
    """

    @classmethod
    def from_config(cls, config):
        config = config.copy()

        if "batch_shape" in config:
            config["batch_input_shape"] = config.pop(
                "batch_shape"
            )

        config.pop("optional", None)

        return super().from_config(config)


# ==================================================
# 5. LOAD DEEP LEARNING MODELS
# ==================================================

def load_models():
    """Load YOLO and the driver behavior classifier."""

    if not os.path.isfile(YOLO_PATH):
        raise FileNotFoundError(
            f"YOLO model not found: {YOLO_PATH}"
        )

    if not os.path.isfile(BEHAVIOR_PATH):
        raise FileNotFoundError(
            f"Driver behavior model not found: {BEHAVIOR_PATH}"
        )

    logger.info("Loading YOLO model...")

    yolo = YOLO(YOLO_PATH)

    logger.info("Loading driver behavior model...")

    behavior = tf.keras.models.load_model(
        BEHAVIOR_PATH,
        custom_objects={
            "Dense": CompatibleDense,
            "Rescaling": CompatibleRescaling,
            "InputLayer": FixedInputLayer,
        },
        compile=False,
    )

    # Verify the model's output class count.
    output_shape = behavior.output_shape

    if isinstance(output_shape, list):
        raise ValueError(
            "The driver behavior model has multiple outputs. "
            "Update the inference code to handle them."
        )

    output_classes = output_shape[-1]

    if output_classes != len(DRIVER_STATES):
        raise ValueError(
            f"Model output classes: {output_classes}; "
            f"expected: {len(DRIVER_STATES)}."
        )

    logger.info(
        "Driver behavior model loaded. Output shape: %s",
        output_shape,
    )

    logger.info("All required models loaded successfully!")

    return yolo, behavior


# Load once when the application starts.
yolo_model, behavior_model = load_models()


# ==================================================
# 6. HOME PAGE
# ==================================================

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


# ==================================================
# 7. HEALTH CHECK
# ==================================================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "yolo_loaded": yolo_model is not None,
        "behavior_model_loaded": behavior_model is not None,
    }), 200


# ==================================================
# 8. ANALYZE UPLOADED IMAGES
# ==================================================

@app.route("/analyze", methods=["POST"])
def analyze():

    dashcam_file = request.files.get("dashcam")
    cabin_file = request.files.get("cabin")

    if dashcam_file is None or cabin_file is None:
        return jsonify({
            "error": (
                "Please upload both dashcam and cabin images."
            )
        }), 400

    if not dashcam_file.filename or not cabin_file.filename:
        return jsonify({
            "error": "Please select both image files."
        }), 400

    # ----------------------------------------------
    # Validate and decode images
    # ----------------------------------------------

    try:
        dashcam_img = Image.open(
            io.BytesIO(dashcam_file.read())
        ).convert("RGB")

        cabin_img = Image.open(
            io.BytesIO(cabin_file.read())
        ).convert("RGB")

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
    ):
        return jsonify({
            "error": "One or both uploaded files are invalid images."
        }), 400

    try:
        # ==========================================
        # A. OBJECT DETECTION — YOLOv8
        # ==========================================

        logger.info("Running YOLO object detection...")

        yolo_results = yolo_model(
            dashcam_img,
            verbose=False,
        )

        vehicle_count = 0
        person_count = 0

        for box in yolo_results[0].boxes:
            class_id = int(box.cls[0])

            # COCO class IDs:
            # 0 = person
            # 2 = car
            # 3 = motorcycle
            # 5 = bus
            # 7 = truck

            if class_id in [2, 3, 5, 7]:
                vehicle_count += 1

            elif class_id == 0:
                person_count += 1

        obj_text = (
            f"Vehicles: {vehicle_count}, "
            f"Pedestrians: {person_count}"
        )

        # ==========================================
        # B. DRIVER BEHAVIOR CLASSIFICATION
        # ==========================================

        logger.info("Running driver behavior classification...")

        cabin_array = np.array(cabin_img)

        cabin_array = cv2.resize(
            cabin_array,
            (224, 224),
        )

        cabin_array = np.expand_dims(
            cabin_array,
            axis=0,
        )

        # IMPORTANT:
        # This assumes the model's Rescaling layer handles
        # normalization internally, as in the original model.

        predictions = behavior_model.predict(
            cabin_array,
            verbose=0,
        )

        class_idx = int(np.argmax(predictions[0]))

        confidence = (
            float(np.max(predictions[0])) * 100
        )

        if not 0 <= class_idx < len(DRIVER_STATES):
            raise ValueError(
                f"Unexpected class index: {class_idx}"
            )

        behavior_text = (
            f"{DRIVER_STATES[class_idx]} "
            f"({confidence:.1f}%)"
        )

        # ==========================================
        # C. RETURN RESULTS
        # ==========================================

        return jsonify({
            "objects": obj_text,
            "behavior": behavior_text,

            # These are placeholders, not actual predictions.
            "sign": "Traffic Sign model integration pending",
            "lane": "Lane Detection model integration pending",
        }), 200

    except Exception:
        logger.exception("Image analysis failed")

        return jsonify({
            "error": (
                "Analysis failed. Please check the server logs."
            )
        }), 500


# ==================================================
# 9. ERROR HANDLERS
# ==================================================

@app.errorhandler(413)
def request_too_large(error):
    return jsonify({
        "error": (
            "Uploaded files exceed the 10 MB total request limit."
        )
    }), 413


@app.errorhandler(500)
def internal_server_error(error):
    logger.error("Internal server error: %s", error)

    return jsonify({
        "error": "An internal server error occurred."
    }), 500


# ==================================================
# 10. APPLICATION ENTRY POINT
# ==================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
    )

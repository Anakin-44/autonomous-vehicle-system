
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

# Maximum total request size: 10 MB
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==========================================================
# 2. MODEL PATHS
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

YOLO_PATH = os.path.join(
    BASE_DIR, "models", "yolov8n.pt"
)

BEHAVIOR_PATH = os.path.join(
    BASE_DIR, "models", "driver_behavior_cnn.h5"
)


# ==========================================================
# 3. DRIVER BEHAVIOR CLASSES
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


# ==========================================================
# 4. MODEL STATE
# ==========================================================

# Models are initialized only when first needed.
yolo_model = None
behavior_model = None

# Prevent simultaneous initialization by multiple requests.
model_lock = Lock()


# ==========================================================
# 5. KERAS COMPATIBILITY LAYERS
# ==========================================================

class CompatibleDense(tf.keras.layers.Dense):
    """Handle unsupported quantization_config metadata."""

    @classmethod
    def from_config(cls, config):
        config = config.copy()
        config.pop("quantization_config", None)

        return super().from_config(config)


class CompatibleRescaling(tf.keras.layers.Rescaling):
    """Handle serialized dtype policies from newer Keras."""

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
    """Handle certain older/newer InputLayer configurations."""

    @classmethod
    def from_config(cls, config):
        config = config.copy()

        if "batch_shape" in config:
            config["batch_input_shape"] = config.pop(
                "batch_shape"
            )

        config.pop("optional", None)

        return super().from_config(config)


# ==========================================================
# 6. LOAD MODELS
# ==========================================================

def load_models():
    """Load and validate both deep-learning models."""

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

    logger.info("YOLO model loaded successfully.")
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

    output_shape = behavior.output_shape

    if isinstance(output_shape, list):
        raise ValueError(
            "The behavior model has multiple outputs. "
            "The current inference code expects one output."
        )

    output_classes = output_shape[-1]

    if output_classes != len(DRIVER_STATES):
        raise ValueError(
            f"Model outputs {output_classes} classes, "
            f"but {len(DRIVER_STATES)} classes are configured."
        )

    logger.info(
        "Driver behavior model loaded. Output shape: %s",
        output_shape,
    )

    logger.info("All required models loaded successfully!")

    return yolo, behavior


# ==========================================================
# 7. LAZY MODEL INITIALIZATION
# ==========================================================

def get_models():
    """
    Initialize models on the first analysis request.
    Subsequent requests reuse the loaded models.
    """

    global yolo_model, behavior_model

    if yolo_model is not None and behavior_model is not None:
        return yolo_model, behavior_model

    with model_lock:

        # Check again after acquiring the lock.
        if yolo_model is None or behavior_model is None:
            logger.info("Initializing deep-learning models...")

            # Assign only after both models load successfully.
            yolo, behavior = load_models()

            yolo_model = yolo
            behavior_model = behavior

            logger.info("Model initialization completed.")

    return yolo_model, behavior_model


# ==========================================================
# 8. HOME PAGE
# ==========================================================

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


# ==========================================================
# 9. HEALTH CHECK
# ==========================================================

@app.route("/health", methods=["GET"])
def health():

    models_loaded = (
        yolo_model is not None
        and behavior_model is not None
    )

    return jsonify({
        "status": "ok",
        "models_loaded": models_loaded,
        "yolo_loaded": yolo_model is not None,
        "behavior_model_loaded": behavior_model is not None,
    }), 200


# ==========================================================
# 10. ANALYZE IMAGES
# ==========================================================

@app.route("/analyze", methods=["POST"])
def analyze():

    # ----------------------------------------------
    # Validate uploaded files
    # ----------------------------------------------

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

    # ----------------------------------------------
    # Read images
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

    # ----------------------------------------------
    # Initialize models when needed
    # ----------------------------------------------

    try:
        yolo, behavior = get_models()

    except Exception:
        logger.exception("Model initialization failed")

        return jsonify({
            "error": (
                "The deep-learning models could not be loaded. "
                "Please check the server logs."
            )
        }), 500

    # ----------------------------------------------
    # Run inference
    # ----------------------------------------------

    try:

        # ==========================================
        # A. OBJECT DETECTION — YOLOv8
        # ==========================================

        logger.info("Running YOLO object detection...")

        yolo_results = yolo(
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

        # The model's own Rescaling layer is expected
        # to perform normalization, as in training.

        predictions = behavior.predict(
            cabin_array,
            verbose=0,
        )

        class_idx = int(np.argmax(predictions[0]))

        confidence = float(
            np.max(predictions[0])
        ) * 100

        if not 0 <= class_idx < len(DRIVER_STATES):
            raise ValueError(
                f"Unexpected behavior class index: {class_idx}"
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

            # These two outputs remain placeholders.
            "sign": "Traffic Sign model integration pending",
            "lane": "Lane Detection model integration pending",
        }), 200

    except Exception:
        logger.exception("Image analysis failed")

        return jsonify({
            "error": (
                "Image analysis failed. "
                "Please check the server logs."
            )
        }), 500


# ==========================================================
# 11. ERROR HANDLERS
# ==========================================================

@app.errorhandler(413)
def request_too_large(error):
    return jsonify({
        "error": (
            "The uploaded files exceed the 10 MB "
            "total request size limit."
        )
    }), 413


@app.errorhandler(500)
def internal_server_error(error):
    logger.error("Internal server error: %s", error)

    return jsonify({
        "error": "An internal server error occurred."
    }), 500


# ==========================================================
# 12. LOCAL DEVELOPMENT ENTRY POINT
# ==========================================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
    )
